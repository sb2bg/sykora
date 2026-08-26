# Train Sykora's rank-32 moment-factorised P3-ANOVA adapter on Stockfish binpacks.
#
# Phase 1 (1..50) keeps the mature v8 FT and first material affine bit-exact.
# Phase 2 resumes the complete optimiser state and fine-tunes the full graph.

param(
    [switch]$DryRun,
    [string]$DataDir = "D:\datasets\chess\stockfish_test80_binpacks",
    [string]$OutputRoot = "D:\training\sykora",
    [string]$WarmStart = "",
    [string]$RunTag = "",
    [int]$AdapterEndSuperbatch = 50,
    [int]$EndSuperbatch = 800,
    [double]$Wdl = 0.25
)

$ErrorActionPreference = "Stop"

if ($AdapterEndSuperbatch -lt 1 -or $EndSuperbatch -le $AdapterEndSuperbatch) {
    Write-Error "Require 1 <= AdapterEndSuperbatch < EndSuperbatch"
    exit 2
}
if ($Wdl -lt 0.0 -or $Wdl -gt 1.0) {
    Write-Error "-Wdl must be in [0, 1]"
    exit 2
}

# Windows CUDA toolchain pinned by the existing Sykora launcher.
$msvcVer = "14.44.35207"
$sdkVer = "10.0.26100.0"
$cudaVer = "12.6"
$msvcRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\$msvcVer"
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVer"
$env:PATH = "$msvcRoot\bin\Hostx64\x64;$sdkRoot\bin\$sdkVer\x64;$cudaRoot\bin;$env:LOCALAPPDATA\Programs\Python\Python312;$env:LOCALAPPDATA\Programs\Python\Python312\Scripts;$env:USERPROFILE\.cargo\bin;$env:PATH"
$env:LIB = "$msvcRoot\lib\x64;$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"
$env:INCLUDE = "$msvcRoot\include;$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared"
$env:CUDA_PATH = $cudaRoot
$env:CUDARC_CUDA_VERSION = "12060"

$python = @(
    "$PSScriptRoot\.venv\Scripts\python.exe",
    "$PSScriptRoot\nnue\.venv\Scripts\python.exe"
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) {
    Write-Error "Could not find a Python virtualenv under .venv or nnue\.venv"
    exit 1
}

if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    Write-Error "Stockfish binpack directory not found: $DataDir"
    exit 1
}
$dataRoot = (Resolve-Path -LiteralPath $DataDir).Path
$heldOut = Get-ChildItem -LiteralPath $dataRoot -File -Filter "test80-2024-06-*.binpack" |
    Sort-Object FullName |
    Select-Object -First 1
if (-not $heldOut) {
    Write-Error "Could not identify the June 2024 held-out SF binpack under $dataRoot"
    exit 1
}
$trainingDatasets = @(
    Get-ChildItem -LiteralPath $dataRoot -File -Filter "*.binpack" |
        Where-Object { $_.FullName -ne $heldOut.FullName } |
        Sort-Object FullName |
        ForEach-Object FullName
)
if ($trainingDatasets.Count -eq 0) {
    Write-Error "No training binpacks remain after reserving $($heldOut.Name)"
    exit 1
}

$validationSample = Join-Path $dataRoot "validation\t80_2024_06_v3filter_262144.data"
if (-not (Test-Path -LiteralPath $validationSample -PathType Leaf)) {
    Write-Error "Held-out validation sample not found: $validationSample"
    exit 1
}

if (-not $WarmStart) {
    $WarmStart = Join-Path $PSScriptRoot "nnue\models\bullet\v8_t1024_finetune_20260725T194431Z\checkpoints\v8_t1024_finetune_20260725T194431Z-1000"
}
if (-not (Test-Path -LiteralPath $WarmStart -PathType Container)) {
    Write-Error "Mature v8 warm-start checkpoint not found: $WarmStart"
    exit 1
}

if (-not $RunTag) {
    $RunTag = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
}
$phase1Id = "p3_anova_r32_adapter_$RunTag"
$phase2Id = "p3_anova_r32_full_$RunTag"
$phase1Checkpoint = Join-Path $OutputRoot "$phase1Id\checkpoints\$phase1Id-$AdapterEndSuperbatch"
$runner = Join-Path $PSScriptRoot "utils\nnue\bullet\train_cuda_longrun.py"
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$wdlArg = $Wdl.ToString("R", $invariantCulture)

$common = @(
    "--bullet-repo", (Join-Path $PSScriptRoot "nnue\bullet_repo"),
    "--output-root", $OutputRoot,
    "--data-format", "binpack",
    "--binpack-buffer-mb", "12288",
    "--binpack-threads", "6",
    "--validation-sample", $validationSample,
    "--validation-positions", "262144",
    "--network-format", "syk9",
    "--architecture", "pairwise-mlp-p3",
    "--p3-rank", "32",
    "--bucket-layout", "v3_10",
    "--hidden", "1024",
    "--dense1", "16",
    "--dense2", "32",
    "--output-buckets", "8",
    "--batch-size", "16384",
    "--batches-per-superbatch", "6104",
    "--save-rate", "10",
    "--threads", "8",
    "--wdl", $wdlArg,
    "--no-validate-after"
)

$phase1Args = @(
    "-u", $runner, "--dataset"
) + $trainingDatasets + $common + @(
    "--run-id", $phase1Id,
    "--start-superbatch", "1",
    "--end-superbatch", $AdapterEndSuperbatch.ToString(),
    "--lr-start", "0.0003",
    "--lr-final", "0.0003",
    "--lr-origin-superbatch", "1",
    "--lr-final-superbatch", $AdapterEndSuperbatch.ToString(),
    "--p3-freeze-base",
    "--warm-start", $WarmStart
)

$phase2Start = $AdapterEndSuperbatch + 1
$phase2Args = @(
    "-u", $runner, "--dataset"
) + $trainingDatasets + $common + @(
    "--run-id", $phase2Id,
    "--start-superbatch", $phase2Start.ToString(),
    "--end-superbatch", $EndSuperbatch.ToString(),
    "--lr-start", "0.0001",
    "--lr-final", "0.00001",
    "--lr-origin-superbatch", $phase2Start.ToString(),
    "--lr-final-superbatch", $EndSuperbatch.ToString(),
    "--resume", $phase1Checkpoint
)

Write-Host "Sykora P3-ANOVA moment-factorised training"
Write-Host "  run tag:       $RunTag"
Write-Host "  training:      $($trainingDatasets.Count) SF binpacks"
Write-Host "  held out:      $($heldOut.Name)"
Write-Host "  validation:    $validationSample"
Write-Host "  WDL:           $wdlArg"
Write-Host "  phase 1:       1..$AdapterEndSuperbatch, FT/l1 frozen, LR 0.0003"
Write-Host "  phase 2:       $phase2Start..$EndSuperbatch, full graph, LR 0.0001 -> 0.00001"
Write-Host "  output:        $OutputRoot"

if ($DryRun) {
    Write-Host "PHASE1> $python $($phase1Args -join ' ')"
    Write-Host "PHASE2> $python $($phase2Args -join ' ')"
    exit 0
}

& $python @phase1Args
if ($LASTEXITCODE -ne 0) {
    Write-Error "P3 adapter phase failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $phase1Checkpoint -PathType Container)) {
    Write-Error "P3 adapter phase did not produce its transition checkpoint: $phase1Checkpoint"
    exit 1
}

& $python @phase2Args
exit $LASTEXITCODE
