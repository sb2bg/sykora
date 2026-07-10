# Train, export, and verify Sykora's pairwise-MLP SYKNNUE7 network.
#
# First run:
#   .\launch_training.ps1 -Smoke
#
# Full run:
#   .\launch_training.ps1

param(
    [switch]$Smoke,
    [switch]$DryRun,
    [string]$Resume = "",
    [int]$StartSuperbatch = 0
)

$ErrorActionPreference = "Stop"

# --- Windows CUDA toolchain ---
$msvcVer = "14.44.35207"
$sdkVer = "10.0.26100.0"
$cudaVer = "12.6"
$cudaDigits = "12060"
$msvcRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\$msvcVer"
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVer"

$env:PATH = "$msvcRoot\bin\Hostx64\x64;$sdkRoot\bin\$sdkVer\x64;$cudaRoot\bin;$env:LOCALAPPDATA\Programs\Python\Python312;$env:LOCALAPPDATA\Programs\Python\Python312\Scripts;$env:USERPROFILE\.cargo\bin;$env:PATH"
$env:LIB = "$msvcRoot\lib\x64;$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"
$env:INCLUDE = "$msvcRoot\include;$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared"
$env:CUDA_PATH = $cudaRoot
$env:CUDARC_CUDA_VERSION = $cudaDigits

$venvActivate = @(
    "$PSScriptRoot\.venv\Scripts\Activate.ps1",
    "$PSScriptRoot\nnue\.venv\Scripts\Activate.ps1"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $venvActivate) {
    Write-Error "Could not find a Python virtualenv under .venv or nnue\.venv"
    exit 1
}
& $venvActivate

# --- Stockfish pretraining data ---
$dataDir = "$PSScriptRoot\nnue\data\binpack"
$trainBinpacks = @(
    "test80-2023-06-jun-2tb7p.min-v2.v6.binpack",
    "test80-2023-07-jul-2tb7p.min-v2.v6.binpack",
    "test80-2023-09-sep-2tb7p.min-v2.v6.binpack",
    "test80-2023-10-oct-2tb7p.min-v2.v6.binpack",
    "test80-2023-11-nov-2tb7p.min-v2.v6.binpack",
    "test80-2023-12-dec-2tb7p.min-v2.v6.binpack",
    "test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
    "test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
    "test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
    "test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
    "test80-2024-05-may-2tb7p.min-v2.v6.binpack"
)
$validationBinpacks = @(
    "test80-2024-06-jun-2tb7p.min-v2.v6.binpack"
)

function Resolve-Binpacks([string[]]$Names) {
    $resolved = @()
    foreach ($name in $Names) {
        $path = Join-Path $dataDir $name
        if (-not (Test-Path $path)) {
            Write-Error "Missing: $path`nDecompress with: zstd -d `"$path.zst`""
            exit 1
        }
        $resolved += (Resolve-Path $path).Path
    }
    return $resolved
}

$trainingDatasets = Resolve-Binpacks $trainBinpacks
$validationDatasets = Resolve-Binpacks $validationBinpacks

# --- Registered SYKNNUE7 profile ---
$hidden = 1024
$dense1 = 16
$dense2 = 32
$outputBuckets = 8
$endSuperbatch = 800
$batchSize = 16384
$batchesPerSuperbatch = 6104
$saveRate = 25
$validationPositions = 262144
if ($Smoke) {
    $endSuperbatch = 2
    $batchSize = 4096
    $batchesPerSuperbatch = 16
    $saveRate = 1
    $validationPositions = 16384
}

$validationCache = Join-Path $dataDir "validation\t80_2024_06_v3filter_$validationPositions.data"
if ($StartSuperbatch -le 0) {
    $StartSuperbatch = 1
    if ($Resume) {
        $resumeName = Split-Path ($Resume -replace '[\\/]+$', '') -Leaf
        if ($resumeName -match '-(\d+)$') {
            $StartSuperbatch = [int]$Matches[1] + 1
        }
    }
}
if ($StartSuperbatch -gt $endSuperbatch) {
    Write-Error "Start superbatch $StartSuperbatch exceeds end $endSuperbatch"
    exit 2
}

$timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$runPrefix = if ($Smoke) { "smoke_v7" } else { "v7" }
$runId = "${runPrefix}_${timestamp}"

Write-Host "============================================"
Write-Host "  Sykora SYKNNUE7 training"
Write-Host "============================================"
Write-Host "Run ID:        $runId"
Write-Host "Architecture:  factorised pairwise-MLP"
Write-Host "Shape:         H=$hidden, $hidden -> $dense1 -> $($dense1 * 2) -> $dense2 -> 1"
Write-Host "Output heads:  $outputBuckets material buckets"
Write-Host "Superbatches:  $StartSuperbatch -> $endSuperbatch"
Write-Host "Batch shape:   $batchSize x $batchesPerSuperbatch"
Write-Host "Train shards:  $($trainingDatasets.Count)"
Write-Host "Held out:      $($validationDatasets.Count) shard, $validationPositions positions"
Write-Host "============================================"

$arguments = @(
    "$PSScriptRoot\utils\nnue\bullet\train_cuda_longrun.py",
    "--dataset"
) + $trainingDatasets + @(
    "--validation-dataset"
) + $validationDatasets + @(
    "--validation-cache", $validationCache,
    "--validation-positions", $validationPositions,
    "--bullet-repo", "$PSScriptRoot\nnue\bullet_repo",
    "--output-root", "$PSScriptRoot\nnue\models\bullet",
    "--run-id", $runId,
    "--data-format", "binpack",
    "--binpack-buffer-mb", 12288,
    "--binpack-threads", 6,
    "--validation-buffer-mb", 512,
    "--network-format", "syk7",
    "--architecture", "pairwise-mlp",
    "--bucket-layout", "v3_10",
    "--hidden", $hidden,
    "--dense1", $dense1,
    "--dense2", $dense2,
    "--output-buckets", $outputBuckets,
    "--start-superbatch", $StartSuperbatch,
    "--end-superbatch", $endSuperbatch,
    "--batch-size", $batchSize,
    "--batches-per-superbatch", $batchesPerSuperbatch,
    "--save-rate", $saveRate,
    "--threads", 8,
    "--wdl", 0.75,
    "--lr-start", 0.001,
    "--export-after"
)
if ($Resume) {
    $arguments += @("--resume", $Resume)
}
if ($DryRun) {
    $arguments += "--dry-run"
} else {
    zig build -Doptimize=ReleaseFast
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $engine = "$PSScriptRoot\zig-out\bin\sykora.exe"
    $arguments += @("--parity-engine", $engine)
}

python @arguments
exit $LASTEXITCODE
