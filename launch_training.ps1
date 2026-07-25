# Train, export, and verify Sykora's registered SYKNNUE8 network.
#
# Dataset roots use this layout:
#   <root>\train\*.binpack       (or *.data)
#   <root>\validation\*.binpack  (or one *.data validation sample)
#
# First v8 pipeline check (random init is diagnostic-only):
#   .\launch_training.ps1 -DataDir D:\nnue-data\smoke -Smoke -AllowRandomV8Init
#
# T1024 pilot from the retained v7 full-precision checkpoint:
#   .\launch_training.ps1 -DataDir D:\nnue-data\broad -WarmStart <v7-checkpoint-directory>
#
# Lower-learning-rate fine-tune from a full-precision v8 checkpoint:
#   .\launch_training.ps1 -Stage finetune -DataDir D:\nnue-data\finetune -Resume <v8-checkpoint-directory>

param(
    [switch]$Smoke,
    [switch]$DryRun,
    [ValidateSet("v8-t1024", "v8-t768")]
    [string]$Profile = "v8-t1024",
    [ValidateSet("pilot", "broad", "finetune")]
    [string]$Stage = "pilot",
    [string]$DataDir = "",
    [string]$Resume = "",
    [string]$WarmStart = "",
    [switch]$AllowRandomV8Init,
    [int]$StartSuperbatch = 0,
    [int]$Superbatches = 0,
    [double]$Wdl = 0.75,
    [double]$LrStart = 0.0,
    [double]$LrFinal = 0.0
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

# --- Source-agnostic local training data ---
if (-not $DataDir) {
    Write-Error (
        "-DataDir is required. Expected <root>\train and <root>\validation " +
        "subdirectories containing .binpack or .data files."
    )
    exit 2
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    Write-Error "Dataset root does not exist: $DataDir"
    exit 1
}

$dataRoot = (Resolve-Path -LiteralPath $DataDir).Path
$trainingDataDir = Join-Path $dataRoot "train"
$validationDataDir = Join-Path $dataRoot "validation"
foreach ($requiredDir in @($trainingDataDir, $validationDataDir)) {
    if (-not (Test-Path -LiteralPath $requiredDir -PathType Container)) {
        Write-Error "Missing dataset directory: $requiredDir"
        exit 1
    }
}

function Get-DatasetFiles([string]$Directory, [string]$Role) {
    $files = @(
        Get-ChildItem -LiteralPath $Directory -Recurse -File |
            Where-Object { $_.Extension -in @(".binpack", ".data") } |
            Sort-Object FullName
    )
    if ($files.Count -eq 0) {
        $compressed = @(
            Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter "*.zst"
        )
        $hint = if ($compressed.Count -gt 0) {
            " Decompress the .zst files first."
        } else {
            ""
        }
        Write-Error "No .binpack or .data files found for $Role under $Directory.$hint"
        exit 1
    }

    $extensions = @($files | ForEach-Object { $_.Extension.ToLowerInvariant() } | Select-Object -Unique)
    if ($extensions.Count -ne 1) {
        Write-Error "$Role data mixes .binpack and .data files under $Directory"
        exit 2
    }
    return $files
}

$trainingFiles = @(Get-DatasetFiles $trainingDataDir "training")
$validationFiles = @(Get-DatasetFiles $validationDataDir "validation")
$trainingExtension = $trainingFiles[0].Extension.ToLowerInvariant()
$validationExtension = $validationFiles[0].Extension.ToLowerInvariant()
if ($trainingExtension -ne $validationExtension) {
    Write-Error (
        "Training and validation formats differ: " +
        "$trainingExtension versus $validationExtension"
    )
    exit 2
}
if ($trainingExtension -eq ".data" -and $validationFiles.Count -ne 1) {
    Write-Error "BulletFormat validation requires exactly one .data file under $validationDataDir"
    exit 2
}

$dataFormat = if ($trainingExtension -eq ".binpack") { "binpack" } else { "bullet" }
$trainingDatasets = @($trainingFiles | ForEach-Object { $_.FullName })
$validationDatasets = @($validationFiles | ForEach-Object { $_.FullName })

# --- Registered network profile and training stage ---
$networkFormat = "syk8"
$hidden = if ($Profile -eq "v8-t768") { 768 } else { 1024 }
$dense1 = 16
$dense2 = 32
$outputBuckets = 8
$batchSize = 16384
$batchesPerSuperbatch = 6104
$saveRate = if ($Stage -eq "broad") { 25 } else { 10 }
$validationPositions = 262144

if ($Resume -and $WarmStart) {
    Write-Error "-Resume and -WarmStart are mutually exclusive"
    exit 2
}
if ($AllowRandomV8Init -and ($Resume -or $WarmStart)) {
    Write-Error "-AllowRandomV8Init cannot be combined with -Resume or -WarmStart"
    exit 2
}
if (-not $Resume -and -not $WarmStart -and -not $AllowRandomV8Init) {
    Write-Error "v8 requires -WarmStart/-Resume (or -AllowRandomV8Init for a smoke diagnostic)"
    exit 2
}
if ($Profile -ne "v8-t1024" -and $WarmStart) {
    Write-Error "-WarmStart is only valid for the v8-t1024 profile"
    exit 2
}
if ($Stage -eq "finetune" -and -not $Resume) {
    Write-Error "The finetune stage requires -Resume with a full-precision v8 checkpoint"
    exit 2
}
if ($Superbatches -lt 0) {
    Write-Error "-Superbatches cannot be negative"
    exit 2
}
if ($Wdl -lt 0.0 -or $Wdl -gt 1.0) {
    Write-Error "-Wdl must be in [0, 1]"
    exit 2
}
if ($LrStart -lt 0.0 -or $LrFinal -lt 0.0) {
    Write-Error "Learning rates cannot be negative"
    exit 2
}

if ($StartSuperbatch -le 0) {
    $StartSuperbatch = 1
    if ($Resume) {
        $resumeName = Split-Path ($Resume -replace '[\\/]+$', '') -Leaf
        if ($resumeName -match '-(\d+)$') {
            $StartSuperbatch = [int]$Matches[1] + 1
        } elseif ($Stage -eq "finetune") {
            Write-Error (
                "Cannot infer the resumed superbatch from checkpoint '$resumeName'. " +
                "Use -StartSuperbatch explicitly."
            )
            exit 2
        }
    }
}

if ($Smoke) {
    $endSuperbatch = $StartSuperbatch + 1
    $batchSize = 4096
    $batchesPerSuperbatch = 16
    $saveRate = 1
    $validationPositions = 16384
} elseif ($Superbatches -gt 0) {
    $endSuperbatch = $StartSuperbatch + $Superbatches - 1
} elseif ($Stage -eq "pilot") {
    $endSuperbatch = 200
} elseif ($Stage -eq "broad") {
    $endSuperbatch = 800
} else {
    $endSuperbatch = $StartSuperbatch + 199
}

if ($StartSuperbatch -gt $endSuperbatch) {
    Write-Error "Start superbatch $StartSuperbatch exceeds end $endSuperbatch"
    exit 2
}

$effectiveLrStart = if ($LrStart -gt 0.0) {
    $LrStart
} elseif ($Stage -eq "finetune") {
    0.0001
} else {
    0.001
}
$effectiveLrFinal = if ($LrFinal -gt 0.0) {
    $LrFinal
} elseif ($Stage -eq "finetune") {
    0.00001
} else {
    $effectiveLrStart * [Math]::Pow(0.3, 5)
}
if ($effectiveLrFinal -gt $effectiveLrStart) {
    Write-Error "Final learning rate cannot exceed the starting learning rate"
    exit 2
}
$lrOriginSuperbatch = if ($Stage -eq "finetune") { $StartSuperbatch } else { 1 }
$lrFinalSuperbatch = if ($Stage -eq "finetune") {
    $endSuperbatch
} else {
    [Math]::Max(800, $endSuperbatch)
}

$validationCache = ""
if ($dataFormat -eq "binpack") {
    $validationManifest = (
        $validationFiles |
            ForEach-Object {
                "$($_.FullName)|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)"
            }
    ) -join "`n"
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $manifestBytes = [System.Text.Encoding]::UTF8.GetBytes($validationManifest)
        $validationHash = (
            [System.BitConverter]::ToString($sha256.ComputeHash($manifestBytes)) -replace "-", ""
        ).Substring(0, 12).ToLowerInvariant()
    } finally {
        $sha256.Dispose()
    }
    $cacheDir = Join-Path $PSScriptRoot "nnue\data\validation"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    $validationCache = Join-Path $cacheDir "validation_${validationHash}_$validationPositions.data"
}

$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$wdlArgument = $Wdl.ToString("R", $invariantCulture)
$lrStartArgument = $effectiveLrStart.ToString("R", $invariantCulture)
$lrFinalArgument = $effectiveLrFinal.ToString("R", $invariantCulture)

$timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$profileTag = $Profile -replace '-', '_'
$runPrefix = if ($Smoke) { "smoke_$profileTag" } else { "${profileTag}_$Stage" }
$runId = "${runPrefix}_${timestamp}"

Write-Host "============================================"
Write-Host "  Sykora $($networkFormat.ToUpper()) training"
Write-Host "============================================"
Write-Host "Run ID:        $runId"
Write-Host "Profile:       $Profile ($Stage)"
Write-Host "Architecture:  factorised pairwise-MLP + full_threats_v1"
Write-Host "Shape:         H=$hidden, $hidden -> $dense1 -> $($dense1 * 2) -> $dense2 -> 1"
Write-Host "Output heads:  $outputBuckets material buckets"
Write-Host "Superbatches:  $StartSuperbatch -> $endSuperbatch"
Write-Host "Batch shape:   $batchSize x $batchesPerSuperbatch"
Write-Host "Data root:     $dataRoot"
Write-Host "Data format:   $dataFormat"
Write-Host "Train shards:  $($trainingDatasets.Count)"
Write-Host "Held out:      $($validationDatasets.Count) shard(s), $validationPositions positions"
Write-Host "WDL:           $wdlArgument"
Write-Host "Learning rate: $lrStartArgument -> $lrFinalArgument"
Write-Host "============================================"

$arguments = @(
    "$PSScriptRoot\utils\nnue\bullet\train_cuda_longrun.py",
    "--dataset"
) + $trainingDatasets + @(
    "--validation-positions", $validationPositions,
    "--bullet-repo", "$PSScriptRoot\nnue\bullet_repo",
    "--output-root", "$PSScriptRoot\nnue\models\bullet",
    "--run-id", $runId,
    "--data-format", $dataFormat,
    "--binpack-buffer-mb", 12288,
    "--binpack-threads", 6,
    "--validation-buffer-mb", 512,
    "--network-format", $networkFormat,
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
    "--wdl", $wdlArgument,
    "--lr-start", $lrStartArgument,
    "--lr-final", $lrFinalArgument,
    "--lr-origin-superbatch", $lrOriginSuperbatch,
    "--lr-final-superbatch", $lrFinalSuperbatch,
    "--export-after"
)
if ($dataFormat -eq "binpack") {
    $arguments += @(
        "--validation-dataset"
    ) + $validationDatasets + @(
        "--validation-cache", $validationCache
    )
} else {
    $arguments += @("--validation-sample", $validationDatasets[0])
}
$arguments += @("--validate-all-checkpoints", "--export-best-validation")
if ($Resume) {
    $arguments += @("--resume", $Resume)
}
if ($WarmStart) {
    $arguments += @("--warm-start", $WarmStart)
}
if ($AllowRandomV8Init) {
    $arguments += "--allow-random-v8-init"
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
