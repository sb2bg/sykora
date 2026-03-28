# Sykora NNUE V4 Training Launch Script
# Run from project root: .\launch_training.ps1
#
# Dataset: T80-2023 (jun-dec) + T80-2024 (jan-jun) .min-v2.v6 binpacks
# Source:  linrock/test80-2023 + linrock/test80-2024 on HuggingFace
#
# Decompress first:
#   cd nnue\data\binpack
#   Get-ChildItem *.zst | ForEach-Object { zstd -d $_.FullName }

$ErrorActionPreference = "Stop"

# --- Environment Setup ---
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

# Activate venv
& "$PSScriptRoot\nnue\.venv\Scripts\Activate.ps1"

# --- Dataset Setup ---
$dataDir = "$PSScriptRoot\nnue\data\binpack"

# T80-2023 jun-dec + T80-2024 jan-jun, all .min-v2.v6 filtered.
$binpacks = @(
    # 2023
    "test80-2023-06-jun-2tb7p.min-v2.v6.binpack",
    "test80-2023-07-jul-2tb7p.min-v2.v6.binpack",
    "test80-2023-09-sep-2tb7p.min-v2.v6.binpack",
    "test80-2023-10-oct-2tb7p.min-v2.v6.binpack",
    "test80-2023-11-nov-2tb7p.min-v2.v6.binpack",
    "test80-2023-12-dec-2tb7p.min-v2.v6.binpack",
    # 2024
    "test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
    "test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
    "test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
    "test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
    "test80-2024-05-may-2tb7p.min-v2.v6.binpack",
    "test80-2024-06-jun-2tb7p.min-v2.v6.binpack"
)

$datasets = @()
foreach ($bp in $binpacks) {
    $file = Join-Path $dataDir $bp
    if (-not (Test-Path $file)) {
        Write-Error "Missing: $file`nDecompress with: zstd -d `"$file.zst`""
        exit 1
    }
    $datasets += (Resolve-Path $file).Path
}

# --- Training Parameters ---
# SYKNNUE4 baseline:
# mirrored king buckets (sykora16) -> FT 2048 + PSQT side path -> shared 32 -> 32 -> 1
$networkFormat = "syk4"
$bucketLayout = "sykora16"
$hidden = 2048
$denseL1 = 32
$denseL2 = 32
$endSuperbatch = 600
$lrStart = 0.001
$wdl = 0.25
$saveRate = 10
$threads = 8

Write-Host "============================================"
Write-Host "  Sykora NNUE V4 Training (RTX 4070 Ti SUPER)"
Write-Host "============================================"
Write-Host "Data:          T80-2023/2024 filtered set"
Write-Host "Filtering:     .min-v2.v6 on T80 inputs"
Write-Host "Binpacks:      $($binpacks.Count) files"
Write-Host "Format:        binpack (sfbinpack)"
Write-Host "Net format:    $networkFormat"
Write-Host "Bucket layout: $bucketLayout"
Write-Host "FT hidden:     $hidden"
Write-Host "Dense head:    shared $($hidden * 2) -> $denseL1 -> $denseL2 -> 1"
Write-Host "PSQT path:     enabled"
Write-Host "Superbatches:  1 -> $endSuperbatch"
Write-Host "Save rate:     every $saveRate superbatches"
Write-Host "Threads:       $threads"
Write-Host "WDL blend:     $wdl"
Write-Host "LR:            $lrStart -> cosine decay"
Write-Host "============================================"
Write-Host ""
foreach ($bp in $binpacks) {
    Write-Host "  $bp"
}
Write-Host ""

python "$PSScriptRoot\utils\nnue\bullet\train_cuda_longrun.py" `
    --dataset @datasets `
    --bullet-repo "$PSScriptRoot\nnue\bullet_repo" `
    --output-root "$PSScriptRoot\nnue\models\bullet" `
    --data-format binpack `
    --binpack-buffer-mb 12288 `
    --binpack-threads 6 `
    --network-format $networkFormat `
    --bucket-layout $bucketLayout `
    --hidden $hidden `
    --dense-l1 $denseL1 `
    --dense-l2 $denseL2 `
    --end-superbatch $endSuperbatch `
    --save-rate $saveRate `
    --threads $threads `
    --wdl $wdl `
    --lr-start $lrStart
