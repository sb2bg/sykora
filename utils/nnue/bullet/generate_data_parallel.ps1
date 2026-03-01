#
# Parallel self-play data generation for Sykora NNUE training.
# Launches N sykora gensfen instances with different seeds, then concatenates output.
#
# Usage:
#   .\utils\nnue\bullet\generate_data_parallel.ps1 `
#       -Shards 12 -Games 100000 -Depth 8 -Output data\train.data -Seed 1
#
# For i9-13900 (24C/32T): 12 shards recommended.
#

param(
    [int]$Shards = 12,
    [int]$Games = 100000,
    [int]$Depth = 8,
    [string]$Output = "data\train.data",
    [int]$Seed = 1,
    [int]$RandomPlies = 10,
    [int]$SamplePct = 25,
    [int]$MinPly = 16,
    [int]$MaxPly = 400,
    [string]$Bin = ".\zig-out\bin\sykora.exe"
)

$ErrorActionPreference = "Stop"

$PerShard = [math]::Ceiling($Games / $Shards)
$TmpDir = Join-Path $env:TEMP "sykora_gensfen_$(Get-Random)"
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

Write-Host "Generating data: $Games games across $Shards shards ($PerShard games/shard)"
Write-Host "Depth: $Depth, Random plies: $RandomPlies, Sample: $SamplePct%"
Write-Host "Output: $Output"
Write-Host "Temp dir: $TmpDir"
Write-Host ""

# Launch all shards as background processes
$processes = @()
for ($i = 0; $i -lt $Shards; $i++) {
    $shardSeed = $Seed + $i
    $shardFile = Join-Path $TmpDir "shard_$i.data"

    Write-Host "  Starting shard $i (seed=$shardSeed, games=$PerShard) ..."

    $proc = Start-Process -FilePath $Bin -ArgumentList @(
        "gensfen",
        "--output", $shardFile,
        "--games", $PerShard,
        "--depth", $Depth,
        "--random-plies", $RandomPlies,
        "--seed", $shardSeed,
        "--sample-pct", $SamplePct,
        "--min-ply", $MinPly,
        "--max-ply", $MaxPly,
        "--report-interval", 500
    ) -NoNewWindow -PassThru -RedirectStandardError (Join-Path $TmpDir "shard_${i}_stderr.txt")

    $processes += $proc
}

Write-Host ""
Write-Host "Waiting for $Shards shards to complete ..."

# Wait for all processes
$failed = 0
foreach ($proc in $processes) {
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
        Write-Host "  Shard PID $($proc.Id) failed with exit code $($proc.ExitCode)!" -ForegroundColor Red
        $failed++
    }
}

if ($failed -gt 0) {
    Write-Host "ERROR: $failed shard(s) failed." -ForegroundColor Red
    # Print stderr from failed shards
    foreach ($proc in $processes) {
        if ($proc.ExitCode -ne 0) {
            $idx = $processes.IndexOf($proc)
            $stderrFile = Join-Path $TmpDir "shard_${idx}_stderr.txt"
            if (Test-Path $stderrFile) {
                Write-Host "  stderr from shard $idx`:" -ForegroundColor Yellow
                Get-Content $stderrFile | Write-Host
            }
        }
    }
    Remove-Item -Recurse -Force $TmpDir
    exit 1
}

Write-Host "All shards complete. Concatenating ..."

# Ensure output directory exists
$outputDir = Split-Path -Parent $Output
if ($outputDir -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Binary concatenation of all shard files
$outputStream = [System.IO.File]::Create($Output)
try {
    $shardFiles = Get-ChildItem -Path $TmpDir -Filter "shard_*.data" | Sort-Object Name
    foreach ($shard in $shardFiles) {
        $bytes = [System.IO.File]::ReadAllBytes($shard.FullName)
        $outputStream.Write($bytes, 0, $bytes.Length)
    }
} finally {
    $outputStream.Close()
}

# Summary
$totalBytes = (Get-Item $Output).Length
$totalRecords = [math]::Floor($totalBytes / 32)

Write-Host ""
Write-Host "Done: $Output ($totalBytes bytes, $totalRecords positions)"

# Cleanup
Remove-Item -Recurse -Force $TmpDir
