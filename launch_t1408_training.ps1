$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$trainingRoot = "D:\training\sykora"
$runId = "t1408_sf96m_20260826T181000"
$sprtDir = "D:\training\sykora\sprt_p3_vs_release4_current_20260826T163042"
$python = Join-Path $repo "nnue\.venv\Scripts\python.exe"
$trainer = Join-Path $repo "utils\nnue\bullet\train_cuda_longrun.py"
$warmStart = Join-Path $repo "nnue\models\bullet\v8_t1024_broad_20260724T205147Z\checkpoints\v8_t1024_broad_20260724T205147Z-800"
$validation = "D:\datasets\chess\stockfish_test80_binpacks\validation\t80_2024_06_v3filter_262144.data"
$parityEngine = "D:\training\sykora\t1408_support\sykora_t1408.exe"

$datasets = @(
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-06-jun-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-07-jul-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-09-sep-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-10-oct-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-11-nov-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2023-12-dec-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
    "D:\datasets\chess\stockfish_test80_binpacks\test80-2024-05-may-2tb7p.min-v2.v6.binpack"
)

foreach ($required in @($python, $trainer, $warmStart, $validation, $parityEngine) + $datasets) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing T1408 training input: $required"
    }
}

# Keep the trainer and its child processes on the sixteen E-cores. The active
# Cute Chess launcher is restricted to logical CPUs 0-15 (the P-cores).
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]([Int64]4294901760)

while ($true) {
    $activeSprt = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "cutechess-cli.exe" -and $_.CommandLine -like "*$sprtDir*"
    }
    if (-not $activeSprt) {
        break
    }
    Start-Sleep -Seconds 30
}

Set-Location -LiteralPath $repo
& $python $trainer `
    --dataset $datasets `
    --validation-sample $validation `
    --validation-positions 262144 `
    --output-root $trainingRoot `
    --run-id $runId `
    --architecture pairwise-mlp `
    --network-format syk8 `
    --hidden 1408 `
    --start-superbatch 1 `
    --end-superbatch 800 `
    --batch-size 16384 `
    --batches-per-superbatch 6104 `
    --lr-start 0.0001 `
    --lr-origin-superbatch 1 `
    --lr-final-superbatch 800 `
    --lr-final 0.00001 `
    --wdl 0.25 `
    --save-rate 25 `
    --threads 8 `
    --backend cuda `
    --data-format binpack `
    --binpack-buffer-mb 12288 `
    --binpack-threads 6 `
    --warm-start $warmStart `
    --validate-after `
    --export-after `
    --parity-engine $parityEngine

exit $LASTEXITCODE
