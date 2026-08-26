param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"

$cutechess = "C:\Program Files (x86)\Cute Chess\cutechess-cli.exe"
$engine = Join-Path $RunDir "sykora.exe"
$p3Net = Join-Path $RunDir "p3_fastdot.sknnue"
$releaseNet = Join-Path $RunDir "release_4_0.sknnue"
$openings = Join-Path $RunDir "openings.epd"
$log = Join-Path $RunDir "match.log"
$pgn = Join-Path $RunDir "games.pgn"

foreach ($required in @($cutechess, $engine, $p3Net, $releaseNet, $openings)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing SPRT input: $required"
    }
}

# The i9-13900 exposes the eight hyper-threaded P-cores as logical CPUs 0-15.
# Cute Chess and every child engine inherit this affinity mask.
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]([Int64]65535)

& $cutechess `
    -engine "name=P3-fastdot" "cmd=$engine" proto=uci "option.EvalFile=$p3Net" `
    -engine "name=Release-4.0" "cmd=$engine" proto=uci "option.EvalFile=$releaseNet" `
    -each tc=120+2 option.Threads=1 option.Hash=64 restart=on `
    -tournament round-robin -games 2 -rounds 5000 -repeat `
    -openings "file=$openings" format=epd order=random `
    -srand 20260826 -concurrency 16 `
    -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 `
    -draw movenumber=40 movecount=8 score=10 `
    -resign movecount=3 score=600 twosided=true `
    -maxmoves 220 -ratinginterval 20 -outcomeinterval 20 `
    -pgnout $pgn fi -recover `
    2>&1 | Tee-Object -FilePath $log

exit $LASTEXITCODE
