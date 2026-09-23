#!/usr/bin/env bash
set -euo pipefail

binary=${1:?Usage: smoke_aarch64.sh PATH_TO_BINARY}
coproc ENGINE { timeout 120s qemu-aarch64 "$binary"; }

printf 'uci\n' >&"${ENGINE[1]}"
uci_seen=false
while IFS= read -r -t 120 line <&"${ENGINE[0]}"; do
    if [[ "$line" == uciok ]]; then
        uci_seen=true
        break
    fi
done
if [[ "$uci_seen" != true ]]; then
    echo 'AArch64 engine did not complete the UCI handshake' >&2
    exit 1
fi

printf 'isready\n' >&"${ENGINE[1]}"
IFS= read -r -t 120 line <&"${ENGINE[0]}"
if [[ "$line" != readyok ]]; then
    echo "Unexpected readiness response: $line" >&2
    exit 1
fi

printf 'quit\n' >&"${ENGINE[1]}"
wait "$ENGINE_PID"
echo 'AArch64 UCI handshake passed'
