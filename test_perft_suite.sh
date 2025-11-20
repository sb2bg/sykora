#!/bin/bash

echo "=== Perft Test Suite ==="
echo ""
echo "Testing various positions with known perft values"
echo ""

# Position 1: Starting position
echo "Position 1: Starting position"
echo "Expected: D1=20, D2=400, D3=8902, D4=197281"
{
    echo "uci"
    echo "isready"
    echo "position startpos"
    echo "perft 4 stats"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep -E "^(Depth|[ ]*[0-9])"
echo ""

# Position 2: Kiwipete (a complex middle-game position)
echo "Position 2: Kiwipete"
echo "Expected: D1=48, D2=2039, D3=97862, D4=4085603"
{
    echo "uci"
    echo "isready"
    echo "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    echo "perft 4 stats"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep -E "^(Depth|[ ]*[0-9])"
echo ""

# Position 3: Position with en passant
echo "Position 3: En Passant position"
echo "Expected: D1=14, D2=191, D3=2812, D4=43238"
{
    echo "uci"
    echo "isready"
    echo "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    echo "perft 4 stats"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep -E "^(Depth|[ ]*[0-9])"
echo ""

# Position 4: Castling rights position
echo "Position 4: Position testing castling"
echo "Expected: D1=6, D2=264, D3=9467, D4=422333"
{
    echo "uci"
    echo "isready"
    echo "position fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    echo "perft 4 stats"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep -E "^(Depth|[ ]*[0-9])"
echo ""

# Position 5: Buggy Finder
echo "Position 5: Finding tricky moves"
echo "Expected: D1=44, D2=1486, D3=62379, D4=2103487"
{
    echo "uci"
    echo "isready"
    echo "position fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
    echo "perft 4 stats"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep -E "^(Depth|[ ]*[0-9])"
echo ""

echo "=== Test Complete ==="
