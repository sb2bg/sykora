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
    echo "perft 4"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep "^Depth"
echo ""

# Position 2: Kiwipete (a complex middle-game position)
echo "Position 2: Kiwipete"
echo "Expected: D1=48, D2=2039, D3=97862, D4=4085603"
{
    echo "uci"
    echo "isready"
    echo "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    echo "perft 3"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep "^Depth"
echo ""

# Position 3: Position with en passant
echo "Position 3: En Passant position"
echo "Expected: D1=9, D2=193, D3=1322"
{
    echo "uci"
    echo "isready"
    echo "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    echo "perft 3"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep "^Depth"
echo ""

# Position 4: Castling rights position
echo "Position 4: Position testing castling"
echo "Expected: D1=6, D2=264, D3=9467"
{
    echo "uci"
    echo "isready"
    echo "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    echo "perft 3"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep "^Depth"
echo ""

# Position 5: Promotions
echo "Position 5: Promotion position"
echo "Expected: D1=24, D2=496, D3=9483"
{
    echo "uci"
    echo "isready"
    echo "position fen n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"
    echo "perft 3"
    echo "quit"
} | ./zig-out/bin/sykora 2>&1 | grep "^Depth"
echo ""

echo "=== Test Complete ==="
