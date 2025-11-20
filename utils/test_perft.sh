#!/bin/bash

echo "Testing Perft - Move Generation Correctness"
echo ""
echo "Expected values for starting position:"
echo "  Depth 1: 20 nodes"
echo "  Depth 2: 400 nodes"
echo "  Depth 3: 8,902 nodes"
echo "  Depth 4: 197,281 nodes"
echo "  Depth 5: 4,865,609 nodes (slow!)"
echo ""
echo "Running perft 4..."
echo ""

{
    echo "uci"
    sleep 0.1
    echo "isready"
    sleep 0.1
    echo "position startpos"
    sleep 0.1
    echo "perft 4"
    sleep 5
    echo "quit"
} | ./zig-out/bin/sykora
