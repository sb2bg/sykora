#!/bin/bash

# Simple test script for move generation
echo "Testing Sykora move generation"
echo ""

# Test commands
{
    echo "uci"
    sleep 0.1
    echo "debug on"
    sleep 0.1
    echo "isready"
    sleep 0.1
    echo "position startpos"
    sleep 0.1
    echo "display"
    sleep 0.1
    echo "go movetime 100"
    sleep 0.5
    echo "position startpos moves e2e4"
    sleep 0.1
    echo "display"
    sleep 0.1
    echo "go movetime 100"
    sleep 0.5
    echo "quit"
} | ./zig-out/bin/sykora
