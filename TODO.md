# Sykora Engine - Search & Eval Improvements

## Search Improvements

### LMR Log Formula + History-Based Reductions (~20-40 Elo)

Replace the current step-function LMR with a logarithmic formula:
`R = base + ln(depth) * ln(moveIndex) / C`. This gives smoother, more aggressive
reductions for late moves at high depths. Additionally:

- Reduce less for killer/counter moves
- Reduce more for moves with bad history scores
- Use the history score to modulate the reduction amount

### Late Move Pruning (LMP) (~10-20 Elo)

At low depths, skip quiet moves entirely after searching N moves. For example:

- Depth 1: prune after 5 moves
- Depth 2: prune after 10 moves
- Depth 3: prune after 15 moves
  Only apply at non-PV, non-check nodes. Do not prune captures or promotions.

### Proper Static Exchange Evaluation (SEE) (~20-40 Elo)

Replace the simple heuristic (`attacker > victim + 50`) with a full SEE algorithm
that resolves the entire capture chain on a square. Benefits:

- More accurate capture ordering (better good/bad capture classification)
- Better quiescence search pruning (skip truly losing captures)
- Can be used in LMR/futility to prune losing captures more aggressively
- Enables SEE-based pruning in the main search

### Internal Iterative Deepening / Reduction (IID/IIR) (~10-15 Elo)

When there is no TT move at a PV node (or any node at sufficient depth), either:

- **IID**: Do a reduced-depth search first to find a likely best move for ordering
- **IIR** (simpler): Just reduce the search depth by 1-2 plies
  Modern engines prefer IIR for simplicity.

### Singular Extensions (~20-30 Elo)

At sufficient depth, if the TT move's score is significantly better than all
alternatives, extend it by 1 ply. Test with a reduced-depth search excluding the
TT move — if all other moves score well below the TT score minus a margin, the TT
move is "singular" and deserves deeper analysis. One of the bigger Elo gains in
modern engines.

## Evaluation Improvements

### Texel Tuning (~100-200 Elo)

Use a labeled dataset of positions (from self-play or existing databases) to
optimize all eval parameters via gradient descent. Parameters to tune:

- Material values (pawn, knight, bishop, rook, queen)
- All piece-square table values (middlegame and endgame)
- Pawn structure weights (doubled, isolated, backward, passed, chains)
- Mobility table values (knight, bishop, rook)
- King safety weights (pawn shield, open files, center penalty)
- Rook bonuses (open file, semi-open, 7th rank, connected)
- Outpost bonuses
- Bishop pair bonus
- Endgame mop-up weights

### Better King Safety with Attack Tables (~30-50 Elo)

Replace basic pawn shield + open file evaluation with attack-count-based king
safety:

- Track how many pieces attack squares near the enemy king
- Weight attacks by piece type (queen attacks worth more than knight attacks)
- Sum up an "attack index" and look up a non-linear penalty table
- Add pawn storm detection (penalize when enemy pawns advance toward king)
- Add virtual king mobility (how many safe squares the king has)

### Passed Pawn Improvements (~15-25 Elo)

- **Tarrasch rule**: Bonus for rooks behind passed pawns (both own and enemy)
- **King proximity**: In endgame, bonus for friendly king near passed pawn, penalty
  for enemy king near it
- **Unstoppable passer**: Detect when a passed pawn can't be caught by enemy king
  (rule of the square)
- **Blockade**: Penalty when a passed pawn is blocked by an enemy piece

## Speed Improvements

### Lazy SMP / Parallel Search (~50-70 Elo)

Run the same search on multiple threads with a shared transposition table. Different
threads naturally explore different subtrees due to TT race conditions. This is the
standard approach and scales well up to 4-8 threads. Requires:

- Shared TT with atomic access
- Thread pool management
- Aggregated node counts and time management

### NNUE Evaluation (~200-400 Elo)

Replace the hand-crafted evaluation with a neural network (NNUE). This is the gold
standard for modern engines. Requires:

- Network architecture design (HalfKP or similar input features)
- Training data generation from self-play
- Efficient inference with incremental updates on make/unmake
- SIMD-optimized matrix operations

### Minor Speed Optimizations

- **Pawn hash table**: Cache pawn structure evaluations in a separate small hash
  table. Pawn structure changes rarely, so this avoids redundant computation.
- **TT prefetch**: After making a move, issue a memory prefetch for the child's TT
  entry to hide cache miss latency.
