const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const pieceInfo = @import("piece.zig");
const fen = @import("fen.zig");
const zobrist = @import("zobrist.zig");
const ZobristHasher = zobrist.ZobristHasher;

// Pre-computed attack tables for knight and king
const KNIGHT_ATTACKS = initKnightAttacks();
const KING_ATTACKS = initKingAttacks();

// Magic bitboard constants for rook attacks
const ROOK_MAGICS = initRookMagics();
const ROOK_MASKS = initRookMasks();
const ROOK_SHIFTS = initRookShifts();
const ROOK_ATTACKS = initRookAttacks();

// Magic bitboard constants for bishop attacks
const BISHOP_MAGICS = initBishopMagics();
const BISHOP_MASKS = initBishopMasks();
const BISHOP_SHIFTS = initBishopShifts();
const BISHOP_ATTACKS = initBishopAttacks();

// Bitboard masks for ranks and files (used in move generation)
const RANK_1_MASK: u64 = 0x00000000000000FF;
const RANK_2_MASK: u64 = 0x000000000000FF00;
const RANK_7_MASK: u64 = 0x00FF000000000000;
const RANK_8_MASK: u64 = 0xFF00000000000000;
const NOT_A_FILE: u64 = 0xFEFEFEFEFEFEFEFE;
const NOT_H_FILE: u64 = 0x7F7F7F7F7F7F7F7F;

fn initKnightAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (file > 0 and square < 48) result |= sq << 15;
        if (file < 7 and square < 48) result |= sq << 17;
        if (file > 1 and square < 56) result |= sq << 6;
        if (file < 6 and square < 56) result |= sq << 10;
        if (file > 0 and square >= 16) result |= sq >> 17;
        if (file < 7 and square >= 16) result |= sq >> 15;
        if (file > 1 and square >= 8) result |= sq >> 10;
        if (file < 6 and square >= 8) result |= sq >> 6;

        attacks[square] = result;
    }
    return attacks;
}

fn initKingAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (square < 56) result |= sq << 8;
        if (square >= 8) result |= sq >> 8;
        if (file > 0) result |= sq >> 1;
        if (file < 7) result |= sq << 1;
        if (file > 0 and square < 56) result |= sq << 7;
        if (file < 7 and square < 56) result |= sq << 9;
        if (file > 0 and square >= 8) result |= sq >> 9;
        if (file < 7 and square >= 8) result |= sq >> 7;

        attacks[square] = result;
    }
    return attacks;
}

// Magic bitboard initialization functions

fn initRookMagics() [64]u64 {
    // These are pre-computed magic numbers for rook attacks
    // Found through trial and error or algorithms like Fancy Magic Bitboards
    return [64]u64{
        0xa8002c000108020,  0x6c00049b0002001,  0x100200010090040,  0x2480041000800801, 0x280028004000800,  0x900410008040022,  0x280020001001080,  0x2880002041000080,
        0xa000800080400034, 0x4808020004000,    0x2290802004801000, 0x411000d00100020,  0x402800800040080,  0xb000401004208,    0x2409000100040200, 0x1002100004082,
        0x22878001e24000,   0x1090810021004010, 0x801030040200012,  0x500808008001000,  0xa08018014000880,  0x8000808004000200, 0x201008080010200,  0x801020000441091,
        0x800080204005,     0x1040200040100048, 0x120200402082,     0xd14880480100080,  0x12040280080080,   0x100040080020080,  0x9020010080800400, 0x813241200148449,
        0x491604001800080,  0x100401000402001,  0x4820010021001040, 0x400402202000812,  0x209009005000802,  0x810800601800400,  0x4301083214000150, 0x204026458e001401,
        0x40204000808000,   0x8001008040010020, 0x8410820820420010, 0x1003001000090020, 0x804040008008080,  0x12000810020004,   0x1000100200040208, 0x430000a044020001,
        0x280009023410300,  0xe0100040002240,   0x200100401700,     0x2244100408008080, 0x8000400801980,    0x2000810040200,    0x8010100228810400, 0x2000009044210200,
        0x4080008040102101, 0x40002080411d01,   0x2005524060000901, 0x502001008400422,  0x489a000810200402, 0x1004400080a13,    0x4000011008020084, 0x26002114058042,
    };
}

fn initRookMasks() [64]u64 {
    @setEvalBranchQuota(20000);
    var masks: [64]u64 = undefined;
    for (0..64) |square| {
        const rank: u8 = @intCast(square / 8);
        const file: u8 = @intCast(square % 8);
        var mask: u64 = 0;

        // North
        var r: u8 = rank + 1;
        while (r < 7) : (r += 1) {
            mask |= @as(u64, 1) << @intCast(r * 8 + file);
        }

        // South
        r = rank;
        while (r > 0) {
            r -= 1;
            if (r == 0) break;
            mask |= @as(u64, 1) << @intCast(r * 8 + file);
        }

        // East
        var f: u8 = file + 1;
        while (f < 7) : (f += 1) {
            mask |= @as(u64, 1) << @intCast(rank * 8 + f);
        }

        // West
        f = file;
        while (f > 0) {
            f -= 1;
            if (f == 0) break;
            mask |= @as(u64, 1) << @intCast(rank * 8 + f);
        }

        masks[square] = mask;
    }
    return masks;
}

fn initRookShifts() [64]u8 {
    @setEvalBranchQuota(10000);
    var shifts: [64]u8 = undefined;
    for (0..64) |square| {
        const bits = @popCount(ROOK_MASKS[square]);
        shifts[square] = @intCast(64 - bits);
    }
    return shifts;
}

fn initBishopMagics() [64]u64 {
    // Pre-computed magic numbers for bishop attacks
    return [64]u64{
        0x40040844404084,   0x2004208a004208,   0x10190041080202,   0x108060845042010,  0x581104180800210,  0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
        0x4050404440404,    0x21001420088,      0x24d0080801082102, 0x1020a0a020400,    0x40308200402,      0x4011002100800,    0x401484104104005,  0x801010402020200,
        0x400210c3880100,   0x404022024108200,  0x810018200204102,  0x4002801a02003,    0x85040820080400,   0x810102c808880400, 0xe900410884800,    0x8002020480840102,
        0x220200865090201,  0x2010100a02021202, 0x152048408022401,  0x20080002081110,   0x4001001021004000, 0x800040400a011002, 0xe4004081011002,   0x1c004001012080,
        0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080, 0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
        0x209188240001000,  0x400408a884001800, 0x110400a6080400,   0x1840060a44020800, 0x90080104000041,   0x201011000808101,  0x1a2208080504f080, 0x8012020600211212,
        0x500861011240000,  0x180806108200800,  0x4000020e01040044, 0x300000261044000a, 0x802241102020002,  0x20906061210001,   0x5a84841004010310, 0x4010801011c04,
        0xa010109502200,    0x4a02012000,       0x500201010098b028, 0x8040002811040900, 0x28000010020204,   0x6000020202d0240,  0x8918844842082200, 0x4010011029020020,
    };
}

fn initBishopMasks() [64]u64 {
    @setEvalBranchQuota(20000);
    var masks: [64]u64 = undefined;
    for (0..64) |square| {
        const rank: i8 = @intCast(square / 8);
        const file: i8 = @intCast(square % 8);
        var mask: u64 = 0;

        // North-East
        var r: i8 = rank + 1;
        var f: i8 = file + 1;
        while (r < 7 and f < 7) : ({
            r += 1;
            f += 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // North-West
        r = rank + 1;
        f = file - 1;
        while (r < 7 and f > 0) : ({
            r += 1;
            f -= 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // South-East
        r = rank - 1;
        f = file + 1;
        while (r > 0 and f < 7) : ({
            r -= 1;
            f += 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // South-West
        r = rank - 1;
        f = file - 1;
        while (r > 0 and f > 0) : ({
            r -= 1;
            f -= 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        masks[square] = mask;
    }
    return masks;
}

fn initBishopShifts() [64]u8 {
    @setEvalBranchQuota(10000);
    var shifts: [64]u8 = undefined;
    for (0..64) |square| {
        const bits = @popCount(BISHOP_MASKS[square]);
        shifts[square] = @intCast(64 - bits);
    }
    return shifts;
}

// Helper function to compute rook attacks for a given square and occupancy
fn computeRookAttacks(square: u6, occupied: u64) u64 {
    var attacks: u64 = 0;
    const file: i8 = @intCast(square % 8);
    const rank: i8 = @intCast(square / 8);

    // North
    var r: i8 = rank + 1;
    while (r < 8) : (r += 1) {
        const sq: u6 = @intCast(r * 8 + file);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South
    r = rank - 1;
    while (r >= 0) : (r -= 1) {
        const sq: u6 = @intCast(r * 8 + file);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // East
    var f: i8 = file + 1;
    while (f < 8) : (f += 1) {
        const sq: u6 = @intCast(rank * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // West
    f = file - 1;
    while (f >= 0) : (f -= 1) {
        const sq: u6 = @intCast(rank * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    return attacks;
}

// Helper function to compute bishop attacks for a given square and occupancy
fn computeBishopAttacks(square: u6, occupied: u64) u64 {
    var attacks: u64 = 0;
    const file: i8 = @intCast(square % 8);
    const rank: i8 = @intCast(square / 8);

    // North-East
    var r: i8 = rank + 1;
    var f: i8 = file + 1;
    while (r < 8 and f < 8) : ({
        r += 1;
        f += 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // North-West
    r = rank + 1;
    f = file - 1;
    while (r < 8 and f >= 0) : ({
        r += 1;
        f -= 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South-East
    r = rank - 1;
    f = file + 1;
    while (r >= 0 and f < 8) : ({
        r -= 1;
        f += 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South-West
    r = rank - 1;
    f = file - 1;
    while (r >= 0 and f >= 0) : ({
        r -= 1;
        f -= 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    return attacks;
}

fn initRookAttacks() [64][4096]u64 {
    @setEvalBranchQuota(1000000);
    var attacks: [64][4096]u64 = undefined;

    for (0..64) |square| {
        const sq: u6 = @intCast(square);
        const mask = ROOK_MASKS[square];
        const bits = @popCount(mask);
        const permutations: usize = @as(usize, 1) << @intCast(bits);

        for (0..permutations) |i| {
            const occupied = indexToOccupancy(i, bits, mask);
            const magic_index = (occupied *% ROOK_MAGICS[square]) >> @intCast(ROOK_SHIFTS[square]);
            attacks[square][magic_index] = computeRookAttacks(sq, occupied);
        }
    }

    return attacks;
}

fn initBishopAttacks() [64][512]u64 {
    @setEvalBranchQuota(1000000);
    var attacks: [64][512]u64 = undefined;

    for (0..64) |square| {
        const sq: u6 = @intCast(square);
        const mask = BISHOP_MASKS[square];
        const bits = @popCount(mask);
        const permutations: usize = @as(usize, 1) << @intCast(bits);

        for (0..permutations) |i| {
            const occupied = indexToOccupancy(i, bits, mask);
            const magic_index = (occupied *% BISHOP_MAGICS[square]) >> @intCast(BISHOP_SHIFTS[square]);
            attacks[square][magic_index] = computeBishopAttacks(sq, occupied);
        }
    }

    return attacks;
}

// Convert an index to an occupancy bitboard for magic bitboard initialization
fn indexToOccupancy(index: usize, bits: u8, mask: u64) u64 {
    var occupancy: u64 = 0;
    var temp_mask = mask;

    for (0..bits) |i| {
        const bit_index = @ctz(temp_mask);
        temp_mask &= temp_mask - 1; // Clear the least significant bit

        if ((index & (@as(usize, 1) << @intCast(i))) != 0) {
            occupancy |= @as(u64, 1) << @intCast(bit_index);
        }
    }

    return occupancy;
}

pub const MAX_MOVES = 256;

pub const MoveList = struct {
    moves: [MAX_MOVES]Move = undefined,
    count: usize = 0,

    pub fn init() MoveList {
        return MoveList{};
    }

    pub fn append(self: *MoveList, move: Move) void {
        self.moves[self.count] = move;
        self.count += 1;
    }

    pub fn slice(self: *MoveList) []const Move {
        return self.moves[0..self.count];
    }

    pub fn sliceMut(self: *MoveList) []Move {
        return self.moves[0..self.count];
    }
};

pub const Move = struct {
    const Self = @This();
    from: u8,
    to: u8,
    promotion: ?pieceInfo.Type = null,

    pub fn init(from: u8, to: u8, promotion: ?pieceInfo.Type) Self {
        return Self{ .from = from, .to = to, .promotion = promotion };
    }

    pub fn fromString(str: []const u8) UciError!Self {
        if (str.len < 4 or str.len > 5) return error.InvalidArgument;

        const promotion_str = if (str.len == 5) std.ascii.toLower(str[4]) else null;
        const from_file = std.ascii.toLower(str[0]);
        const from_rank = str[1];
        const to_file = std.ascii.toLower(str[2]);
        const to_rank = str[3];

        if (from_file < 'a' or from_file > 'h' or from_rank < '1' or from_rank > '8' or
            to_file < 'a' or to_file > 'h' or to_rank < '1' or to_rank > '8')
        {
            return error.InvalidArgument;
        }

        const promotion = if (promotion_str) |p| pieceInfo.Type.fromChar(p) else null;
        const from_index = Board.rankFileToIndex(from_rank, from_file);
        const to_index = Board.rankFileToIndex(to_rank, to_file);

        return Self.init(from_index, to_index, promotion);
    }

    pub fn format(
        self: Self,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        if (self.from == 0 and self.to == 0 and self.promotion == null) {
            // null move
            try writer.writeAll("0000");
            return;
        }

        const from_file = 'a' + (self.from % 8);
        const from_rank = '1' + (self.from / 8);
        const to_file = 'a' + (self.to % 8);
        const to_rank = '1' + (self.to / 8);

        try writer.print("{c}{c}{c}{c}", .{ from_file, from_rank, to_file, to_rank });

        if (self.promotion) |p| {
            try writer.print("{c}", .{std.ascii.toLower(p.getName())});
        }
    }
};

pub const Board = struct {
    const Self = @This();
    board: BitBoard,
    zobrist_hasher: ZobristHasher = ZobristHasher.init(),

    pub fn init() Self {
        const board = BitBoard.init();
        var self = Self{ .board = board };
        self.zobrist_hasher.hash(self.board);
        return self;
    }

    pub fn startpos() Self {
        return fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") catch unreachable;
    }

    pub fn fromFen(fen_str: []const u8) UciError!Self {
        const board = try fen.FenParser.parse(fen_str);
        var self = Self{ .board = board };
        self.zobrist_hasher.hash(self.board);
        return self;
    }

    /// Make a move on the board using a Move structure.
    /// This updates the board state and zobrist hash.
    /// Note: This does NOT validate that the move is legal. Use with caution.
    pub fn makeMove(self: *Self, move: Move) UciError!void {
        const color = self.getTurn();
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
        const piece_type = self.board.getPieceAt(move.from, color) orelse return error.InvalidMove;
        const captured = self.board.getPieceAt(move.to, opponent_color);

        const prev_castle = self.board.castle_rights;
        const prev_ep = self.board.en_passant_square;

        // Apply the move unchecked
        self.applyMoveUnchecked(move);

        // Update zobrist hash for the move
        self.zobrist_hasher.updateHash(
            move.from,
            move.to,
            piece_type,
            color,
            captured,
            prev_castle,
            self.board.castle_rights,
            prev_ep,
            self.board.en_passant_square,
        );
    }

    /// Make a move from string notation (e.g., "e2e4").
    /// This validates that the move is legal before applying it.
    pub fn makeStrMove(self: *Self, move_str: []const u8) UciError!void {
        const move = try Move.fromString(move_str);

        // Validate that the move is legal
        var legal_moves = MoveList.init();
        try self.generateLegalMoves(&legal_moves);

        for (legal_moves.slice()) |legal_move| {
            if (legal_move.from == move.from and
                legal_move.to == move.to and
                legal_move.promotion == move.promotion)
            {
                return self.makeMove(move);
            }
        }

        return error.IllegalMove;
    }

    /// Generate all legal moves for the current position
    pub fn generateLegalMoves(self: *Self, moves: *MoveList) !void {
        var pseudo_legal = MoveList.init();
        try self.generatePseudoLegalMoves(&pseudo_legal);

        const color = self.board.move;

        // Filter out moves that leave the king in check
        // Optimize by saving/restoring state only once per move
        for (pseudo_legal.slice()) |move| {
            // Save state
            const old_board = self.board;
            const old_hash = self.zobrist_hasher.zobrist_hash;

            // Make move
            self.applyMoveUnchecked(move);

            // Check if our king is in check after the move (illegal)
            const legal = !self.isInCheck(color);

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;

            if (legal) {
                moves.append(move);
            }
        }
    }

    pub const PerftStats = struct {
        nodes: u64 = 0,
        captures: u64 = 0,
        en_passant: u64 = 0,
        castles: u64 = 0,
        promotions: u64 = 0,
        checks: u64 = 0,
        discovery_checks: u64 = 0,
        double_checks: u64 = 0,
        checkmates: u64 = 0,

        pub fn add(self: *PerftStats, other: PerftStats) void {
            self.nodes += other.nodes;
            self.captures += other.captures;
            self.en_passant += other.en_passant;
            self.castles += other.castles;
            self.promotions += other.promotions;
            self.checks += other.checks;
            self.discovery_checks += other.discovery_checks;
            self.double_checks += other.double_checks;
            self.checkmates += other.checkmates;
        }
    };

    /// Perft - Performance test for move generation (fast version without stats)
    /// Returns the number of leaf nodes at the given depth
    pub fn perft(self: *Self, depth: u32) UciError!u64 {
        if (depth == 0) {
            return 1;
        }

        // Generate pseudo-legal moves once
        var pseudo_legal = MoveList.init();
        try self.generatePseudoLegalMoves(&pseudo_legal);

        var nodes: u64 = 0;
        const color = self.board.move;

        // Test each pseudo-legal move for legality
        for (pseudo_legal.slice()) |move| {
            // Save state
            const old_board = self.board;
            const old_hash = self.zobrist_hasher.zobrist_hash;

            // Make move
            self.applyMoveUnchecked(move);

            // Check if our king is in check after the move (illegal)
            if (self.isInCheck(color)) {
                // Illegal move, restore and skip
                self.board = old_board;
                self.zobrist_hasher.zobrist_hash = old_hash;
                continue;
            }

            // Legal move
            if (depth == 1) {
                // Bulk counting at depth 1
                nodes += 1;
            } else {
                // Recurse
                nodes += try self.perft(depth - 1);
            }

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;
        }

        return nodes;
    }

    /// Perft with detailed statistics
    pub fn perftWithStats(self: *Self, depth: u32, stats: *PerftStats) UciError!void {
        if (depth == 0) {
            stats.nodes = 1;
            return;
        }

        // Generate pseudo-legal moves once
        var pseudo_legal = MoveList.init();
        try self.generatePseudoLegalMoves(&pseudo_legal);

        const moving_color = self.board.move;
        const opponent_color = if (moving_color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        if (depth == 1) {
            // At depth 1, count move types for legal moves only
            for (pseudo_legal.slice()) |move| {
                // Save state
                const old_board = self.board;
                const old_hash = self.zobrist_hasher.zobrist_hash;

                // Make move
                self.applyMoveUnchecked(move);

                // Check if our king is in check after the move (illegal)
                if (self.isInCheck(moving_color)) {
                    // Illegal move, restore and skip
                    self.board = old_board;
                    self.zobrist_hasher.zobrist_hash = old_hash;
                    continue;
                }

                // Legal move - count it and check properties
                stats.nodes += 1;

                // Check piece type before checking move properties
                const piece_type = old_board.getPieceAt(move.from, moving_color);

                // Check if it's en passant
                const is_en_passant = piece_type == .pawn and old_board.en_passant_square == move.to;
                if (is_en_passant) {
                    stats.en_passant += 1;
                }

                // Check if it's a capture (including en passant)
                const piece_at_dest = old_board.getPieceAt(move.to, opponent_color);
                if (piece_at_dest != null or is_en_passant) {
                    stats.captures += 1;
                }

                // Check if it's a castle
                if (piece_type == .king) {
                    const from_file = move.from % 8;
                    const to_file = move.to % 8;
                    if (from_file == 4 and (to_file == 6 or to_file == 2)) {
                        stats.castles += 1;
                    }
                }

                // Check if it's a promotion
                if (move.promotion != null) {
                    stats.promotions += 1;
                }

                // Check if opponent is in check after this move
                const opponent_in_check = self.isInCheck(opponent_color);
                if (opponent_in_check) {
                    // Count the number of pieces giving check
                    const checking_pieces = self.countCheckingPieces(opponent_color);

                    if (checking_pieces >= 2) {
                        stats.double_checks += 1;
                    } else if (checking_pieces == 1) {
                        // Check if it's a discovery check
                        const direct_check = self.isDirectCheck(move, moving_color, opponent_color);
                        if (!direct_check) {
                            stats.discovery_checks += 1;
                        }
                    }

                    stats.checks += 1;

                    // Check if it's checkmate - use fast perft to count legal moves
                    const opponent_legal = try self.perft(1);
                    if (opponent_legal == 0) {
                        stats.checkmates += 1;
                    }
                }

                // Restore state
                self.board = old_board;
                self.zobrist_hasher.zobrist_hash = old_hash;
            }
            return;
        }

        // Recursive case
        for (pseudo_legal.slice()) |move| {
            // Save state
            const old_board = self.board;
            const old_hash = self.zobrist_hasher.zobrist_hash;

            // Make move
            self.applyMoveUnchecked(move);

            // Check if our king is in check after the move (illegal)
            if (self.isInCheck(moving_color)) {
                // Illegal move, restore and skip
                self.board = old_board;
                self.zobrist_hasher.zobrist_hash = old_hash;
                continue;
            }

            // Recurse
            var child_stats = PerftStats{};
            try self.perftWithStats(depth - 1, &child_stats);
            stats.add(child_stats);

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;
        }
    }

    /// Count how many pieces are giving check to the king of the specified color.
    ///
    /// This function examines all possible attack vectors (pawns, knights, bishops,
    /// rooks, and queens) to determine how many enemy pieces are currently attacking
    /// the king. This is useful for distinguishing between single checks, double checks,
    /// and discovered checks.
    ///
    /// Parameters:
    ///   - king_color: The color of the king to check attacks against
    ///
    /// Returns: The number of pieces attacking the king (0-2 typically, though
    ///          theoretically more is possible)
    fn countCheckingPieces(self: *Self, king_color: pieceInfo.Color) u32 {
        const king_bb = self.board.getColorBitboard(king_color) & self.board.getKindBitboard(.king);
        if (king_bb == 0) return 0;

        const king_square: u6 = @intCast(@ctz(king_bb));
        const attacker_color = if (king_color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
        const attacker_bb = self.board.getColorBitboard(attacker_color);
        const occupied = self.board.occupied();

        var count: u32 = 0;

        // Check pawns
        const pawn_attacks = self.getPawnAttacks(king_square, king_color);
        if ((pawn_attacks & attacker_bb & self.board.getKindBitboard(.pawn)) != 0) {
            count += @popCount(pawn_attacks & attacker_bb & self.board.getKindBitboard(.pawn));
        }

        // Check knights
        const knight_attacks = self.getKnightAttacks(king_square);
        if ((knight_attacks & attacker_bb & self.board.getKindBitboard(.knight)) != 0) {
            count += @popCount(knight_attacks & attacker_bb & self.board.getKindBitboard(.knight));
        }

        // Check bishops/queens (diagonal)
        const bishop_attacks = self.getBishopAttacks(king_square, occupied);
        if ((bishop_attacks & attacker_bb & (self.board.getKindBitboard(.bishop) | self.board.getKindBitboard(.queen))) != 0) {
            count += @popCount(bishop_attacks & attacker_bb & (self.board.getKindBitboard(.bishop) | self.board.getKindBitboard(.queen)));
        }

        // Check rooks/queens (straight)
        const rook_attacks = self.getRookAttacks(king_square, occupied);
        if ((rook_attacks & attacker_bb & (self.board.getKindBitboard(.rook) | self.board.getKindBitboard(.queen))) != 0) {
            count += @popCount(rook_attacks & attacker_bb & (self.board.getKindBitboard(.rook) | self.board.getKindBitboard(.queen)));
        }

        return count;
    }

    /// Check if a move gives direct check.
    ///
    /// A "direct check" means the piece that just moved is the one giving check,
    /// as opposed to a "discovered check" where moving one piece uncovers an
    /// attack from another piece.
    ///
    /// This is determined by checking if the moved piece (now at the 'to' square)
    /// can attack the opponent's king from its new position.
    ///
    /// Parameters:
    ///   - move: The move that was just made
    ///   - moving_color: The color of the side that made the move
    ///   - opponent_color: The color of the opponent (whose king we check)
    ///
    /// Returns: true if the piece that moved is directly attacking the opponent's king
    fn isDirectCheck(self: *Self, move: Move, moving_color: pieceInfo.Color, opponent_color: pieceInfo.Color) bool {
        const king_bb = self.board.getColorBitboard(opponent_color) & self.board.getKindBitboard(.king);
        if (king_bb == 0) return false;

        const king_square: u6 = @intCast(@ctz(king_bb));
        const piece_type = self.board.getPieceAt(move.to, moving_color) orelse return false;
        const occupied = self.board.occupied();
        const move_to: u6 = @intCast(move.to);

        const result = switch (piece_type) {
            .pawn => blk: {
                const pawn_attacks = self.getPawnAttacks(king_square, opponent_color);
                break :blk (pawn_attacks & (@as(u64, 1) << move_to)) != 0;
            },
            .knight => blk: {
                const knight_attacks = self.getKnightAttacks(king_square);
                break :blk (knight_attacks & (@as(u64, 1) << move_to)) != 0;
            },
            .bishop => blk: {
                const bishop_attacks = self.getBishopAttacks(king_square, occupied);
                break :blk (bishop_attacks & (@as(u64, 1) << move_to)) != 0;
            },
            .rook => blk: {
                const rook_attacks = self.getRookAttacks(king_square, occupied);
                break :blk (rook_attacks & (@as(u64, 1) << move_to)) != 0;
            },
            .queen => blk: {
                const bishop_attacks = self.getBishopAttacks(king_square, occupied);
                const rook_attacks = self.getRookAttacks(king_square, occupied);
                const queen_attacks = bishop_attacks | rook_attacks;
                break :blk (queen_attacks & (@as(u64, 1) << move_to)) != 0;
            },
            .king => false, // Kings don't give check
        };
        return result;
    }

    /// Perft divide - Shows the number of nodes for each root move
    pub fn perftDivide(self: *Self, depth: u32, writer: anytype) UciError!u64 {
        var moves = MoveList.init();
        try self.generateLegalMoves(&moves);

        var total_nodes: u64 = 0;

        for (moves.slice()) |move| {
            // Save state
            const old_board = self.board;
            const old_hash = self.zobrist_hasher.zobrist_hash;

            // Make move
            self.applyMoveUnchecked(move);

            // Count nodes
            const nodes = if (depth <= 1) 1 else try self.perft(depth - 1);
            total_nodes += nodes;

            // Print result
            writer.print("{s}: {d}\n", .{ move, nodes }) catch return UciError.IOError;

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;
        }

        writer.print("\nTotal nodes: {d}\n", .{total_nodes}) catch return UciError.IOError;
        return total_nodes;
    }

    /// Check if the given color's king is currently in check
    inline fn isInCheck(self: *Self, color: pieceInfo.Color) bool {
        const king_bb = self.board.getColorBitboard(color) & self.board.getKindBitboard(.king);
        if (king_bb == 0) return false;

        const king_square: u6 = @intCast(@ctz(king_bb));
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        return self.isSquareAttackedBy(king_square, opponent_color);
    }

    /// Check if a square is attacked by any piece of the given color
    fn isSquareAttackedBy(self: *Self, square: u6, attacker_color: pieceInfo.Color) bool {
        const attacker_bb = self.board.getColorBitboard(attacker_color);

        // Check pawn attacks
        const pawn_attacks = self.getPawnAttacks(square, if (attacker_color == .white) .black else .white);
        if ((pawn_attacks & attacker_bb & self.board.getKindBitboard(.pawn)) != 0) {
            return true;
        }

        // Check knight attacks
        const knight_attacks = self.getKnightAttacks(square);
        if ((knight_attacks & attacker_bb & self.board.getKindBitboard(.knight)) != 0) {
            return true;
        }

        // Check king attacks
        const king_attacks = self.getKingAttacks(square);
        if ((king_attacks & attacker_bb & self.board.getKindBitboard(.king)) != 0) {
            return true;
        }

        // Check sliding pieces (bishop, rook, queen)
        const occupied = self.board.occupied();

        const bishop_attacks = self.getBishopAttacks(square, occupied);
        if ((bishop_attacks & attacker_bb & (self.board.getKindBitboard(.bishop) | self.board.getKindBitboard(.queen))) != 0) {
            return true;
        }

        const rook_attacks = self.getRookAttacks(square, occupied);
        if ((rook_attacks & attacker_bb & (self.board.getKindBitboard(.rook) | self.board.getKindBitboard(.queen))) != 0) {
            return true;
        }

        return false;
    }

    /// Apply a move without checking legality.
    ///
    /// This function assumes the move is pseudo-legal and updates the board state
    /// accordingly. It handles:
    /// - Regular captures (clearing destination square)
    /// - En passant captures (clearing captured pawn square)
    /// - Castling (moving the rook)
    /// - Pawn promotions
    /// - En passant square updates (for double pawn pushes)
    /// - Castling rights updates
    /// - Turn switching
    ///
    /// IMPORTANT: This does NOT:
    /// - Check if the move is legal (king may be in check after)
    /// - Update the zobrist hash (caller must do this)
    /// - Validate the move exists or is pseudo-legal
    ///
    /// Use only:
    /// - During move generation legality testing (with save/restore)
    /// - After verifying move legality explicitly
    /// - When implementing makeMove (which handles zobrist)
    ///
    /// Parameters:
    ///   - move: The move to apply (must be pseudo-legal)
    fn applyMoveUnchecked(self: *Self, move: Move) void {
        const color = self.board.move;
        const piece_type = self.board.getPieceAt(move.from, color) orelse return;
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        const captured_piece = self.board.getPieceAt(move.to, opponent_color);

        // Handle captures - must clear the destination square first
        self.board.clearSquare(move.to);

        // Handle en passant capture
        if (piece_type == .pawn and self.board.en_passant_square == move.to) {
            const ep_capture_square = if (color == .white) move.to - 8 else move.to + 8;
            self.board.clearSquare(ep_capture_square);
        }

        // Handle castling
        if (piece_type == .king) {
            const from_file = move.from % 8;
            const to_file = move.to % 8;

            // Kingside castling
            if (from_file == 4 and to_file == 6) {
                const rook_from = move.from + 3;
                const rook_to = move.from + 1;
                self.board.clearSquare(rook_from);
                self.board.setPieceAt(rook_to, color, .rook);
            }
            // Queenside castling
            else if (from_file == 4 and to_file == 2) {
                const rook_from = move.from - 4;
                const rook_to = move.from - 1;
                self.board.clearSquare(rook_from);
                self.board.setPieceAt(rook_to, color, .rook);
            }
        }

        // Move the piece
        self.board.clearSquare(move.from);
        const final_piece = if (move.promotion) |promo| promo else piece_type;
        self.board.setPieceAt(move.to, color, final_piece);

        // Update en passant square
        self.board.en_passant_square = null;
        if (piece_type == .pawn) {
            const from_rank = move.from / 8;
            const to_rank = move.to / 8;
            if (@as(i16, to_rank) - @as(i16, from_rank) == 2 or @as(i16, to_rank) - @as(i16, from_rank) == -2) {
                self.board.en_passant_square = if (color == .white) move.from + 8 else move.from - 8;
            }
        }

        // Update castling rights
        if (piece_type == .king) {
            if (color == .white) {
                self.board.castle_rights.white_kingside = false;
                self.board.castle_rights.white_queenside = false;
            } else {
                self.board.castle_rights.black_kingside = false;
                self.board.castle_rights.black_queenside = false;
            }
        } else if (piece_type == .rook) {
            if (color == .white) {
                if (move.from == 0) self.board.castle_rights.white_queenside = false;
                if (move.from == 7) self.board.castle_rights.white_kingside = false;
            } else {
                if (move.from == 56) self.board.castle_rights.black_queenside = false;
                if (move.from == 63) self.board.castle_rights.black_kingside = false;
            }
        }

        // If a rook is captured, remove castling rights
        if (captured_piece) |captured| {
            if (captured == .rook) {
                if (opponent_color == .white) {
                    if (move.to == 0) self.board.castle_rights.white_queenside = false;
                    if (move.to == 7) self.board.castle_rights.white_kingside = false;
                } else {
                    if (move.to == 56) self.board.castle_rights.black_queenside = false;
                    if (move.to == 63) self.board.castle_rights.black_kingside = false;
                }
            }
        }

        // Switch turn
        self.board.move = opponent_color;
    }

    /// Generate all pseudo-legal moves (may leave king in check)
    fn generatePseudoLegalMoves(self: *Self, moves: *MoveList) !void {
        const color = self.board.move;
        const our_pieces = self.board.getColorBitboard(color);
        const opponent_pieces = self.board.getColorBitboard(if (color == .white) .black else .white);
        const occupied = self.board.occupied();

        // Generate pawn moves
        try self.generatePawnMoves(moves, color, our_pieces, opponent_pieces, occupied);

        // Generate knight moves
        try self.generateKnightMoves(moves, color, our_pieces);

        // Generate bishop moves
        try self.generateBishopMoves(moves, color, our_pieces, occupied);

        // Generate rook moves
        try self.generateRookMoves(moves, color, our_pieces, occupied);

        // Generate queen moves
        try self.generateQueenMoves(moves, color, our_pieces, occupied);

        // Generate king moves
        try self.generateKingMoves(moves, color, our_pieces, opponent_pieces, occupied);
    }

    fn generatePawnMoves(self: *Self, moves: *MoveList, color: pieceInfo.Color, our_pieces: u64, opponent_pieces: u64, occupied: u64) !void {
        const pawns = our_pieces & self.board.getKindBitboard(.pawn);
        const empty = ~occupied;

        if (color == .white) {
            // White pawns move up the board (towards rank 8)
            const promo_rank_mask: u64 = RANK_8_MASK;

            // Single pushes
            const push_one = (pawns << 8) & empty;
            const push_one_no_promo = push_one & ~promo_rank_mask;
            const push_one_promo = push_one & promo_rank_mask;

            // Generate single push moves (non-promotion)
            var bb = push_one_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to - 8, to, null));
            }

            // Generate promotion moves from single pushes
            bb = push_one_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to - 8;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // Double pushes (only from rank 2, so pushed pawns are on rank 3)
            const rank_3_mask: u64 = 0x0000000000FF0000;
            const push_two = ((push_one & rank_3_mask) << 8) & empty;
            bb = push_two;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to - 16, to, null));
            }

            // Left captures (not A-file)
            const left_captures = ((pawns & NOT_A_FILE) << 7) & opponent_pieces;
            const left_captures_no_promo = left_captures & ~promo_rank_mask;
            const left_captures_promo = left_captures & promo_rank_mask;

            bb = left_captures_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to - 7, to, null));
            }

            bb = left_captures_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to - 7;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // Right captures (not H-file)
            const right_captures = ((pawns & NOT_H_FILE) << 9) & opponent_pieces;
            const right_captures_no_promo = right_captures & ~promo_rank_mask;
            const right_captures_promo = right_captures & promo_rank_mask;

            bb = right_captures_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to - 9, to, null));
            }

            bb = right_captures_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to - 9;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // En passant
            if (self.board.en_passant_square) |ep_sq| {
                const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
                const ep_left = ((pawns & NOT_A_FILE) << 7) & ep_bb;
                const ep_right = ((pawns & NOT_H_FILE) << 9) & ep_bb;

                if (ep_left != 0) {
                    moves.append(Move.init(ep_sq - 7, ep_sq, null));
                }
                if (ep_right != 0) {
                    moves.append(Move.init(ep_sq - 9, ep_sq, null));
                }
            }
        } else {
            // Black pawns move down the board (towards rank 1)
            const promo_rank_mask: u64 = RANK_1_MASK;

            // Single pushes
            const push_one = (pawns >> 8) & empty;
            const push_one_no_promo = push_one & ~promo_rank_mask;
            const push_one_promo = push_one & promo_rank_mask;

            // Generate single push moves (non-promotion)
            var bb = push_one_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to + 8, to, null));
            }

            // Generate promotion moves from single pushes
            bb = push_one_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to + 8;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // Double pushes (only from rank 7, so pushed pawns are on rank 6)
            const rank_6_mask: u64 = 0x0000FF0000000000;
            const push_two = ((push_one & rank_6_mask) >> 8) & empty;
            bb = push_two;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to + 16, to, null));
            }

            // Left captures (not H-file for black, moving down-left)
            const left_captures = ((pawns & NOT_H_FILE) >> 7) & opponent_pieces;
            const left_captures_no_promo = left_captures & ~promo_rank_mask;
            const left_captures_promo = left_captures & promo_rank_mask;

            bb = left_captures_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to + 7, to, null));
            }

            bb = left_captures_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to + 7;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // Right captures (not A-file for black, moving down-right)
            const right_captures = ((pawns & NOT_A_FILE) >> 9) & opponent_pieces;
            const right_captures_no_promo = right_captures & ~promo_rank_mask;
            const right_captures_promo = right_captures & promo_rank_mask;

            bb = right_captures_no_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                moves.append(Move.init(to + 9, to, null));
            }

            bb = right_captures_promo;
            while (bb != 0) {
                const to: u6 = @intCast(@ctz(bb));
                bb &= bb - 1;
                const from: u6 = to + 9;
                moves.append(Move.init(from, to, .queen));
                moves.append(Move.init(from, to, .knight));
                moves.append(Move.init(from, to, .rook));
                moves.append(Move.init(from, to, .bishop));
            }

            // En passant
            if (self.board.en_passant_square) |ep_sq| {
                const ep_bb: u64 = @as(u64, 1) << @intCast(ep_sq);
                const ep_left = ((pawns & NOT_H_FILE) >> 7) & ep_bb;
                const ep_right = ((pawns & NOT_A_FILE) >> 9) & ep_bb;

                if (ep_left != 0) {
                    moves.append(Move.init(ep_sq + 7, ep_sq, null));
                }
                if (ep_right != 0) {
                    moves.append(Move.init(ep_sq + 9, ep_sq, null));
                }
            }
        }
    }

    fn generateKnightMoves(self: *Self, moves: *MoveList, _: pieceInfo.Color, our_pieces: u64) !void {
        const knights = our_pieces & self.board.getKindBitboard(.knight);
        var knight_bb = knights;

        while (knight_bb != 0) {
            const from: u6 = @intCast(@ctz(knight_bb));
            knight_bb &= knight_bb - 1;

            const attacks = self.getKnightAttacks(from);
            var attack_bb = attacks & ~our_pieces;

            while (attack_bb != 0) {
                const to: u6 = @intCast(@ctz(attack_bb));
                attack_bb &= attack_bb - 1;
                moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateBishopMoves(self: *Self, moves: *MoveList, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
        const bishops = our_pieces & self.board.getKindBitboard(.bishop);
        var bishop_bb = bishops;

        while (bishop_bb != 0) {
            const from: u6 = @intCast(@ctz(bishop_bb));
            bishop_bb &= bishop_bb - 1;

            const attacks = self.getBishopAttacks(from, occupied);
            var attack_bb = attacks & ~our_pieces;

            while (attack_bb != 0) {
                const to: u6 = @intCast(@ctz(attack_bb));
                attack_bb &= attack_bb - 1;
                moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateRookMoves(self: *Self, moves: *MoveList, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
        const rooks = our_pieces & self.board.getKindBitboard(.rook);
        var rook_bb = rooks;

        while (rook_bb != 0) {
            const from: u6 = @intCast(@ctz(rook_bb));
            rook_bb &= rook_bb - 1;

            const attacks = self.getRookAttacks(from, occupied);
            var attack_bb = attacks & ~our_pieces;

            while (attack_bb != 0) {
                const to: u6 = @intCast(@ctz(attack_bb));
                attack_bb &= attack_bb - 1;
                moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateQueenMoves(self: *Self, moves: *MoveList, _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
        const queens = our_pieces & self.board.getKindBitboard(.queen);
        var queen_bb = queens;

        while (queen_bb != 0) {
            const from: u6 = @intCast(@ctz(queen_bb));
            queen_bb &= queen_bb - 1;

            const bishop_attacks = self.getBishopAttacks(from, occupied);
            const rook_attacks = self.getRookAttacks(from, occupied);
            const attacks = bishop_attacks | rook_attacks;
            var attack_bb = attacks & ~our_pieces;

            while (attack_bb != 0) {
                const to: u6 = @intCast(@ctz(attack_bb));
                attack_bb &= attack_bb - 1;
                moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateKingMoves(self: *Self, moves: *MoveList, color: pieceInfo.Color, our_pieces: u64, _: u64, occupied: u64) !void {
        const kings = our_pieces & self.board.getKindBitboard(.king);
        if (kings == 0) return;

        const from: u6 = @intCast(@ctz(kings));

        // Regular king moves
        const attacks = self.getKingAttacks(from);
        var attack_bb = attacks & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            moves.append(Move.init(from, to, null));
        }

        // Castling - only add if not currently in check and path is clear
        // Check attacks will be verified during move legality testing
        const opponent = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        if (color == .white) {
            // White kingside
            if (self.board.castle_rights.white_kingside) {
                if ((occupied & ((@as(u64, 1) << 5) | (@as(u64, 1) << 6))) == 0 and
                    !self.isSquareAttackedBy(4, opponent) and
                    !self.isSquareAttackedBy(5, opponent))
                {
                    moves.append(Move.init(4, 6, null));
                }
            }
            // White queenside
            if (self.board.castle_rights.white_queenside) {
                if ((occupied & ((@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3))) == 0 and
                    !self.isSquareAttackedBy(4, opponent) and
                    !self.isSquareAttackedBy(3, opponent))
                {
                    moves.append(Move.init(4, 2, null));
                }
            }
        } else {
            // Black kingside
            if (self.board.castle_rights.black_kingside) {
                if ((occupied & ((@as(u64, 1) << 61) | (@as(u64, 1) << 62))) == 0 and
                    !self.isSquareAttackedBy(60, opponent) and
                    !self.isSquareAttackedBy(61, opponent))
                {
                    moves.append(Move.init(60, 62, null));
                }
            }
            // Black queenside
            if (self.board.castle_rights.black_queenside) {
                if ((occupied & ((@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59))) == 0 and
                    !self.isSquareAttackedBy(60, opponent) and
                    !self.isSquareAttackedBy(59, opponent))
                {
                    moves.append(Move.init(60, 58, null));
                }
            }
        }
    }

    // Attack pattern generators
    inline fn getKnightAttacks(self: *Self, square: u6) u64 {
        _ = self;
        return KNIGHT_ATTACKS[square];
    }

    inline fn getKingAttacks(self: *Self, square: u6) u64 {
        _ = self;
        return KING_ATTACKS[square];
    }

    inline fn getPawnAttacks(self: *Self, square: u6, pawn_color: pieceInfo.Color) u64 {
        _ = self;
        const sq: u64 = @as(u64, 1) << square;
        const file = square % 8;

        var attacks: u64 = 0;

        if (pawn_color == .white) {
            if (file > 0 and square < 56) attacks |= sq << 7; // Up-left
            if (file < 7 and square < 56) attacks |= sq << 9; // Up-right
        } else {
            if (file > 0 and square >= 8) attacks |= sq >> 9; // Down-left
            if (file < 7 and square >= 8) attacks |= sq >> 7; // Down-right
        }

        return attacks;
    }

    inline fn getRookAttacks(self: *Self, square: u6, occupied: u64) u64 {
        _ = self;
        const mask = ROOK_MASKS[square];
        const masked_occupied = occupied & mask;
        const index = (masked_occupied *% ROOK_MAGICS[square]) >> @intCast(ROOK_SHIFTS[square]);
        return ROOK_ATTACKS[square][index];
    }
    inline fn getBishopAttacks(self: *Self, square: u6, occupied: u64) u64 {
        _ = self;
        const mask = BISHOP_MASKS[square];
        const masked_occupied = occupied & mask;
        const index = (masked_occupied *% BISHOP_MAGICS[square]) >> @intCast(BISHOP_SHIFTS[square]);
        return BISHOP_ATTACKS[square][index];
    }
    inline fn rankFileToIndex(rank: u8, file: u8) u8 {
        return rankFileToSquare(rank - '1', file - 'a');
    }

    inline fn rankFileToSquare(rank: u8, file: u8) u8 {
        return rank * 8 + file;
    }

    inline fn getTurn(self: Self) pieceInfo.Color {
        return self.board.move;
    }

    pub fn format(
        self: Self,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("  +---+---+---+---+---+---+---+---+\n");

        var rank: i32 = 7;
        while (rank >= 0) : (rank -= 1) {
            try writer.print("{d} ", .{rank + 1});
            for (0..8) |file| {
                const index: u8 = @intCast(@as(u32, @intCast(rank)) * 8 + file);
                const white_piece = self.board.getPieceAt(index, .white);
                const black_piece = self.board.getPieceAt(index, .black);

                try writer.writeAll("| ");
                if (white_piece) |p| {
                    try writer.print("{c}", .{p.getName()});
                } else if (black_piece) |p| {
                    const name = p.getName();
                    try writer.print("{c}", .{std.ascii.toLower(name)});
                } else {
                    try writer.writeAll(" ");
                }
                try writer.writeAll(" ");
            }
            try writer.writeAll("|\n");
            try writer.writeAll("  +---+---+---+---+---+---+---+---+\n");
        }

        try writer.writeAll("    a   b   c   d   e   f   g   h\n");
    }

    pub fn getFenString(self: Self, allocator: std.mem.Allocator) UciError![]u8 {
        return getFenStringGenericError(self, allocator) catch UciError.IOError;
    }

    fn getFenStringGenericError(self: Self, allocator: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        defer buffer.deinit();
        const writer = buffer.writer();

        for (0..8) |rank| {
            const actual_rank = 7 - rank;
            var empty_count: u8 = 0;

            for (0..8) |file| {
                const index = actual_rank * 8 + file;
                const white_piece = self.board.getPieceAt(@intCast(index), .white);
                const black_piece = self.board.getPieceAt(@intCast(index), .black);

                if (white_piece) |p| {
                    if (empty_count > 0) {
                        try writer.print("{}", .{empty_count});
                        empty_count = 0;
                    }
                    try writer.writeByte(p.getName());
                } else if (black_piece) |p| {
                    if (empty_count > 0) {
                        try writer.print("{}", .{empty_count});
                        empty_count = 0;
                    }
                    try writer.writeByte(std.ascii.toLower(p.getName()));
                } else {
                    empty_count += 1;
                }
            }

            if (empty_count > 0) {
                try writer.print("{}", .{empty_count});
            }

            if (rank != 7) {
                try writer.writeByte('/');
            }
        }

        // Turn
        const turn: u8 = if (self.board.move == .white) 'w' else 'b';
        try writer.writeByte(' ');
        try writer.writeByte(turn);

        // Castling rights
        try writer.writeByte(' ');
        var any_castle = false;
        if (self.board.castle_rights.white_kingside) {
            try writer.writeByte('K');
            any_castle = true;
        }
        if (self.board.castle_rights.white_queenside) {
            try writer.writeByte('Q');
            any_castle = true;
        }
        if (self.board.castle_rights.black_kingside) {
            try writer.writeByte('k');
            any_castle = true;
        }
        if (self.board.castle_rights.black_queenside) {
            try writer.writeByte('q');
            any_castle = true;
        }
        if (!any_castle) try writer.writeByte('-');

        // En passant
        try writer.writeByte(' ');
        if (self.board.en_passant_square) |sq| {
            const file = sq % 8 + 'a';
            const rank = sq / 8 + '1';
            try writer.writeByte(file);
            try writer.writeByte(rank);
        } else {
            try writer.writeByte('-');
        }

        // Halfmove and fullmove
        try writer.print(" {} {}", .{ self.board.halfmove_clock, self.board.fullmove_number });
        return buffer.toOwnedSlice();
    }
};

pub const CastleRights = struct {
    white_kingside: bool = false,
    white_queenside: bool = false,
    black_kingside: bool = false,
    black_queenside: bool = false,
};

pub const BitBoard = struct {
    const Self = @This();

    color_sets: [2]u64 = undefined,
    kind_sets: [6]u64 = undefined,

    move: pieceInfo.Color = pieceInfo.Color.white,

    castle_rights: CastleRights = CastleRights{},

    en_passant_square: ?u8 = null,
    halfmove_clock: u8 = 0,
    fullmove_number: u16 = 1,

    pub fn init() Self {
        return Self{};
    }

    fn getColorBitboard(self: Self, color: pieceInfo.Color) u64 {
        return self.color_sets[@intFromEnum(color)];
    }

    fn getKindBitboard(self: Self, kind: pieceInfo.Type) u64 {
        return self.kind_sets[@intFromEnum(kind)];
    }

    fn clearSquare(self: *Self, index: u8) void {
        const mask = @as(u64, 1) << @intCast(index);

        for (0..2) |i| {
            self.color_sets[i] &= ~mask;
        }
        for (0..6) |i| {
            self.kind_sets[i] &= ~mask;
        }
    }

    fn setPieceAt(self: *Self, index: u8, color: pieceInfo.Color, kind: pieceInfo.Type) void {
        const mask = @as(u64, 1) << @intCast(index);
        self.color_sets[@intFromEnum(color)] |= mask;
        self.kind_sets[@intFromEnum(kind)] |= mask;
    }

    pub fn getPieceAt(self: Self, index: u8, color: pieceInfo.Color) ?pieceInfo.Type {
        const color_mask = @as(u64, 1) << @intCast(index);
        if ((self.color_sets[@intFromEnum(color)] & color_mask) == 0)
            return null;

        for (self.kind_sets, 0..) |kind_mask, i| {
            if ((kind_mask & color_mask) != 0)
                return @enumFromInt(i);
        }

        return null;
    }

    /// Check if there are any pawns of the given color adjacent to the specified square.
    /// This is used for determining if an en passant capture is possible.
    ///
    /// Parameters:
    ///   - board: The current board position
    ///   - ep_sq: The en passant square to check
    ///   - color: The color of the pawns to look for
    ///
    /// Returns: true if there are adjacent pawns that could make an en passant capture
    pub fn hasAdjacentPawn(self: Self, ep_sq: u8, color: pieceInfo.Color) bool {
        const file = ep_sq % 8;

        var result = false;

        if (file > 0) {
            const left = ep_sq - 1;
            if (self.getPieceAt(left, color)) |pt| {
                if (pt == .pawn) result = true;
            }
        }

        if (file < 7) {
            const right = ep_sq + 1;
            if (self.getPieceAt(right, color)) |pt| {
                if (pt == .pawn) result = true;
            }
        }

        return result;
    }

    pub fn whiteToMove(self: Self) bool {
        return self.move == pieceInfo.Color.white;
    }

    inline fn occupied(self: Self) u64 {
        return self.color_sets[0] | self.color_sets[1];
    }

    inline fn empty(self: Self) u64 {
        return ~self.occupied();
    }
};
