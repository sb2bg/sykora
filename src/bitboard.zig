const std = @import("std");
const UciError = @import("uci_error.zig").UciError;
const pieceInfo = @import("piece.zig");
const fen = @import("fen.zig");
const zobrist = @import("zobrist.zig");
const ZobristHasher = zobrist.ZobristHasher;

// Pre-computed attack tables for knight and king
const KNIGHT_ATTACKS = initKnightAttacks();
const KING_ATTACKS = initKingAttacks();

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

    pub fn makeStrMove(self: *Self, move: []const u8) UciError!void {
        if (move.len != 4) return error.InvalidArgument;

        const from_file = std.ascii.toLower(move[0]);
        const from_rank = move[1];
        const to_file = std.ascii.toLower(move[2]);
        const to_rank = move[3];

        if (from_file < 'a' or from_file > 'h' or from_rank < '1' or from_rank > '8' or
            to_file < 'a' or to_file > 'h' or to_rank < '1' or to_rank > '8')
        {
            return error.InvalidArgument;
        }

        const from_index = rankFileToIndex(from_rank, from_file);
        const to_index = rankFileToIndex(to_rank, to_file);

        const color = self.getTurn();
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
        const piece_type = self.board.getPieceAt(from_index, color) orelse return error.InvalidMove;
        const captured = self.board.getPieceAt(to_index, opponent_color);
        const is_capture = captured != null;

        const prev_castle = self.board.castle_rights;
        const prev_ep = self.board.en_passant_square;

        // Handle en passant capture - must clear the captured pawn's square
        if (piece_type == .pawn) {
            if (self.board.en_passant_square) |en_passant_square| {
                if (en_passant_square == to_index) {
                    // Clear the captured pawn (which is one rank behind the en passant square)
                    const captured_pawn_square = if (color == .white) to_index - 8 else to_index + 8;
                    self.board.clearSquare(captured_pawn_square);
                }
            }
        }

        self.board.clearSquare(to_index);
        self.board.clearSquare(from_index);
        self.board.setPieceAt(to_index, color, piece_type);

        // Update en passant square for double pawn pushes
        self.board.en_passant_square = null;
        if (piece_type == .pawn) {
            const pawn_from_rank = from_index / 8;
            const pawn_to_rank = to_index / 8;
            // Check if it's a double pawn push
            if (@as(i16, @intCast(pawn_to_rank)) - @as(i16, @intCast(pawn_from_rank)) == 2) {
                // White double push (e.g., e2-e4), en passant square is e3
                self.board.en_passant_square = from_index + 8;
            } else if (@as(i16, @intCast(pawn_to_rank)) - @as(i16, @intCast(pawn_from_rank)) == -2) {
                // Black double push (e.g., e7-e5), en passant square is e6
                self.board.en_passant_square = from_index - 8;
            }
        }

        self.board.move = if (self.board.whiteToMove()) pieceInfo.Color.black else pieceInfo.Color.white;
        self.board.fullmove_number += if (color == .black) 1 else 0;

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
                if (from_index == 0) self.board.castle_rights.white_queenside = false;
                if (from_index == 7) self.board.castle_rights.white_kingside = false;
            } else {
                if (from_index == 56) self.board.castle_rights.black_queenside = false;
                if (from_index == 63) self.board.castle_rights.black_kingside = false;
            }
        }
        // If a rook is captured, remove castling rights
        if (captured == .rook) {
            const opponent = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
            if (opponent == .white) {
                if (to_index == 0) self.board.castle_rights.white_queenside = false;
                if (to_index == 7) self.board.castle_rights.white_kingside = false;
            } else {
                if (to_index == 56) self.board.castle_rights.black_queenside = false;
                if (to_index == 63) self.board.castle_rights.black_kingside = false;
            }
        }

        if (piece_type == .pawn or is_capture) {
            self.board.halfmove_clock = 0;
        } else {
            self.board.halfmove_clock += 1;
        }

        self.zobrist_hasher.updateHash(
            from_index,
            to_index,
            piece_type,
            color,
            captured,
            prev_castle,
            self.board.castle_rights,
            prev_ep,
            self.board.en_passant_square,
        );
    }

    pub fn makeMove(self: *Self, move: Move) UciError!void {
        _ = move;
        _ = self;
        return error.Unimplemented;
    }

    /// Generate all legal moves for the current position
    pub fn generateLegalMoves(self: *Self, allocator: std.mem.Allocator) ![]Move {
        var moves = std.ArrayList(Move).init(allocator);
        errdefer moves.deinit();

        const pseudo_legal = try self.generatePseudoLegalMoves(allocator);
        defer allocator.free(pseudo_legal);

        const color = self.board.move;

        // Filter out moves that leave the king in check
        // Optimize by saving/restoring state only once per move
        for (pseudo_legal) |move| {
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
                try moves.append(move);
            }
        }

        return moves.toOwnedSlice();
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
    pub fn perft(self: *Self, depth: u32, allocator: std.mem.Allocator) UciError!u64 {
        if (depth == 0) {
            return 1;
        }

        // Generate pseudo-legal moves once
        const pseudo_legal = try self.generatePseudoLegalMoves(allocator);
        defer allocator.free(pseudo_legal);

        var nodes: u64 = 0;
        const color = self.board.move;

        // Test each pseudo-legal move for legality
        for (pseudo_legal) |move| {
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
                nodes += try self.perft(depth - 1, allocator);
            }

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;
        }

        return nodes;
    }

    /// Perft with detailed statistics
    pub fn perftWithStats(self: *Self, depth: u32, allocator: std.mem.Allocator, stats: *PerftStats) UciError!void {
        if (depth == 0) {
            stats.nodes = 1;
            return;
        }

        // Generate pseudo-legal moves once
        const pseudo_legal = try self.generatePseudoLegalMoves(allocator);
        defer allocator.free(pseudo_legal);

        const moving_color = self.board.move;
        const opponent_color = if (moving_color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        if (depth == 1) {
            // At depth 1, count move types for legal moves only
            for (pseudo_legal) |move| {
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
                    const opponent_legal = try self.perft(1, allocator);
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
        for (pseudo_legal) |move| {
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
            try self.perftWithStats(depth - 1, allocator, &child_stats);
            stats.add(child_stats);

            // Restore state
            self.board = old_board;
            self.zobrist_hasher.zobrist_hash = old_hash;
        }
    }

    /// Count how many pieces are giving check to the king of the specified color
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

    /// Check if a move gives direct check (the piece that moved is giving check)
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
    pub fn perftDivide(self: *Self, depth: u32, allocator: std.mem.Allocator, writer: anytype) UciError!u64 {
        const moves = try self.generateLegalMoves(allocator);
        defer allocator.free(moves);

        var total_nodes: u64 = 0;

        for (moves) |move| {
            // Save state
            const old_board = self.board;
            const old_hash = self.zobrist_hasher.zobrist_hash;

            // Make move
            self.applyMoveUnchecked(move);

            // Count nodes
            const nodes = if (depth <= 1) 1 else try self.perft(depth - 1, allocator);
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

    /// Apply a move without checking legality (used for legal move testing)
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
    fn generatePseudoLegalMoves(self: *Self, allocator: std.mem.Allocator) ![]Move {
        var moves = std.ArrayList(Move).init(allocator);
        errdefer moves.deinit();

        const color = self.board.move;
        const our_pieces = self.board.getColorBitboard(color);
        const opponent_pieces = self.board.getColorBitboard(if (color == .white) .black else .white);
        const occupied = self.board.occupied();

        // Generate pawn moves
        try self.generatePawnMoves(&moves, color, our_pieces, opponent_pieces, occupied);

        // Generate knight moves
        try self.generateKnightMoves(&moves, color, our_pieces);

        // Generate bishop moves
        try self.generateBishopMoves(&moves, color, our_pieces, occupied);

        // Generate rook moves
        try self.generateRookMoves(&moves, color, our_pieces, occupied);

        // Generate queen moves
        try self.generateQueenMoves(&moves, color, our_pieces, occupied);

        // Generate king moves
        try self.generateKingMoves(&moves, color, our_pieces, opponent_pieces, occupied);

        return moves.toOwnedSlice();
    }

    fn generatePawnMoves(self: *Self, moves: *std.ArrayList(Move), color: pieceInfo.Color, our_pieces: u64, opponent_pieces: u64, occupied: u64) !void {
        const pawns = our_pieces & self.board.getKindBitboard(.pawn);
        const direction: i8 = if (color == .white) 8 else -8;
        const start_rank: u8 = if (color == .white) 1 else 6;
        const promo_rank: u8 = if (color == .white) 7 else 0;

        var pawn_bb = pawns;
        while (pawn_bb != 0) {
            const from: u6 = @intCast(@ctz(pawn_bb));
            pawn_bb &= pawn_bb - 1; // Clear least significant bit

            const rank = from / 8;
            const file = from % 8;

            // Single push
            const to_single: i16 = @as(i16, from) + direction;
            if (to_single >= 0 and to_single < 64) {
                const to_sq: u6 = @intCast(to_single);
                if ((occupied & (@as(u64, 1) << to_sq)) == 0) {
                    const to_rank = to_sq / 8;
                    if (to_rank == promo_rank) {
                        // Promotions - order: queen, knight, rook, bishop (MVV-LVA order)
                        try moves.append(Move.init(from, to_sq, .queen));
                        try moves.append(Move.init(from, to_sq, .knight));
                        try moves.append(Move.init(from, to_sq, .rook));
                        try moves.append(Move.init(from, to_sq, .bishop));
                    } else {
                        try moves.append(Move.init(from, to_sq, null));

                        // Double push
                        if (rank == start_rank) {
                            const to_double: i16 = @as(i16, from) + direction * 2;
                            const to_sq_double: u6 = @intCast(to_double);
                            if ((occupied & (@as(u64, 1) << to_sq_double)) == 0) {
                                try moves.append(Move.init(from, to_sq_double, null));
                            }
                        }
                    }
                }
            }

            // Captures - left and right
            if (file > 0) {
                const to: i16 = @as(i16, from) + direction - 1;
                if (to >= 0 and to < 64) {
                    const to_sq: u6 = @intCast(to);
                    const to_bb = @as(u64, 1) << to_sq;

                    if ((opponent_pieces & to_bb) != 0) {
                        if (to_sq / 8 == promo_rank) {
                            try moves.append(Move.init(from, to_sq, .queen));
                            try moves.append(Move.init(from, to_sq, .knight));
                            try moves.append(Move.init(from, to_sq, .rook));
                            try moves.append(Move.init(from, to_sq, .bishop));
                        } else {
                            try moves.append(Move.init(from, to_sq, null));
                        }
                    } else if (self.board.en_passant_square) |ep_sq| {
                        if (ep_sq == to_sq) {
                            try moves.append(Move.init(from, to_sq, null));
                        }
                    }
                }
            }

            if (file < 7) {
                const to: i16 = @as(i16, from) + direction + 1;
                if (to >= 0 and to < 64) {
                    const to_sq: u6 = @intCast(to);
                    const to_bb = @as(u64, 1) << to_sq;

                    if ((opponent_pieces & to_bb) != 0) {
                        if (to_sq / 8 == promo_rank) {
                            try moves.append(Move.init(from, to_sq, .queen));
                            try moves.append(Move.init(from, to_sq, .knight));
                            try moves.append(Move.init(from, to_sq, .rook));
                            try moves.append(Move.init(from, to_sq, .bishop));
                        } else {
                            try moves.append(Move.init(from, to_sq, null));
                        }
                    } else if (self.board.en_passant_square) |ep_sq| {
                        if (ep_sq == to_sq) {
                            try moves.append(Move.init(from, to_sq, null));
                        }
                    }
                }
            }
        }
    }

    fn generateKnightMoves(self: *Self, moves: *std.ArrayList(Move), _: pieceInfo.Color, our_pieces: u64) !void {
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
                try moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateBishopMoves(self: *Self, moves: *std.ArrayList(Move), _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
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
                try moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateRookMoves(self: *Self, moves: *std.ArrayList(Move), _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
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
                try moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateQueenMoves(self: *Self, moves: *std.ArrayList(Move), _: pieceInfo.Color, our_pieces: u64, occupied: u64) !void {
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
                try moves.append(Move.init(from, to, null));
            }
        }
    }

    fn generateKingMoves(self: *Self, moves: *std.ArrayList(Move), color: pieceInfo.Color, our_pieces: u64, _: u64, occupied: u64) !void {
        const kings = our_pieces & self.board.getKindBitboard(.king);
        if (kings == 0) return;

        const from: u6 = @intCast(@ctz(kings));

        // Regular king moves
        const attacks = self.getKingAttacks(from);
        var attack_bb = attacks & ~our_pieces;

        while (attack_bb != 0) {
            const to: u6 = @intCast(@ctz(attack_bb));
            attack_bb &= attack_bb - 1;
            try moves.append(Move.init(from, to, null));
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
                    try moves.append(Move.init(4, 6, null));
                }
            }
            // White queenside
            if (self.board.castle_rights.white_queenside) {
                if ((occupied & ((@as(u64, 1) << 1) | (@as(u64, 1) << 2) | (@as(u64, 1) << 3))) == 0 and
                    !self.isSquareAttackedBy(4, opponent) and
                    !self.isSquareAttackedBy(3, opponent))
                {
                    try moves.append(Move.init(4, 2, null));
                }
            }
        } else {
            // Black kingside
            if (self.board.castle_rights.black_kingside) {
                if ((occupied & ((@as(u64, 1) << 61) | (@as(u64, 1) << 62))) == 0 and
                    !self.isSquareAttackedBy(60, opponent) and
                    !self.isSquareAttackedBy(61, opponent))
                {
                    try moves.append(Move.init(60, 62, null));
                }
            }
            // Black queenside
            if (self.board.castle_rights.black_queenside) {
                if ((occupied & ((@as(u64, 1) << 57) | (@as(u64, 1) << 58) | (@as(u64, 1) << 59))) == 0 and
                    !self.isSquareAttackedBy(60, opponent) and
                    !self.isSquareAttackedBy(59, opponent))
                {
                    try moves.append(Move.init(60, 58, null));
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

    fn getRookAttacks(self: *Self, square: u6, occupied: u64) u64 {
        _ = self;
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
    fn getBishopAttacks(self: *Self, square: u6, occupied: u64) u64 {
        _ = self;
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
