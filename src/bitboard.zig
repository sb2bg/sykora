const std = @import("std");
const builtin = @import("builtin");
const UciError = @import("uci_error.zig").UciError;
const pieceInfo = @import("piece.zig");
const fen = @import("fen.zig");
const zobrist = @import("zobrist.zig");
const ZobristHasher = zobrist.ZobristHasher;

const attacks = @import("board/attacks.zig");
const legality = @import("board/legality.zig");
const make_unmake = @import("board/make_unmake.zig");
const movegen = @import("board/movegen.zig");

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

/// Compact move representation packed into 16 bits for cache efficiency
/// Bit layout: [15:12] flags | [11:6] to square | [5:0] from square
/// Flags: 0=none, 1=knight promo, 2=bishop promo, 3=rook promo, 4=queen promo
///        5=en passant, 6=castling (can extend for other special moves)
pub const Move = struct {
    const Self = @This();

    // Flag constants
    pub const FLAG_NONE: u4 = 0;
    pub const FLAG_PROMO_KNIGHT: u4 = 1;
    pub const FLAG_PROMO_BISHOP: u4 = 2;
    pub const FLAG_PROMO_ROOK: u4 = 3;
    pub const FLAG_PROMO_QUEEN: u4 = 4;
    // Note: en passant and castling are detected by piece type + move pattern, not stored

    data: u16,

    pub inline fn init(from_sq: u8, to_sq: u8, promo: ?pieceInfo.Type) Self {
        const flag: u4 = if (promo) |p| switch (p) {
            .knight => FLAG_PROMO_KNIGHT,
            .bishop => FLAG_PROMO_BISHOP,
            .rook => FLAG_PROMO_ROOK,
            .queen => FLAG_PROMO_QUEEN,
            else => FLAG_NONE,
        } else FLAG_NONE;

        return Self{
            .data = @as(u16, from_sq) | (@as(u16, to_sq) << 6) | (@as(u16, flag) << 12),
        };
    }

    pub inline fn from(self: Self) u8 {
        return @intCast(self.data & 0x3F);
    }

    pub inline fn to(self: Self) u8 {
        return @intCast((self.data >> 6) & 0x3F);
    }

    pub inline fn promotion(self: Self) ?pieceInfo.Type {
        const flag: u4 = @intCast((self.data >> 12) & 0xF);
        return switch (flag) {
            FLAG_PROMO_KNIGHT => .knight,
            FLAG_PROMO_BISHOP => .bishop,
            FLAG_PROMO_ROOK => .rook,
            FLAG_PROMO_QUEEN => .queen,
            else => null,
        };
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

        const promo = if (promotion_str) |p| pieceInfo.Type.fromChar(p) else null;
        const from_index = Board.rankFileToIndex(from_rank, from_file);
        const to_index = Board.rankFileToIndex(to_rank, to_file);

        return Self.init(from_index, to_index, promo);
    }

    /// Compare two optional promotions for equality
    pub inline fn eqlPromotion(a: ?pieceInfo.Type, b: ?pieceInfo.Type) bool {
        if (a == null and b == null) return true;
        if (a == null or b == null) return false;
        return a.? == b.?;
    }

    pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const from_sq = self.from();
        const to_sq = self.to();

        if (from_sq == 0 and to_sq == 0 and self.promotion() == null) {
            // null move
            try writer.writeAll("0000");
            return;
        }

        const from_file = 'a' + (from_sq % 8);
        const from_rank = '1' + (from_sq / 8);
        const to_file = 'a' + (to_sq % 8);
        const to_rank = '1' + (to_sq / 8);

        try writer.print("{c}{c}{c}{c}", .{ from_file, from_rank, to_file, to_rank });

        if (self.promotion()) |p| {
            try writer.print("{c}", .{std.ascii.toLower(p.getName())});
        }
    }
};

pub const Board = struct {
    const Self = @This();
    board: BitBoard,
    zobrist_hasher: ZobristHasher = ZobristHasher.init(),

    pub const Undo = struct {
        prev_hash: u64,
        prev_castle_rights: CastleRights,
        prev_en_passant_square: ?u8,
        prev_halfmove_clock: u8,
        prev_fullmove_number: u16,
        mover_color: pieceInfo.Color,
        moved_piece: pieceInfo.Type,
        captured_piece: ?pieceInfo.Type,
        captured_square: ?u8,
        castle_rook_from: ?u8,
        castle_rook_to: ?u8,
    };

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

    pub inline fn epFileForHash(b: BitBoard) ?u8 {
        if (b.en_passant_square) |ep_sq| {
            if (b.hasAdjacentPawn(ep_sq, b.move)) {
                return ep_sq % 8;
            }
        }
        return null;
    }

    fn updateIncrementalHash(self: *Self, old_board: BitBoard) void {
        // Side to move always flips.
        self.zobrist_hasher.zobrist_hash ^= zobrist.RandomTurn;

        const colors = [_]pieceInfo.Color{ .white, .black };
        const piece_types = [_]pieceInfo.Type{ .pawn, .knight, .bishop, .rook, .queen, .king };

        inline for (colors) |color| {
            const old_color_bb = old_board.getColorBitboard(color);
            const new_color_bb = self.board.getColorBitboard(color);

            inline for (piece_types) |piece_type| {
                const old_set = old_color_bb & old_board.getKindBitboard(piece_type);
                const new_set = new_color_bb & self.board.getKindBitboard(piece_type);

                var removed = old_set & ~new_set;
                while (removed != 0) {
                    const sq: u8 = @intCast(@ctz(removed));
                    removed &= removed - 1;
                    self.zobrist_hasher.zobrist_hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(piece_type, color, sq)];
                }

                var added = new_set & ~old_set;
                while (added != 0) {
                    const sq: u8 = @intCast(@ctz(added));
                    added &= added - 1;
                    self.zobrist_hasher.zobrist_hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(piece_type, color, sq)];
                }
            }
        }

        // Toggle castling flags that changed.
        if (old_board.castle_rights.white_kingside != self.board.castle_rights.white_kingside) {
            self.zobrist_hasher.zobrist_hash ^= zobrist.RandomCastle[0];
        }
        if (old_board.castle_rights.white_queenside != self.board.castle_rights.white_queenside) {
            self.zobrist_hasher.zobrist_hash ^= zobrist.RandomCastle[1];
        }
        if (old_board.castle_rights.black_kingside != self.board.castle_rights.black_kingside) {
            self.zobrist_hasher.zobrist_hash ^= zobrist.RandomCastle[2];
        }
        if (old_board.castle_rights.black_queenside != self.board.castle_rights.black_queenside) {
            self.zobrist_hasher.zobrist_hash ^= zobrist.RandomCastle[3];
        }

        // Handle en passant hash when applicable under Polyglot rules.
        const old_ep_file = epFileForHash(old_board);
        const new_ep_file = epFileForHash(self.board);
        if (old_ep_file != new_ep_file) {
            if (old_ep_file) |f| self.zobrist_hasher.zobrist_hash ^= zobrist.RandomEnPassant[f];
            if (new_ep_file) |f| self.zobrist_hasher.zobrist_hash ^= zobrist.RandomEnPassant[f];
        }
    }

    inline fn verifyHashDebug(self: *Self) void {
        if (builtin.mode == .Debug) {
            var verifier = ZobristHasher.init();
            verifier.hash(self.board);
            std.debug.assert(verifier.zobrist_hash == self.zobrist_hasher.zobrist_hash);
        }
    }

    /// Make a pseudo-legal move and return undo data for fast unmake.
    pub inline fn makeMoveWithUndoUnchecked(self: *Self, move: Move) Undo {
        const color = self.board.move;
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;
        const from_sq = move.from();
        const to_sq = move.to();
        const moved_piece = self.board.getPieceAt(from_sq, color).?;

        const prev_hash = self.zobrist_hasher.zobrist_hash;
        const prev_castle_rights = self.board.castle_rights;
        const prev_en_passant_square = self.board.en_passant_square;
        const prev_halfmove_clock = self.board.halfmove_clock;
        const prev_fullmove_number = self.board.fullmove_number;
        const old_ep_file = epFileForHash(self.board);

        var castle_rook_from: ?u8 = null;
        var castle_rook_to: ?u8 = null;
        if (moved_piece == .king) {
            const from_file = from_sq % 8;
            const to_file = to_sq % 8;
            if (from_file == 4 and to_file == 6) {
                castle_rook_from = from_sq + 3;
                castle_rook_to = from_sq + 1;
            } else if (from_file == 4 and to_file == 2) {
                castle_rook_from = from_sq - 4;
                castle_rook_to = from_sq - 1;
            }
        }

        var captured_piece = self.board.getPieceAt(to_sq, opponent_color);
        var captured_square: ?u8 = if (captured_piece != null) to_sq else null;
        if (moved_piece == .pawn and prev_en_passant_square != null and prev_en_passant_square.? == to_sq and captured_piece == null) {
            captured_piece = .pawn;
            captured_square = if (color == .white) to_sq - 8 else to_sq + 8;
        }

        self.applyMoveUnchecked(move);

        var hash = prev_hash;
        hash ^= zobrist.RandomTurn;
        hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(moved_piece, color, from_sq)];
        const final_piece = if (move.promotion()) |promo| promo else moved_piece;
        hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(final_piece, color, to_sq)];

        if (captured_piece) |cp| {
            hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(cp, opponent_color, captured_square.?)];
        }

        if (castle_rook_from) |rook_from| {
            hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(.rook, color, rook_from)];
            hash ^= zobrist.RandomPiece[zobrist.ZobristHasher.pieceRandomIndex(.rook, color, castle_rook_to.?)];
        }

        const new_castle = self.board.castle_rights;
        if (prev_castle_rights.white_kingside != new_castle.white_kingside) hash ^= zobrist.RandomCastle[0];
        if (prev_castle_rights.white_queenside != new_castle.white_queenside) hash ^= zobrist.RandomCastle[1];
        if (prev_castle_rights.black_kingside != new_castle.black_kingside) hash ^= zobrist.RandomCastle[2];
        if (prev_castle_rights.black_queenside != new_castle.black_queenside) hash ^= zobrist.RandomCastle[3];

        const new_ep_file = epFileForHash(self.board);
        if (old_ep_file != new_ep_file) {
            if (old_ep_file) |f| hash ^= zobrist.RandomEnPassant[f];
            if (new_ep_file) |f| hash ^= zobrist.RandomEnPassant[f];
        }

        self.zobrist_hasher.zobrist_hash = hash;
        self.verifyHashDebug();

        return Undo{
            .prev_hash = prev_hash,
            .prev_castle_rights = prev_castle_rights,
            .prev_en_passant_square = prev_en_passant_square,
            .prev_halfmove_clock = prev_halfmove_clock,
            .prev_fullmove_number = prev_fullmove_number,
            .mover_color = color,
            .moved_piece = moved_piece,
            .captured_piece = captured_piece,
            .captured_square = captured_square,
            .castle_rook_from = castle_rook_from,
            .castle_rook_to = castle_rook_to,
        };
    }

    /// Restore board state from undo data produced by `makeMoveWithUndoUnchecked`.
    pub inline fn unmakeMoveUnchecked(self: *Self, move: Move, undo: Undo) void {
        const from_sq = move.from();
        const to_sq = move.to();
        const color = undo.mover_color;
        const opponent_color = if (color == .white) pieceInfo.Color.black else pieceInfo.Color.white;

        self.board.clearSquare(to_sq);
        self.board.setPieceAt(from_sq, color, undo.moved_piece);

        if (undo.castle_rook_from) |rook_from| {
            const rook_to = undo.castle_rook_to.?;
            self.board.clearSquare(rook_to);
            self.board.setPieceAt(rook_from, color, .rook);
        }

        if (undo.captured_piece) |cp| {
            self.board.setPieceAt(undo.captured_square.?, opponent_color, cp);
        }

        self.board.move = color;
        self.board.castle_rights = undo.prev_castle_rights;
        self.board.en_passant_square = undo.prev_en_passant_square;
        self.board.halfmove_clock = undo.prev_halfmove_clock;
        self.board.fullmove_number = undo.prev_fullmove_number;
        self.zobrist_hasher.zobrist_hash = undo.prev_hash;
        self.verifyHashDebug();
    }

    /// Make a move on the board using a Move structure.
    /// This updates the board state and zobrist hash.
    /// Note: This does NOT validate that the move is legal. Use with caution.
    pub fn makeMove(self: *Self, move: Move) UciError!void {
        const color = self.getTurn();
        _ = self.board.getPieceAt(move.from(), color) orelse return error.InvalidMove;
        _ = self.makeMoveWithUndoUnchecked(move);
    }

    /// Make a move and update the hash, assuming the move is pseudo-legal.
    /// This is faster than makeMove as it skips some checks.
    pub fn makeMoveUnchecked(self: *Self, move: Move) void {
        const color = self.board.move;
        // We assume the move is valid so piece must exist
        _ = self.board.getPieceAt(move.from(), color).?;
        _ = self.makeMoveWithUndoUnchecked(move);
    }

    /// Make a move from string notation (e.g., "e2e4").
    /// This validates that the move is legal before applying it.
    pub fn makeStrMove(self: *Self, move_str: []const u8) UciError!void {
        const move = try Move.fromString(move_str);

        // Validate that the move is legal
        var legal_moves = MoveList.init();
        try self.generateLegalMoves(&legal_moves);

        for (legal_moves.slice()) |legal_move| {
            if (legal_move.from() == move.from() and
                legal_move.to() == move.to() and
                Move.eqlPromotion(legal_move.promotion(), move.promotion()))
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

            // Make move
            self.applyMoveUncheckedForLegality(move);

            // Check if our king is in check after the move (illegal)
            const legal = !self.isInCheck(color);

            // Restore state
            self.board = old_board;

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

            // Make move
            self.applyMoveUnchecked(move);

            // Check if our king is in check after the move (illegal)
            if (self.isInCheck(color)) {
                // Illegal move, restore and skip
                self.board = old_board;
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

                // Make move
                self.applyMoveUnchecked(move);

                // Check if our king is in check after the move (illegal)
                if (self.isInCheck(moving_color)) {
                    // Illegal move, restore and skip
                    self.board = old_board;
                    continue;
                }

                // Legal move - count it and check properties
                stats.nodes += 1;

                // Check piece type before checking move properties
                const piece_type = old_board.getPieceAt(move.from(), moving_color);

                // Check if it's en passant
                const is_en_passant = piece_type == .pawn and old_board.en_passant_square == move.to();
                if (is_en_passant) {
                    stats.en_passant += 1;
                }

                // Check if it's a capture (including en passant)
                const piece_at_dest = old_board.getPieceAt(move.to(), opponent_color);
                if (piece_at_dest != null or is_en_passant) {
                    stats.captures += 1;
                }

                // Check if it's a castle
                if (piece_type == .king) {
                    const from_file = move.from() % 8;
                    const to_file = move.to() % 8;
                    if (from_file == 4 and (to_file == 6 or to_file == 2)) {
                        stats.castles += 1;
                    }
                }

                // Check if it's a promotion
                if (move.promotion() != null) {
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
            }
            return;
        }

        // Recursive case
        for (pseudo_legal.slice()) |move| {
            // Save state
            const old_board = self.board;

            // Make move
            self.applyMoveUnchecked(move);

            // Check if our king is in check after the move (illegal)
            if (self.isInCheck(moving_color)) {
                // Illegal move, restore and skip
                self.board = old_board;
                continue;
            }

            // Recurse
            var child_stats = PerftStats{};
            try self.perftWithStats(depth - 1, &child_stats);
            stats.add(child_stats);

            // Restore state
            self.board = old_board;
        }
    }

    // Delegated board state helpers.
    fn countCheckingPieces(self: *Self, king_color: pieceInfo.Color) u32 {
        return legality.countCheckingPieces(self, king_color);
    }

    fn isDirectCheck(self: *Self, move: Move, moving_color: pieceInfo.Color, opponent_color: pieceInfo.Color) bool {
        return legality.isDirectCheck(self, move, moving_color, opponent_color);
    }

    /// Perft divide - Shows the number of nodes for each root move
    pub fn perftDivide(self: *Self, depth: u32, writer: anytype) UciError!u64 {
        var moves = MoveList.init();
        try self.generateLegalMoves(&moves);

        var total_nodes: u64 = 0;

        for (moves.slice()) |move| {
            // Save state
            const old_board = self.board;

            // Make move
            self.applyMoveUnchecked(move);

            // Count nodes
            const nodes = if (depth <= 1) 1 else try self.perft(depth - 1);
            total_nodes += nodes;

            // Print result
            writer.print("{f}: {d}\n", .{ move, nodes }) catch return UciError.IOError;

            // Restore state
            self.board = old_board;
        }

        writer.print("\nTotal nodes: {d}\n", .{total_nodes}) catch return UciError.IOError;
        return total_nodes;
    }

    pub inline fn isInCheck(self: *Self, color: pieceInfo.Color) bool {
        return legality.isInCheck(self, color);
    }

    fn isSquareAttackedBy(self: *Self, square: u6, attacker_color: pieceInfo.Color) bool {
        return legality.isSquareAttackedBy(self, square, attacker_color);
    }

    pub fn applyMoveUnchecked(self: *Self, move: Move) void {
        make_unmake.applyMoveUnchecked(self, move);
    }

    pub inline fn applyMoveUncheckedForLegality(self: *Self, move: Move) void {
        make_unmake.applyMoveUncheckedForLegality(self, move);
    }

    fn generatePseudoLegalMoves(self: *Self, moves: *MoveList) !void {
        try movegen.generatePseudoLegalMoves(self, moves);
    }

    pub fn generateCaptures(self: *Self, moves: *MoveList) void {
        movegen.generateCaptures(self, moves);
    }

    pub fn generateLegalCaptures(self: *Self, moves: *MoveList) void {
        movegen.generateLegalCaptures(self, moves);
    }

    pub fn generateQuietMoves(self: *Self, moves: *MoveList) void {
        movegen.generateQuietMoves(self, moves);
    }

    pub fn generateLegalQuietMoves(self: *Self, moves: *MoveList) void {
        movegen.generateLegalQuietMoves(self, moves);
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

    pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
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
        var buffer = std.ArrayList(u8).empty;
        defer buffer.deinit(allocator);

        for (0..8) |rank| {
            const actual_rank = 7 - rank;
            var empty_count: u8 = 0;

            for (0..8) |file| {
                const index = actual_rank * 8 + file;
                const white_piece = self.board.getPieceAt(@intCast(index), .white);
                const black_piece = self.board.getPieceAt(@intCast(index), .black);

                if (white_piece) |p| {
                    if (empty_count > 0) {
                        var count_buf: [3]u8 = undefined;
                        const count_str = try std.fmt.bufPrint(&count_buf, "{d}", .{empty_count});
                        try buffer.appendSlice(allocator, count_str);
                        empty_count = 0;
                    }
                    try buffer.append(allocator, p.getName());
                } else if (black_piece) |p| {
                    if (empty_count > 0) {
                        var count_buf: [3]u8 = undefined;
                        const count_str = try std.fmt.bufPrint(&count_buf, "{d}", .{empty_count});
                        try buffer.appendSlice(allocator, count_str);
                        empty_count = 0;
                    }
                    try buffer.append(allocator, std.ascii.toLower(p.getName()));
                } else {
                    empty_count += 1;
                }
            }

            if (empty_count > 0) {
                var count_buf: [3]u8 = undefined;
                const count_str = try std.fmt.bufPrint(&count_buf, "{d}", .{empty_count});
                try buffer.appendSlice(allocator, count_str);
            }

            if (rank != 7) {
                try buffer.append(allocator, '/');
            }
        }

        // Turn
        const turn: u8 = if (self.board.move == .white) 'w' else 'b';
        try buffer.append(allocator, ' ');
        try buffer.append(allocator, turn);

        // Castling rights
        try buffer.append(allocator, ' ');
        var any_castle = false;
        if (self.board.castle_rights.white_kingside) {
            try buffer.append(allocator, 'K');
            any_castle = true;
        }
        if (self.board.castle_rights.white_queenside) {
            try buffer.append(allocator, 'Q');
            any_castle = true;
        }
        if (self.board.castle_rights.black_kingside) {
            try buffer.append(allocator, 'k');
            any_castle = true;
        }
        if (self.board.castle_rights.black_queenside) {
            try buffer.append(allocator, 'q');
            any_castle = true;
        }
        if (!any_castle) try buffer.append(allocator, '-');

        // En passant
        try buffer.append(allocator, ' ');
        if (self.board.en_passant_square) |sq| {
            const file = sq % 8 + 'a';
            const rank = sq / 8 + '1';
            try buffer.append(allocator, file);
            try buffer.append(allocator, rank);
        } else {
            try buffer.append(allocator, '-');
        }

        // Halfmove and fullmove
        var moves_buf: [32]u8 = undefined;
        const moves_str = try std.fmt.bufPrint(&moves_buf, " {d} {d}", .{ self.board.halfmove_clock, self.board.fullmove_number });
        try buffer.appendSlice(allocator, moves_str);
        return buffer.toOwnedSlice(allocator);
    }
};

// Static attack functions - can be called without a Board instance
pub inline fn getKnightAttacks(square: u6) u64 {
    return attacks.getKnightAttacks(square);
}

pub inline fn getKingAttacks(square: u6) u64 {
    return attacks.getKingAttacks(square);
}

pub inline fn getPawnAttacks(square: u6, pawn_color: pieceInfo.Color) u64 {
    return attacks.getPawnAttacks(square, pawn_color);
}

pub inline fn getRookAttacks(square: u6, occupied: u64) u64 {
    return attacks.getRookAttacks(square, occupied);
}

pub inline fn getBishopAttacks(square: u6, occupied: u64) u64 {
    return attacks.getBishopAttacks(square, occupied);
}

pub inline fn getQueenAttacks(square: u6, occupied: u64) u64 {
    return attacks.getQueenAttacks(square, occupied);
}

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
        return Self{
            .color_sets = [_]u64{0} ** 2,
            .kind_sets = [_]u64{0} ** 6,
        };
    }

    pub fn getColorBitboard(self: Self, color: pieceInfo.Color) u64 {
        return self.color_sets[@intFromEnum(color)];
    }

    pub fn getKindBitboard(self: Self, kind: pieceInfo.Type) u64 {
        return self.kind_sets[@intFromEnum(kind)];
    }

    pub fn clearSquare(self: *Self, index: u8) void {
        const mask = @as(u64, 1) << @intCast(index);

        for (0..2) |i| {
            self.color_sets[i] &= ~mask;
        }
        for (0..6) |i| {
            self.kind_sets[i] &= ~mask;
        }
    }

    pub fn setPieceAt(self: *Self, index: u8, color: pieceInfo.Color, kind: pieceInfo.Type) void {
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

    pub inline fn occupied(self: Self) u64 {
        return self.color_sets[0] | self.color_sets[1];
    }

    inline fn empty(self: Self) u64 {
        return ~self.occupied();
    }
};

test "makeMove updates halfmove and fullmove clocks" {
    var b = Board.startpos();

    try b.makeStrMove("g1f3");
    try std.testing.expectEqual(@as(u8, 1), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 1), b.board.fullmove_number);

    try b.makeStrMove("g8f6");
    try std.testing.expectEqual(@as(u8, 2), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 2), b.board.fullmove_number);

    try b.makeStrMove("e2e4");
    try std.testing.expectEqual(@as(u8, 0), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 2), b.board.fullmove_number);

    try b.makeStrMove("d7d5");
    try std.testing.expectEqual(@as(u8, 0), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 3), b.board.fullmove_number);

    try b.makeStrMove("e4d5");
    try std.testing.expectEqual(@as(u8, 0), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 3), b.board.fullmove_number);

    try b.makeStrMove("f6d5");
    try std.testing.expectEqual(@as(u8, 0), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 4), b.board.fullmove_number);
}

test "repetition shuffle sequence increments clocks from fen" {
    var b = try Board.fromFen("r3k1n1/1R1b4/2p2pp1/8/1BP1PP2/P6p/2PK3P/5B2 w - - 2 35");

    try b.makeStrMove("b4c5");
    try b.makeStrMove("a8a5");
    try b.makeStrMove("c5b4");
    try b.makeStrMove("a5a8");
    try b.makeStrMove("b4c5");
    try b.makeStrMove("a8a5");

    try std.testing.expectEqual(@as(u8, 8), b.board.halfmove_clock);
    try std.testing.expectEqual(@as(u16, 38), b.board.fullmove_number);
}
