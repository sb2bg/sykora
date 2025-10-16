const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const Move = board.Move;
const piece = @import("piece.zig");
const UciError = @import("uci_error.zig").UciError;

pub const SearchOptions = struct {
    infinite: bool = false,
    move_time: ?u64 = null,
    wtime: ?u64 = null,
    btime: ?u64 = null,
    depth: ?u64 = null,
};

pub const SearchResult = struct {
    best_move: Move,
    score: i32,
    nodes: usize,
    time_ms: i64,
};

pub const SearchEngine = struct {
    const Self = @This();
    
    board: *Board,
    allocator: std.mem.Allocator,
    stop_search: *std.atomic.Value(bool),
    info_callback: ?*const fn ([]const u8) void,
    
    pub fn init(
        board_ptr: *Board,
        allocator: std.mem.Allocator,
        stop_search: *std.atomic.Value(bool),
    ) Self {
        return Self{
            .board = board_ptr,
            .allocator = allocator,
            .stop_search = stop_search,
            .info_callback = null,
        };
    }
    
    /// Run a search and return the best move
    pub fn search(self: *Self, options: SearchOptions) !SearchResult {
        const start_time = std.time.milliTimestamp();
        
        // Calculate time limit
        const time_limit = self.calculateTimeLimit(options);
        
        // Generate legal moves
        const legal_moves = try self.board.generateLegalMoves(self.allocator);
        defer self.allocator.free(legal_moves);
        
        if (legal_moves.len == 0) {
            return SearchResult{
                .best_move = Move.init(0, 0, null),
                .score = 0,
                .nodes = 0,
                .time_ms = 0,
            };
        }
        
        var best_eval: i32 = -9999;
        var best_move: Move = legal_moves[0];
        var nodes: usize = 0;
        
        // Simple search: evaluate each move
        for (legal_moves) |mv| {
            // Check if search was stopped
            if (self.stop_search.load(.seq_cst)) {
                break;
            }
            
            const score = self.evaluateMove(mv);
            nodes += 1;
            
            if (score > best_eval) {
                best_eval = score;
                best_move = mv;
            }
            
            // Check time limit
            if (time_limit) |limit| {
                const elapsed = std.time.milliTimestamp() - start_time;
                if (elapsed > @as(i64, @intCast(limit))) {
                    break;
                }
            }
            
            // Simulate some thinking time (can be removed later)
            std.time.sleep(5 * std.time.ns_per_ms);
        }
        
        const elapsed = std.time.milliTimestamp() - start_time;
        
        return SearchResult{
            .best_move = best_move,
            .score = best_eval,
            .nodes = nodes,
            .time_ms = elapsed,
        };
    }
    
    fn calculateTimeLimit(self: *Self, options: SearchOptions) ?u64 {
        _ = self;
        
        if (options.infinite) {
            return null;
        } else if (options.move_time) |move_time| {
            return move_time;
        } else if (options.wtime) |wtime| {
            return wtime / 100;
        } else if (options.btime) |btime| {
            return btime / 100;
        }
        
        return null;
    }
    
    /// Evaluate a single move (static evaluation)
    fn evaluateMove(self: *Self, move: Move) i32 {
        var score: i32 = 0;
        const color = self.board.board.move;
        const opponent = if (color == .white) piece.Color.black else piece.Color.white;
        
        // Check if it's a capture
        const captured_piece = self.board.board.getPieceAt(move.to, opponent);
        if (captured_piece) |piece_type| {
            // Material values (in centipawns)
            score += switch (piece_type) {
                .pawn => 100,
                .knight => 300,
                .bishop => 310,
                .rook => 500,
                .queen => 900,
                .king => 0, // Can't actually capture the king
            };
        }
        
        // Reward central squares (d4, e4, d5, e5)
        const to_file = move.to % 8;
        const to_rank = move.to / 8;
        const is_center = (to_file == 3 or to_file == 4) and (to_rank == 3 or to_rank == 4);
        if (is_center) {
            score += 30;
        }
        
        // Reward moving pieces off the back rank (development)
        const from_rank = move.from / 8;
        const back_rank: u8 = if (color == .white) 0 else 7;
        if (from_rank == back_rank and to_rank != back_rank) {
            score += 20;
        }
        
        // Small bonus for pawn advances
        const moving_piece = self.board.board.getPieceAt(move.from, color);
        if (moving_piece) |p| {
            if (p == .pawn) {
                if (color == .white and to_rank > from_rank) {
                    score += @as(i32, @intCast(to_rank - from_rank)) * 5;
                } else if (color == .black and to_rank < from_rank) {
                    score += @as(i32, @intCast(from_rank - to_rank)) * 5;
                }
            }
        }
        
        // Bonus for promotions
        if (move.promotion) |promo| {
            score += switch (promo) {
                .queen => 800,
                .rook => 400,
                .bishop => 200,
                .knight => 200,
                else => 0,
            };
        }
        
        return score;
    }
};
