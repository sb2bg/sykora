const std = @import("std");
const board = @import("bitboard.zig");
const uci_command = @import("uci_command.zig");
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const search_module = @import("search.zig");
const SearchEngine = search_module.SearchEngine;
const SearchOptions = search_module.SearchOptions;
const nnue = @import("nnue.zig");
const Uci = @import("interface.zig").Uci;

pub const MAX_HELPERS = 63;

pub const HelperResult = struct {
    best_move: board.Move,
    score: i32,
    depth: u32,
    nodes: usize,
};

const EMPTY_HELPER_RESULT = HelperResult{
    .best_move = board.Move.init(0, 0, null),
    .score = 0,
    .depth = 0,
    .nodes = 0,
};

pub fn elapsedMs(start: std.time.Instant) i64 {
    const now = std.time.Instant.now() catch return 0;
    const ns = now.since(start);
    return @intCast(@divFloor(ns, std.time.ns_per_ms));
}

pub fn terminateSearch(self: *Uci) !void {
    self.stop_search.store(true, .seq_cst);

    if (self.search_thread) |thread| {
        const start = std.time.Instant.now() catch return UciError.IOError;
        thread.join();
        self.search_thread = null;
        const duration = elapsedMs(start);
        try self.writeInfoString("search thread joined in {d}ms", .{duration});
    }
}

pub fn search(self: *Uci, go_opts: uci_command.GoOptions) UciError!void {
    try self.writeInfoString("search thread started", .{});

    const net_ptr: ?*const nnue.Network = if (self.nnue_network) |*network| network else null;
    const use_nnue_for_search = self.use_nnue and net_ptr != null;

    const prior_count = self.position_hash_count;

    self.tt.nextAge();

    const num_helpers = self.num_threads - 1;
    for (0..num_helpers) |i| {
        self.helper_threads[i] = null;
        self.helper_results[i] = EMPTY_HELPER_RESULT;
    }

    for (0..num_helpers) |i| {
        self.helper_threads[i] = std.Thread.spawn(.{ .stack_size = 8 * 1024 * 1024 }, helperSearch, .{
            self,
            i,
            go_opts,
            net_ptr,
            use_nnue_for_search,
            prior_count,
        }) catch {
            self.helper_threads[i] = null;
            self.helper_results[i] = EMPTY_HELPER_RESULT;
            continue;
        };
    }

    var search_board = self.board;
    var search_engine = SearchEngine.init(
        &search_board,
        self.allocator,
        &self.stop_search,
        &self.tt,
        use_nnue_for_search,
        net_ptr,
        self.nnue_blend,
        self.nnue_scale,
        self.uci_chess960,
    );

    search_engine.uci_output = self.stdout;
    if (prior_count > 0) {
        search_engine.setGameHistory(self.position_hash_history[0..prior_count]);
    } else {
        search_engine.setGameHistory(&.{});
    }

    const search_opts = SearchOptions{
        .infinite = go_opts.infinite orelse false,
        .move_time = go_opts.move_time,
        .wtime = go_opts.wtime,
        .btime = go_opts.btime,
        .winc = go_opts.winc,
        .binc = go_opts.binc,
        .depth = go_opts.depth,
    };

    const result = search_engine.search(search_opts) catch {
        self.stop_search.store(true, .seq_cst);
        joinHelpers(self, num_helpers);
        return UciError.IOError;
    };

    self.stop_search.store(true, .seq_cst);
    joinHelpers(self, num_helpers);

    if (num_helpers > 0) {
        self.best_move = voteBestMove(self, result, num_helpers);
    } else {
        self.best_move = result.best_move;
    }

    var total_nodes = result.nodes;
    for (0..num_helpers) |i| {
        total_nodes += self.helper_results[i].nodes;
    }

    try self.writeInfoString("search thread stopped, total nodes {d}", .{total_nodes});
    var move_buf: [5]u8 = undefined;
    const move_str = self.board.formatMoveUci(self.best_move, self.uci_chess960, &move_buf);
    try self.writeStdout("bestmove {s}", .{move_str});
}

fn helperSearch(
    self: *Uci,
    idx: usize,
    go_opts: uci_command.GoOptions,
    net_ptr: ?*const nnue.Network,
    use_nnue_for_search: bool,
    prior_count: usize,
) void {
    var helper_board = self.board;
    var search_engine = SearchEngine.init(
        &helper_board,
        self.allocator,
        &self.stop_search,
        &self.tt,
        use_nnue_for_search,
        net_ptr,
        self.nnue_blend,
        self.nnue_scale,
        self.uci_chess960,
    );

    search_engine.uci_output = null;
    if (prior_count > 0) {
        search_engine.setGameHistory(self.position_hash_history[0..prior_count]);
    } else {
        search_engine.setGameHistory(&.{});
    }

    const max_depth: u32 = if (go_opts.depth) |d| @intCast(d) else 64;
    const start_depth: u32 = @min(maxDepthStagger(idx), max_depth);

    const search_opts = SearchOptions{
        .infinite = true,
        .depth = go_opts.depth,
        .start_depth = start_depth,
    };

    const result = search_engine.search(search_opts) catch {
        self.helper_results[idx] = EMPTY_HELPER_RESULT;
        return;
    };

    if (result.depth == 0) {
        self.helper_results[idx] = EMPTY_HELPER_RESULT;
        return;
    }

    self.helper_results[idx] = .{
        .best_move = result.best_move,
        .score = result.score,
        .depth = result.depth,
        .nodes = result.nodes,
    };
}

fn maxDepthStagger(idx: usize) u32 {
    const schedule = [_]u32{ 1, 2, 3, 4, 5, 6, 3, 5 };
    return schedule[idx % schedule.len];
}

pub fn joinHelpers(self: *Uci, num_helpers: usize) void {
    for (0..num_helpers) |i| {
        if (self.helper_threads[i]) |thread| {
            thread.join();
            self.helper_threads[i] = null;
        }
    }
}

pub fn voteBestMove(self: *Uci, main_result: search_module.SearchResult, num_helpers: usize) board.Move {
    const max_voters = MAX_HELPERS + 1;
    var moves: [max_voters]board.Move = undefined;
    var scores: [max_voters]i32 = undefined;
    var depths: [max_voters]u32 = undefined;
    var is_main_vote: [max_voters]bool = undefined;
    var count: usize = 0;

    if (main_result.best_move.from() != 0 or main_result.best_move.to() != 0) {
        moves[count] = main_result.best_move;
        scores[count] = main_result.score;
        depths[count] = main_result.depth;
        is_main_vote[count] = true;
        count += 1;
    }

    for (0..num_helpers) |i| {
        const hr = self.helper_results[i];
        if (hr.best_move.from() != 0 or hr.best_move.to() != 0) {
            moves[count] = hr.best_move;
            scores[count] = hr.score;
            depths[count] = hr.depth;
            is_main_vote[count] = false;
            count += 1;
        }
    }

    if (count == 0) return main_result.best_move;
    if (count == 1) return moves[0];

    var worst_score: i32 = scores[0];
    for (0..count) |i| {
        if (scores[i] < worst_score) worst_score = scores[i];
    }

    var vote_moves: [max_voters]board.Move = undefined;
    var vote_weights: [max_voters]i32 = undefined;
    var vote_best_depth: [max_voters]u32 = undefined;
    var vote_best_score: [max_voters]i32 = undefined;
    var vote_support: [max_voters]u32 = undefined;
    var vote_has_main: [max_voters]bool = undefined;
    var num_unique: usize = 0;
    var main_unique_idx: ?usize = null;

    for (0..count) |i| {
        var weight = @as(i32, @intCast(depths[i])) + @divTrunc(scores[i] - worst_score, 10);
        if (is_main_vote[i]) {
            weight += 2;
        }

        var found: bool = false;
        for (0..num_unique) |j| {
            if (vote_moves[j].from() == moves[i].from() and
                vote_moves[j].to() == moves[i].to() and
                board.Move.eqlPromotion(vote_moves[j].promotion(), moves[i].promotion()))
            {
                vote_weights[j] += weight;
                vote_support[j] += 1;
                vote_has_main[j] = vote_has_main[j] or is_main_vote[i];
                if (depths[i] > vote_best_depth[j] or
                    (depths[i] == vote_best_depth[j] and scores[i] > vote_best_score[j]))
                {
                    vote_best_depth[j] = depths[i];
                    vote_best_score[j] = scores[i];
                }
                if (is_main_vote[i]) {
                    main_unique_idx = j;
                }
                found = true;
                break;
            }
        }

        if (!found) {
            vote_moves[num_unique] = moves[i];
            vote_weights[num_unique] = weight;
            vote_best_depth[num_unique] = depths[i];
            vote_best_score[num_unique] = scores[i];
            vote_support[num_unique] = 1;
            vote_has_main[num_unique] = is_main_vote[i];
            if (is_main_vote[i]) {
                main_unique_idx = num_unique;
            }
            num_unique += 1;
        }
    }

    var best_idx: usize = 0;
    for (1..num_unique) |i| {
        if (vote_weights[i] > vote_weights[best_idx] or
            (vote_weights[i] == vote_weights[best_idx] and vote_best_depth[i] > vote_best_depth[best_idx]) or
            (vote_weights[i] == vote_weights[best_idx] and vote_best_depth[i] == vote_best_depth[best_idx] and vote_best_score[i] > vote_best_score[best_idx]))
        {
            best_idx = i;
        }
    }

    if (main_unique_idx) |main_idx| {
        if (best_idx != main_idx) {
            const best_has_real_consensus = vote_support[best_idx] >= 2;
            const best_is_at_least_as_deep = vote_best_depth[best_idx] >= vote_best_depth[main_idx];
            const best_weight_clearly_wins = vote_weights[best_idx] > vote_weights[main_idx];
            if (!best_has_real_consensus or !best_is_at_least_as_deep or !best_weight_clearly_wins) {
                return vote_moves[main_idx];
            }
        }
    }

    return vote_moves[best_idx];
}
