const std = @import("std");
const UciParser = @import("uci_parser.zig").UciParser;
const uci_command = @import("uci_command.zig");
const ToEngineCommand = uci_command.ToEngineCommand;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const board = @import("bitboard.zig");
const Board = board.Board;
const options = @import("options.zig");
const Options = options.Options;
const Option = options.Option;
const ZobristHasher = @import("zobrist.zig").ZobristHasher;
const search_module = @import("search.zig");
const SearchEngine = search_module.SearchEngine;
const SearchOptions = search_module.SearchOptions;
const TranspositionTable = search_module.TranspositionTable;
const nnue = @import("nnue.zig");

const name = "Sykora";
const author = "Sullivan Bognar";
const version = "0.1.0";

pub const Uci = struct {
    const Self = @This();
    stdin: std.fs.File,
    stdout: std.fs.File,
    uci_parser: UciParser,
    debug: bool,
    options: Options,
    stop_search: std.atomic.Value(bool),
    board: Board,
    allocator: std.mem.Allocator,
    search_thread: ?std.Thread,
    best_move: board.Move,
    log_file: ?std.fs.File,
    use_nnue: bool,
    eval_file_path: ?[]u8,
    nnue_network: ?nnue.Network,
    nnue_blend: i32,
    nnue_scale: i32,
    nnue_screlu: bool,
    position_hash_history: [512]u64,
    position_hash_count: usize,
    tt: TranspositionTable,
    num_threads: usize,
    hash_size_mb: usize,
    helper_threads: [MAX_HELPERS]?std.Thread,
    helper_results: [MAX_HELPERS]HelperResult,

    // Helper thread state for Lazy SMP
    const MAX_HELPERS = 63;
    const HelperResult = struct {
        best_move: board.Move,
        score: i32,
        depth: u32,
        nodes: usize,
    };

    pub fn init(stdin: std.fs.File, stdout: std.fs.File, allocator: std.mem.Allocator) !*Self {
        const uci_ptr = try allocator.create(Self);
        const stop_search = std.atomic.Value(bool).init(false);
        const default_hash_mb: usize = 64;
        const tt = try TranspositionTable.init(allocator, default_hash_mb);

        uci_ptr.* = Uci{
            .stdin = stdin,
            .stdout = stdout,
            .uci_parser = UciParser.init(allocator),
            .debug = false,
            .options = Options.init(allocator),
            .board = Board.startpos(),
            .allocator = allocator,
            .stop_search = stop_search,
            .search_thread = null,
            .best_move = board.Move.init(0, 0, null), // null move
            .log_file = null,
            .use_nnue = false,
            .eval_file_path = null,
            .nnue_network = null,
            .nnue_blend = 2,
            .nnue_scale = 100,
            .nnue_screlu = false,
            .position_hash_history = undefined,
            .position_hash_count = 0,
            .tt = tt,
            .num_threads = 1,
            .hash_size_mb = default_hash_mb,
            .helper_threads = [_]?std.Thread{null} ** MAX_HELPERS,
            .helper_results = [_]HelperResult{.{ .best_move = board.Move.init(0, 0, null), .score = 0, .depth = 0, .nodes = 0 }} ** MAX_HELPERS,
        };

        uci_ptr.resetPositionHistory();

        // Add logging option
        try uci_ptr.options.items.append(allocator, Option{
            .name = "Debug Log File",
            .type = .string,
            .default_value = "<empty>",
            .on_changed = handleLogFileChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "UseNNUE",
            .type = .check,
            .default_value = "false",
            .on_changed = handleUseNnueChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "EvalFile",
            .type = .string,
            .default_value = "<empty>",
            .on_changed = handleEvalFileChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "NnueBlend",
            .type = .spin,
            .default_value = "2",
            .min_value = 0,
            .max_value = 100,
            .on_changed = handleNnueBlendChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "NnueScale",
            .type = .spin,
            .default_value = "100",
            .min_value = 10,
            .max_value = 400,
            .on_changed = handleNnueScaleChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "NnueSCReLU",
            .type = .check,
            .default_value = "false",
            .on_changed = handleNnueScReluChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "Threads",
            .type = .spin,
            .default_value = "1",
            .min_value = 1,
            .max_value = 64,
            .on_changed = handleThreadsChange,
            .context = uci_ptr,
        });
        try uci_ptr.options.items.append(allocator, Option{
            .name = "Hash",
            .type = .spin,
            .default_value = "64",
            .min_value = 1,
            .max_value = 4096,
            .on_changed = handleHashChange,
            .context = uci_ptr,
        });

        try uci_ptr.writeStdout("{s} version {s} by {s}", .{ name, version, author });
        return uci_ptr;
    }

    pub fn deinit(self: *Self) void {
        // Ensure search thread is terminated before cleanup
        self.stop_search.store(true, .seq_cst);
        if (self.search_thread) |thread| {
            thread.join();
            self.search_thread = null;
        }

        if (self.log_file) |file| {
            file.close();
        }
        if (self.eval_file_path) |path| {
            self.allocator.free(path);
        }
        if (self.nnue_network) |*network| {
            network.deinit();
        }
        self.tt.deinit();
        self.options.deinit();
        self.allocator.destroy(self);
    }

    fn elapsedMs(start: std.time.Instant) i64 {
        const now = std.time.Instant.now() catch return 0;
        const ns = now.since(start);
        return @intCast(@divFloor(ns, std.time.ns_per_ms));
    }

    pub fn run(self: *Self) UciError!void {
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(self.allocator);

        while (true) {
            defer buf.clearRetainingCapacity();

            // Read a single command line from stdin.
            while (true) {
                var byte: [1]u8 = undefined;
                const n = self.stdin.read(&byte) catch return UciError.IOError;
                if (n == 0) {
                    // EOF: if partial command exists, process it; otherwise exit cleanly.
                    if (buf.items.len == 0) return;
                    break;
                }

                if (byte[0] == '\n') break;
                if (byte[0] == '\r') continue;
                buf.append(self.allocator, byte[0]) catch return UciError.OutOfMemory;
            }

            // Log input
            if (self.log_file) |file| {
                file.writeAll("> ") catch return UciError.IOError;
                file.writeAll(buf.items) catch return UciError.IOError;
                file.writeAll("\n") catch return UciError.IOError;
            }

            const command = self.uci_parser.parseCommand(buf.items) catch |err| {
                try self.writeInfoString("{s}", .{uciErr.getErrorDescriptor(err)});
                continue;
            };

            self.handleCommand(command) catch |err| {
                if (err == UciError.Quit) {
                    try self.terminateSearch();
                    return;
                }

                try self.writeInfoString("{s}", .{uciErr.getErrorDescriptor(err)});
            };
        }
    }

    fn terminateSearch(self: *Self) !void {
        self.stop_search.store(true, .seq_cst);

        if (self.search_thread) |thread| {
            // time to see how long it takes to join the thread
            const start = std.time.Instant.now() catch return UciError.IOError;
            thread.join(); // block until it finishes
            self.search_thread = null;
            const duration = elapsedMs(start);
            try self.writeInfoString("search thread joined in {d}ms", .{duration});
        }
    }

    fn handleCommand(self: *Self, command: ToEngineCommand) UciError!void {
        switch (command) {
            .uci => {
                try self.writeStdout("id name {s} {s}", .{ name, version });
                try self.writeStdout("id author {s}", .{author});
                var stdout_buf: [1024]u8 = undefined;
                var stdout_writer = self.stdout.writer(&stdout_buf);
                try self.options.printOptions(&stdout_writer.interface);
                stdout_writer.interface.flush() catch return UciError.IOError;
                try self.writeStdout("uciok", .{});
            },
            .debug => |value| {
                self.debug = value;
            },
            .isready => {
                try self.writeStdout("readyok", .{});
            },
            .ucinewgame => {
                try self.terminateSearch();
                self.board = Board.startpos();
                self.resetPositionHistory();
                self.tt.clear();
            },
            .position => |pos_opts| {
                switch (pos_opts.value) {
                    .startpos => {
                        self.board = Board.startpos();
                    },
                    .fen => {
                        defer self.allocator.free(pos_opts.value.fen);
                        self.board = try Board.fromFen(pos_opts.value.fen);
                    },
                }

                self.resetPositionHistory();

                if (pos_opts.moves) |moves| {
                    defer self.allocator.free(moves);

                    for (moves) |move| {
                        try self.board.makeStrMove(move);
                        self.pushCurrentHashToPositionHistory();
                    }
                }
            },
            .display => {
                try self.writeStdout("{f}", .{self.board});
                const fen = try self.board.getFenString(self.allocator);
                defer self.allocator.free(fen);
                try self.writeStdout("fen {s}", .{fen});
                try self.writeStdout("key {x}", .{self.board.zobrist_hasher.zobrist_hash});
            },
            .go => |go_opts| {
                try self.terminateSearch();
                try self.writeInfoString("{any}", .{go_opts});
                try self.writeInfoString("starting search thread", .{});
                self.stop_search.store(false, .seq_cst);
                self.best_move = board.Move.init(0, 0, null);

                // Use larger stack size (8MB) for search thread due to large SearchEngine struct
                self.search_thread = std.Thread.spawn(.{ .stack_size = 8 * 1024 * 1024 }, Uci.search, .{ self, go_opts }) catch return UciError.ThreadCreationFailed;
            },
            .stop => {
                try self.terminateSearch();
            },
            .ponderhit => {
                return error.Unimplemented;
            },
            .setoption => |opts| {
                defer self.allocator.free(opts.name);
                const opt_name = opts.name;

                if (opts.value) |value| {
                    const success = try self.options.setOption(opt_name, value);

                    if (success) {
                        try self.writeInfoString("option {s} set to {s}", .{ opt_name, value });
                    } else {
                        try self.writeInfoString("option {s} does not exist", .{opt_name});
                    }
                } else {
                    const option = self.options.getOption(opt_name) orelse {
                        try self.writeInfoString("option {s} does not exist", .{opt_name});
                        return;
                    };

                    if (option.type == .button) {
                        try self.writeInfoString("button {s} pressed", .{opt_name});
                        // TODO: handle button press
                    } else {
                        try self.writeInfoString("option {s} is not a button", .{opt_name});
                    }
                }
            },
            .perft => |perft_opts| {
                switch (perft_opts.mode) {
                    .stats => {
                        // Perft with detailed statistics
                        try self.writeStdout("", .{});
                        try self.writeStdout("Running perft to depth {d}...", .{perft_opts.depth});
                        try self.writeStdout("", .{});

                        // Print header
                        try self.writeStdout("Depth | Nodes      | Captures   | E.p. | Castles | Promotions | Checks | Discovery | Double | Checkmates | Time(ms)", .{});
                        try self.writeStdout("------|------------|------------|------|---------|------------|--------|-----------|--------|------------|----------", .{});

                        const start_time = std.time.Instant.now() catch return UciError.IOError;

                        // Run perft for each depth up to target depth
                        var d: u32 = 1;
                        while (d <= perft_opts.depth) : (d += 1) {
                            const depth_start = std.time.Instant.now() catch return UciError.IOError;
                            var stats = board.Board.PerftStats{};
                            try self.board.perftWithStats(@intCast(d), &stats);
                            const depth_time = elapsedMs(depth_start);

                            try self.writeStdout("{d: >5} | {d: >10} | {d: >10} | {d: >4} | {d: >7} | {d: >10} | {d: >6} | {d: >9} | {d: >6} | {d: >10} | {d: >8}", .{
                                d,
                                stats.nodes,
                                stats.captures,
                                stats.en_passant,
                                stats.castles,
                                stats.promotions,
                                stats.checks,
                                stats.discovery_checks,
                                stats.double_checks,
                                stats.checkmates,
                                depth_time,
                            });
                        }

                        const total_time = elapsedMs(start_time);

                        try self.writeStdout("", .{});
                        try self.writeStdout("Total time: {d}ms", .{total_time});
                    },
                    .divide => {
                        // Perft divide - show per-move breakdown
                        try self.writeStdout("", .{});
                        try self.writeStdout("Perft divide at depth {d}:", .{perft_opts.depth});
                        try self.writeStdout("", .{});

                        const start_time = std.time.Instant.now() catch return UciError.IOError;
                        var stdout_buf: [1024]u8 = undefined;
                        var stdout_writer = self.stdout.writer(&stdout_buf);
                        const total_nodes = try self.board.perftDivide(@intCast(perft_opts.depth), &stdout_writer.interface);
                        stdout_writer.interface.flush() catch return UciError.IOError;
                        const total_time = elapsedMs(start_time);
                        const total_nps = if (total_time > 0) (total_nodes * 1000) / @as(u64, @intCast(total_time)) else total_nodes * 1000;

                        try self.writeStdout("", .{});
                        try self.writeStdout("Total time: {d}ms", .{total_time});
                        try self.writeStdout("Nodes per second: {d}", .{total_nps});
                    },
                }
            },
            .quit => {
                // user wanted to quit, we return an error to break out of the loop
                return error.Quit;
            },
        }
    }

    fn search(self: *Self, go_opts: uci_command.GoOptions) UciError!void {
        try self.writeInfoString("search thread started", .{});

        const net_ptr: ?*const nnue.Network = if (self.nnue_network) |*network| network else null;
        const use_nnue_for_search = self.use_nnue and net_ptr != null;

        const prior_count = if (self.position_hash_count > 0) self.position_hash_count - 1 else 0;

        // Age TT before search (caller responsibility now)
        self.tt.nextAge();

        // Spawn helper threads for Lazy SMP (num_threads - 1 helpers)
        const num_helpers = self.num_threads - 1;
        for (0..num_helpers) |i| {
            self.helper_threads[i] = std.Thread.spawn(.{}, helperSearch, .{
                self,
                i,
                go_opts,
                net_ptr,
                use_nnue_for_search,
                prior_count,
            }) catch null;
        }

        // Main thread search
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
            self.nnue_screlu,
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
            self.joinHelpers(num_helpers);
            return UciError.IOError;
        };

        // Main thread done — stop all helpers
        self.stop_search.store(true, .seq_cst);
        self.joinHelpers(num_helpers);

        // Vote on best move if multi-threaded
        if (num_helpers > 0) {
            self.best_move = self.voteBestMove(result, num_helpers);
        } else {
            self.best_move = result.best_move;
        }

        // Sum nodes across all threads
        var total_nodes = result.nodes;
        for (0..num_helpers) |i| {
            total_nodes += self.helper_results[i].nodes;
        }

        try self.writeInfoString("search thread stopped, total nodes {d}", .{total_nodes});
        try self.writeStdout("bestmove {f}", .{self.best_move});
    }

    fn helperSearch(
        self: *Self,
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
            self.nnue_screlu,
        );

        // No UCI output from helpers
        search_engine.uci_output = null;
        if (prior_count > 0) {
            search_engine.setGameHistory(self.position_hash_history[0..prior_count]);
        } else {
            search_engine.setGameHistory(&.{});
        }

        // Depth stagger: even-indexed helpers start at depth 2
        const start_depth: u32 = if (idx % 2 == 0) 2 else 1;

        const search_opts = SearchOptions{
            .infinite = go_opts.infinite orelse false,
            .move_time = go_opts.move_time,
            .wtime = go_opts.wtime,
            .btime = go_opts.btime,
            .winc = go_opts.winc,
            .binc = go_opts.binc,
            .depth = go_opts.depth,
            .start_depth = start_depth,
        };

        const result = search_engine.search(search_opts) catch {
            self.helper_results[idx] = .{
                .best_move = board.Move.init(0, 0, null),
                .score = 0,
                .depth = 0,
                .nodes = 0,
            };
            return;
        };

        self.helper_results[idx] = .{
            .best_move = result.best_move,
            .score = result.score,
            .depth = result.depth,
            .nodes = result.nodes,
        };
    }

    fn joinHelpers(self: *Self, num_helpers: usize) void {
        for (0..num_helpers) |i| {
            if (self.helper_threads[i]) |thread| {
                thread.join();
                self.helper_threads[i] = null;
            }
        }
    }

    fn voteBestMove(self: *Self, main_result: search_module.SearchResult, num_helpers: usize) board.Move {
        // Collect all results (main + helpers)
        const max_voters = MAX_HELPERS + 1;
        var moves: [max_voters]board.Move = undefined;
        var scores: [max_voters]i32 = undefined;
        var depths: [max_voters]u32 = undefined;
        var count: usize = 0;

        // Add main thread result
        if (main_result.best_move.from() != 0 or main_result.best_move.to() != 0) {
            moves[count] = main_result.best_move;
            scores[count] = main_result.score;
            depths[count] = main_result.depth;
            count += 1;
        }

        // Add helper results
        for (0..num_helpers) |i| {
            const hr = self.helper_results[i];
            if (hr.best_move.from() != 0 or hr.best_move.to() != 0) {
                moves[count] = hr.best_move;
                scores[count] = hr.score;
                depths[count] = hr.depth;
                count += 1;
            }
        }

        if (count == 0) return main_result.best_move;
        if (count == 1) return moves[0];

        // Find worst score for normalization
        var worst_score: i32 = scores[0];
        for (0..count) |i| {
            if (scores[i] < worst_score) worst_score = scores[i];
        }

        // Vote: each thread votes for its move, weighted by depth + score bonus
        // We accumulate votes per unique move
        var vote_moves: [max_voters]board.Move = undefined;
        var vote_weights: [max_voters]i32 = undefined;
        var vote_best_depth: [max_voters]u32 = undefined;
        var vote_best_score: [max_voters]i32 = undefined;
        var num_unique: usize = 0;

        for (0..count) |i| {
            const weight = @as(i32, @intCast(depths[i])) + @divTrunc(scores[i] - worst_score, 10);

            // Find if this move already exists in votes
            var found: bool = false;
            for (0..num_unique) |j| {
                if (vote_moves[j].from() == moves[i].from() and
                    vote_moves[j].to() == moves[i].to() and
                    board.Move.eqlPromotion(vote_moves[j].promotion(), moves[i].promotion()))
                {
                    vote_weights[j] += weight;
                    if (depths[i] > vote_best_depth[j] or
                        (depths[i] == vote_best_depth[j] and scores[i] > vote_best_score[j]))
                    {
                        vote_best_depth[j] = depths[i];
                        vote_best_score[j] = scores[i];
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
                num_unique += 1;
            }
        }

        // Pick move with highest total vote weight
        var best_idx: usize = 0;
        for (1..num_unique) |i| {
            if (vote_weights[i] > vote_weights[best_idx] or
                (vote_weights[i] == vote_weights[best_idx] and vote_best_depth[i] > vote_best_depth[best_idx]) or
                (vote_weights[i] == vote_weights[best_idx] and vote_best_depth[i] == vote_best_depth[best_idx] and vote_best_score[i] > vote_best_score[best_idx]))
            {
                best_idx = i;
            }
        }

        return vote_moves[best_idx];
    }

    fn writeStdout(self: *Self, comptime fmt: []const u8, args: anytype) UciError!void {
        const line = std.fmt.allocPrint(self.allocator, fmt, args) catch return UciError.OutOfMemory;
        defer self.allocator.free(line);

        self.stdout.writeAll(line) catch return UciError.IOError;
        self.stdout.writeAll("\n") catch return UciError.IOError;

        // Log output
        if (self.log_file) |file| {
            file.writeAll(line) catch return UciError.IOError;
            file.writeAll("\n") catch return UciError.IOError;
        }
    }

    fn writeInfoString(self: *Self, comptime fmt: []const u8, args: anytype) UciError!void {
        if (!self.debug) {
            return;
        }

        const line = std.fmt.allocPrint(self.allocator, fmt, args) catch return UciError.OutOfMemory;
        defer self.allocator.free(line);

        try self.writeStdout("info string {s}", .{line});
    }

    fn handleLogFileChange(self: *Self, value: []const u8) UciError!void {
        if (self.log_file) |file| {
            file.close();
            self.log_file = null;
        }

        if (std.mem.eql(u8, value, "<empty>") or value.len == 0) {
            return;
        }

        self.log_file = std.fs.cwd().openFile(value, .{ .mode = .read_write }) catch
            std.fs.cwd().createFile(value, .{}) catch return UciError.IOError;

        self.log_file.?.seekFromEnd(0) catch return UciError.IOError;
    }

    fn handleUseNnueChange(self: *Self, value: []const u8) UciError!void {
        if (std.mem.eql(u8, value, "true")) {
            self.use_nnue = true;
            return;
        }
        if (std.mem.eql(u8, value, "false")) {
            self.use_nnue = false;
            return;
        }
        return UciError.InvalidArgument;
    }

    fn handleEvalFileChange(self: *Self, value: []const u8) UciError!void {
        if (std.mem.eql(u8, value, "<empty>") or value.len == 0) {
            if (self.nnue_network) |*network| {
                network.deinit();
                self.nnue_network = null;
            }
            if (self.eval_file_path) |old_path| {
                self.allocator.free(old_path);
                self.eval_file_path = null;
            }
            return;
        }

        var loaded = nnue.Network.loadFromFile(self.allocator, value) catch |err| {
            return switch (err) {
                nnue.LoadError.OutOfMemory => UciError.OutOfMemory,
                nnue.LoadError.InvalidNetwork,
                nnue.LoadError.UnsupportedVersion,
                nnue.LoadError.NetworkTooLarge,
                => UciError.InvalidArgument,
                else => UciError.IOError,
            };
        };

        const dup_path = self.allocator.dupe(u8, value) catch {
            loaded.deinit();
            return UciError.OutOfMemory;
        };

        if (self.nnue_network) |*old_network| {
            old_network.deinit();
        }
        if (self.eval_file_path) |old_path| {
            self.allocator.free(old_path);
        }

        self.nnue_network = loaded;
        self.eval_file_path = dup_path;
    }

    fn handleNnueBlendChange(self: *Self, value: []const u8) UciError!void {
        const parsed = std.fmt.parseInt(i32, value, 10) catch return UciError.InvalidArgument;
        if (parsed < 0 or parsed > 100) {
            return UciError.InvalidArgument;
        }
        self.nnue_blend = parsed;
    }

    fn handleNnueScaleChange(self: *Self, value: []const u8) UciError!void {
        const parsed = std.fmt.parseInt(i32, value, 10) catch return UciError.InvalidArgument;
        if (parsed < 10 or parsed > 400) {
            return UciError.InvalidArgument;
        }
        self.nnue_scale = parsed;
    }

    fn handleThreadsChange(self: *Self, value: []const u8) UciError!void {
        const parsed = std.fmt.parseInt(usize, value, 10) catch return UciError.InvalidArgument;
        if (parsed < 1 or parsed > 64) {
            return UciError.InvalidArgument;
        }
        self.num_threads = parsed;
    }

    fn handleHashChange(self: *Self, value: []const u8) UciError!void {
        const parsed = std.fmt.parseInt(usize, value, 10) catch return UciError.InvalidArgument;
        if (parsed < 1 or parsed > 4096) {
            return UciError.InvalidArgument;
        }
        self.hash_size_mb = parsed;
        self.tt.resize(parsed) catch return UciError.OutOfMemory;
    }

    fn handleNnueScReluChange(self: *Self, value: []const u8) UciError!void {
        if (std.mem.eql(u8, value, "true")) {
            self.nnue_screlu = true;
            return;
        }
        if (std.mem.eql(u8, value, "false")) {
            self.nnue_screlu = false;
            return;
        }
        return UciError.InvalidArgument;
    }

    fn resetPositionHistory(self: *Self) void {
        self.position_hash_count = 0;
        self.pushCurrentHashToPositionHistory();
    }

    fn pushCurrentHashToPositionHistory(self: *Self) void {
        const hash = self.board.zobrist_hasher.zobrist_hash;
        if (self.position_hash_count < self.position_hash_history.len) {
            self.position_hash_history[self.position_hash_count] = hash;
            self.position_hash_count += 1;
            return;
        }

        // Keep the most recent positions if history grows beyond fixed capacity.
        std.mem.copyForwards(
            u64,
            self.position_hash_history[0 .. self.position_hash_history.len - 1],
            self.position_hash_history[1..self.position_hash_history.len],
        );
        self.position_hash_history[self.position_hash_history.len - 1] = hash;
    }
};
