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

const name = "Sykora";
const author = "Sullivan Bognar";
const version = "0.1.0";

pub const Uci = struct {
    const Self = @This();
    stdin: std.io.AnyReader,
    stdout: std.io.AnyWriter,
    uci_parser: UciParser,
    debug: bool,
    options: Options,
    stop_search: std.atomic.Value(bool),
    board: Board,
    allocator: std.mem.Allocator,
    search_thread: ?std.Thread,
    best_move: board.Move,
    log_file: ?std.fs.File,

    pub fn init(stdin: std.io.AnyReader, stdout: std.io.AnyWriter, allocator: std.mem.Allocator) !*Self {
        const uci_ptr = try allocator.create(Self);
        const stop_search = std.atomic.Value(bool).init(false);

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
        };

        // Add logging option
        try uci_ptr.options.items.append(Option{
            .name = "Debug Log File",
            .type = .string,
            .default_value = "<empty>",
            .on_changed = handleLogFileChange,
            .context = uci_ptr,
        });

        try uci_ptr.writeStdout("{s} version {s} by {s}", .{ name, version, author });
        return uci_ptr;
    }

    pub fn deinit(self: *Self) void {
        if (self.log_file) |file| {
            file.close();
        }
        self.options.deinit();
        self.allocator.destroy(self);
    }

    pub fn run(self: *Self) UciError!void {
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();

        while (true) {
            defer buf.clearRetainingCapacity();
            self.stdin.streamUntilDelimiter(buf.writer(), '\n', null) catch return UciError.IOError;

            // Log input
            if (self.log_file) |file| {
                const writer = file.writer();
                writer.print("> {s}\n", .{buf.items}) catch return UciError.IOError;
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
            const start = std.time.milliTimestamp();
            thread.join(); // block until it finishes
            self.search_thread = null;
            const end = std.time.milliTimestamp();

            const duration = end - start;
            try self.writeInfoString("search thread joined in {d}ms", .{duration});
        }
    }

    fn handleCommand(self: *Self, command: ToEngineCommand) UciError!void {
        switch (command) {
            .uci => {
                try self.writeStdout("id name {s}", .{name});
                try self.writeStdout("id author {s}", .{author});
                try self.options.printOptions(self.stdout);
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

                if (pos_opts.moves) |moves| {
                    defer self.allocator.free(moves);

                    for (moves) |move| {
                        try self.board.makeStrMove(move);
                    }
                }
            },
            .display => {
                try self.writeStdout("{}", .{self.board});
                const fen = try self.board.getFenString(self.allocator);
                defer self.allocator.free(fen);
                try self.writeStdout("fen {s}", .{fen});
                try self.writeStdout("key {x}", .{self.board.zobrist_hasher.zobrist_hash});
            },
            .go => |go_opts| {
                try self.writeInfoString("{any}", .{go_opts});
                try self.writeInfoString("starting search thread", .{});
                self.stop_search.store(false, .seq_cst);
                self.best_move = board.Move.init(0, 0, null);

                self.search_thread = std.Thread.spawn(.{}, Uci.search, .{ self, go_opts }) catch return UciError.ThreadCreationFailed;
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
            .perft => |depth| {
                try self.writeStdout("", .{});
                try self.writeStdout("Running perft to depth {d}...", .{depth});
                try self.writeStdout("", .{});

                // Print header
                try self.writeStdout("Depth | Nodes      | Captures   | E.p. | Castles | Promotions | Checks | Discovery | Double | Checkmates | Time(ms)", .{});
                try self.writeStdout("------|------------|------------|------|---------|------------|--------|-----------|--------|------------|----------", .{});

                const start_time = std.time.milliTimestamp();

                // Run perft for each depth up to target depth
                var d: u32 = 1;
                while (d <= depth) : (d += 1) {
                    const depth_start = std.time.milliTimestamp();
                    var stats = board.Board.PerftStats{};
                    try self.board.perftWithStats(@intCast(d), self.allocator, &stats);
                    const depth_time = std.time.milliTimestamp() - depth_start;

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

                try self.writeStdout("", .{});
                try self.writeStdout("Perft divide at depth {d}:", .{depth});
                try self.writeStdout("", .{});

                const total_nodes = try self.board.perftDivide(@intCast(depth), self.allocator, self.stdout);
                const total_time = std.time.milliTimestamp() - start_time;
                const total_nps = if (total_time > 0) (total_nodes * 1000) / @as(u64, @intCast(total_time)) else total_nodes * 1000;

                try self.writeStdout("", .{});
                try self.writeStdout("Total time: {d}ms", .{total_time});
                try self.writeStdout("Nodes per second: {d}", .{total_nps});
            },
            .quit => {
                // user wanted to quit, we return an error to break out of the loop
                return error.Quit;
            },
        }
    }

    fn search(self: *Self, go_opts: uci_command.GoOptions) UciError!void {
        try self.writeInfoString("search thread started", .{});

        const start_time = std.time.milliTimestamp();
        var time_limit: ?u64 = null;

        if (go_opts.infinite) |infinite| {
            if (infinite) {
                time_limit = null;
            }
        } else if (go_opts.move_time) |move_time| {
            time_limit = move_time;
        } else if (go_opts.wtime) |wtime| {
            time_limit = wtime / 100;
        } else if (go_opts.btime) |btime| {
            time_limit = btime / 100;
        }

        const legal_moves = try self.board.generateLegalMoves(self.allocator);
        defer self.allocator.free(legal_moves);

        if (legal_moves.len == 0) {
            try self.writeInfoString("no legal moves available", .{});
            self.best_move = board.Move.init(0, 0, null);
            try self.writeStdout("bestmove {s}", .{self.best_move});
            return;
        }

        var best_eval: i32 = -9999;
        var best_move: ?board.Move = null;
        var nodes: usize = 0;

        for (legal_moves) |mv| {
            if (self.stop_search.load(.seq_cst)) break;

            // Dummy evaluation for demonstration - just prefer captures and central squares
            const score: i32 = 20 + @as(i32, @intCast(mv.to));
            nodes += 1;

            try self.writeStdout(
                "info depth 1 seldepth 1 score cp {d} nodes {d} nps {d} time {d} pv {s}",
                .{
                    score,
                    nodes,
                    if (std.time.milliTimestamp() - start_time > 0)
                        nodes * 1000 / @as(usize, @intCast(std.time.milliTimestamp() - start_time))
                    else
                        nodes * 1000,
                    std.time.milliTimestamp() - start_time,
                    mv,
                },
            );

            if (score > best_eval) {
                best_eval = score;
                best_move = mv;
            }

            // Check time limit
            if (time_limit) |limit| {
                if (std.time.milliTimestamp() - start_time > limit) break;
            }

            std.time.sleep(5 * std.time.ns_per_ms); // simulate work
        }

        self.best_move = best_move orelse legal_moves[0];
        try self.writeInfoString("search thread stopped", .{});
        try self.writeStdout("bestmove {s}", .{self.best_move});
    }

    fn writeStdout(self: Self, comptime fmt: []const u8, args: anytype) UciError!void {
        self.stdout.print(fmt, args) catch return UciError.IOError;
        self.stdout.writeByte('\n') catch return UciError.IOError;

        // Log output
        if (self.log_file) |file| {
            const writer = file.writer();
            writer.print(fmt, args) catch return UciError.IOError;
            writer.writeByte('\n') catch return UciError.IOError;
        }
    }

    fn writeInfoString(self: Self, comptime fmt: []const u8, args: anytype) UciError!void {
        if (!self.debug) {
            return;
        }

        self.stdout.print("info string ", .{}) catch return UciError.IOError;
        try self.writeStdout(fmt, args);
    }

    fn handleLogFileChange(self: *Self, value: []const u8) !void {
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
};
