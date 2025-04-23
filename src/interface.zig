const std = @import("std");
const UciParser = @import("uci_parser.zig").UciParser;
const ToEngineCommand = @import("uci_command.zig").ToEngineCommand;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const board = @import("bitboard.zig");
const Board = board.Board;

const name = "Sykora";
const author = "Sullivan Bognar";
const version = "0.1.0";

pub const Uci = struct {
    const Self = @This();
    stdin: std.io.AnyReader,
    stdout: std.io.AnyWriter,
    uci_parser: UciParser,
    debug: bool,
    options: std.StringHashMap([]const u8),
    search_thread: ?std.Thread,
    stop_search: std.atomic.Value(bool),
    board: Board,
    allocator: std.mem.Allocator,

    pub fn init(stdin: std.io.AnyReader, stdout: std.io.AnyWriter, allocator: std.mem.Allocator) !Self {
        const stop_search = std.atomic.Value(bool).init(false);
        const uci = Uci{
            .stdin = stdin,
            .stdout = stdout,
            .uci_parser = UciParser.init(allocator),
            .search_thread = null,
            .debug = false,
            .options = std.StringHashMap([]const u8).init(allocator),
            .board = Board.startpos(),
            .allocator = allocator,
            .stop_search = stop_search,
        };

        try uci.writeStdout("{s} version {s} by {s}", .{ name, version, author });
        return uci;
    }

    pub fn run(self: *Self) UciError!void {
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();

        while (true) {
            defer buf.clearRetainingCapacity();

            self.stdin.streamUntilDelimiter(buf.writer(), '\n', null) catch {
                return error.IOError;
            };

            const command = self.uci_parser.parseCommand(buf.items) catch |err| {
                try self.writeInfoString("{s}", .{uciErr.getErrorDescriptor(err)});
                continue;
            };

            self.handleCommand(command) catch |err| {
                if (err == UciError.Quit) {
                    // short circuit the loop to exit
                    return;
                }

                try self.writeInfoString("{s}", .{uciErr.getErrorDescriptor(err)});
            };
        }
    }

    fn handleCommand(self: *Self, command: ToEngineCommand) UciError!void {
        switch (command) {
            .uci => {
                try self.writeStdout("id name {s}", .{name});
                try self.writeStdout("id author {s}", .{author});
                try self.writeStdout("uciok", .{});
            },
            .debug => |value| {
                self.debug = value;
            },
            .isready => {
                try self.writeStdout("readyok", .{});
            },
            .ucinewgame => {
                self.board = Board.startpos();

                if (self.search_thread) |thread| {
                    self.stop_search.store(true, .seq_cst);
                    thread.detach();
                    self.search_thread = null;
                }
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
                        try self.board.makeMove(move);
                    }
                }
            },
            .display => {
                try self.writeStdout("{}", .{self.board});
                try self.writeStdout("fen {s}", .{try self.board.getFenString()});
            },
            .go => |_| {
                return error.Unimplemented;
            },
            .stop => {
                self.stop_search.store(true, .seq_cst);
            },
            .ponderhit => {
                return error.Unimplemented;
            },
            .quit => {
                // user wanted to quit, we return an error to break out of the loop
                return error.Quit;
            },
            else => {
                return error.Unimplemented;
            },
        }
    }

    fn writeStdout(self: Self, comptime fmt: []const u8, args: anytype) UciError!void {
        self.stdout.print(fmt, args) catch {
            return error.IOError;
        };

        self.stdout.writeByte('\n') catch {
            return error.IOError;
        };
    }

    fn writeInfoString(self: Self, comptime fmt: []const u8, args: anytype) UciError!void {
        if (!self.debug) {
            return;
        }

        self.stdout.print("info string ", .{}) catch {
            return error.IOError;
        };

        try self.writeStdout(fmt, args);
    }
};
