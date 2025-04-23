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
    uciParser: UciParser,
    debug: bool,
    options: std.StringHashMap([]const u8),
    searchThread: ?std.Thread,
    board: Board,
    allocator: std.mem.Allocator,

    pub fn init(stdin: std.io.AnyReader, stdout: std.io.AnyWriter, allocator: std.mem.Allocator) !Self {
        const uci = Uci{
            .stdin = stdin,
            .stdout = stdout,
            .uciParser = UciParser.init(allocator),
            .searchThread = null,
            .debug = false,
            .options = std.StringHashMap([]const u8).init(allocator),
            .board = Board.startpos(),
            .allocator = allocator,
        };

        try uci.writeStdout("{s} version {s} by {s}", .{ name, version, author });
        return uci;
    }

    pub fn run(self: *Self) UciError!void {
        var buf = std.ArrayList(u8).init(self.allocator);
        defer buf.deinit();

        while (true) {
            self.stdin.streamUntilDelimiter(buf.writer(), '\n', null) catch {
                return error.IOError;
            };

            const command = self.uciParser.parseCommand(buf.items) catch |err| {
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

            buf.clearRetainingCapacity();
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

                if (self.searchThread != null) {
                    // TODO: is this what we want?
                    self.searchThread.?.detach();
                }
            },
            .position => |positionOptions| {
                switch (positionOptions.value) {
                    .startpos => {
                        self.board = Board.startpos();
                    },
                    .fen => {
                        const parsedBoard = Board.fromFen(positionOptions.value.fen);
                        // free the fen string since we don't need it anymore
                        self.allocator.free(positionOptions.value.fen);
                        self.board = try parsedBoard;
                    },
                }

                if (positionOptions.moves) |moves| {
                    for (moves) |move| {
                        try self.board.makeMove(move);
                    }
                }
            },
            .display => {
                try self.writeStdout("{}", .{self.board});
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
