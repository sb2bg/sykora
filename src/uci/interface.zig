const std = @import("std");
const UciParser = @import("uci_parser.zig").UciParser;
const ToEngineCommand = @import("uci_command.zig").ToEngineCommand;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const board = @import("../board/bitboard.zig");
const BitBoard = board.BitBoard;

const name = "Sykora";
const author = "Sullivan Bognar";
const version = "0.1.0";

pub const Uci = struct {
    const Self = @This();
    const maxCommandLength = 1024;
    stdin: std.io.AnyReader,
    stdout: std.io.AnyWriter,
    uciParser: UciParser,
    debug: bool,
    options: std.StringHashMap([]const u8),
    searchThread: ?std.Thread,
    board: BitBoard,

    pub fn init(stdin: std.io.AnyReader, stdout: std.io.AnyWriter, allocator: std.mem.Allocator) !Self {
        const uci = Uci{
            .stdin = stdin,
            .stdout = stdout,
            .uciParser = UciParser{},
            .searchThread = null,
            .debug = false,
            .options = std.StringHashMap([]const u8).init(allocator),
            .board = BitBoard{},
        };

        try uci.writeStdout("{s} version {s} by {s}", .{ name, version, author });
        return uci;
    }

    pub fn run(self: *Self) UciError!void {
        while (true) {
            var buf: std.BoundedArray(u8, maxCommandLength) = .{};

            self.stdin.streamUntilDelimiter(
                buf.writer(),
                '\n',
                maxCommandLength,
            ) catch {
                return error.CommandTooLong;
            };

            const command = self.uciParser.parseCommand(buf.buffer[0..buf.len]) catch |err| {
                try self.writeInfoString("{s}", .{uciErr.getErrorDescriptor(err)});
                continue;
            };

            self.handleCommand(command) catch |err| {
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
                try self.board.reset();

                if (self.searchThread != null) {
                    // TODO: is this what we want?
                    self.searchThread.?.detach();
                }
            },
            .position => |positionOptions| {
                switch (positionOptions.value) {
                    .startpos => {
                        try self.board.reset();
                    },
                    .fen => {
                        try self.writeInfoString("FEN not yet implemented.", .{});
                        // TODO: implement
                    },
                }

                if (positionOptions.moves) |moves| {
                    for (moves) |algebraicMove| {
                        const move = try BitBoard.algebraicToBitboard(algebraicMove);
                        try self.board.makeMove(move);
                    }
                }
            },
            .quit => {
                // user wanted to quit, no need to return an error
                std.process.exit(0);
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
