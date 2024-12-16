const std = @import("std");
const UciParser = @import("uci_parser.zig").UciParser;
const ToEngineCommand = @import("uci_command.zig").ToEngineCommand;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;

pub const Uci = struct {
    const Self = @This();
    const maxCommandLength = 1024;
    stdin: std.io.AnyReader,
    stdout: std.io.AnyWriter,
    uciParser: UciParser,
    searchThread: ?std.Thread,
    debug: bool,
    options: std.StringHashMap([]const u8),

    pub fn init(stdin: std.io.AnyReader, stdout: std.io.AnyWriter, allocator: std.mem.Allocator) Self {
        return Uci{
            .stdin = stdin,
            .stdout = stdout,
            .uciParser = UciParser{},
            .searchThread = null,
            .debug = false,
            .options = std.StringHashMap([]const u8).init(allocator),
        };
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
                try self.writeInfoString(uciErr.getErrorDescriptor(err));
                continue;
            };

            self.handleCommand(command) catch |err| {
                try self.writeInfoString(uciErr.getErrorDescriptor(err));
            };
        }
    }

    const name = "Sykora";
    const author = "Sullivan Bognar";

    fn handleCommand(self: *Self, command: ToEngineCommand) UciError!void {
        switch (command) {
            .uci => {
                try self.writeStdout("id name " ++ name);
                try self.writeStdout("id author " ++ author);
                try self.writeStdout("uciok");
            },
            .debug => |value| {
                self.debug = value;
            },
            .isready => {
                try self.writeStdout("readyok");
            },
            .setoption => |setOptionOptions| {
                self.options.put(setOptionOptions.name, setOptionOptions.value) catch {
                    return error.OutOfMemory;
                };
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

    fn writeStdout(self: Self, message: []const u8) UciError!void {
        self.stdout.print("{s}\n", .{message}) catch {
            return error.IOError;
        };
    }

    fn writeInfoString(self: Self, message: []const u8) UciError!void {
        if (!self.debug) {
            return;
        }

        self.stdout.print("info string {s}\n", .{message}) catch {
            return error.IOError;
        };
    }
};
