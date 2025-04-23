const std = @import("std");
const cmd = @import("uci_command.zig");
const ToEngine = cmd.ToEngine;
const ToEngineCommand = cmd.ToEngineCommand;
const UciError = @import("uci_error.zig").UciError;

pub const UciParser = struct {
    const Self = @This();
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn parseCommand(self: Self, command: []const u8) UciError!ToEngineCommand {
        var parser = std.mem.tokenizeAny(u8, command, " ");
        const uciCmd = parser.next() orelse return error.UnexpectedEOF;

        const case = std.meta.stringToEnum(ToEngine, uciCmd) orelse {
            return error.UnknownCommand;
        };

        return switch (case) {
            .uci => {
                return ToEngineCommand.uci;
            },
            .debug => {
                const option = parser.next() orelse return error.UnexpectedEOF;

                if (std.mem.eql(u8, option, "on")) {
                    return ToEngineCommand{ .debug = true };
                } else if (std.mem.eql(u8, option, "off")) {
                    return ToEngineCommand{ .debug = false };
                } else {
                    return error.UnknownCommand;
                }
            },
            .isready => {
                return ToEngineCommand.isready;
            },
            .setoption => {
                const name = parser.next() orelse return error.UnexpectedEOF;
                // FIXME: there may not be a value in the case of a button, which is perfectly valid.
                const value = parser.next() orelse return error.UnexpectedEOF;
                return ToEngineCommand{ .setoption = .{ .name = name, .value = value } };
            },
            .register => {
                const subCommand = parser.next() orelse return error.UnexpectedEOF;

                if (std.mem.eql(u8, subCommand, "later")) {
                    return ToEngineCommand{
                        .register = .later,
                    };
                } else if (std.mem.eql(u8, subCommand, "now")) {
                    const name = parser.next() orelse return error.UnexpectedEOF;
                    const code = parser.next() orelse return error.UnexpectedEOF;
                    return ToEngineCommand{ .register = .{
                        .now = .{
                            .name = name,
                            .code = code,
                        },
                    } };
                } else {
                    return error.UnknownCommand;
                }
            },
            .ucinewgame => {
                return ToEngineCommand.ucinewgame;
            },
            .position => {
                const subCommand = parser.next() orelse return error.UnexpectedEOF;

                if (std.mem.eql(u8, subCommand, "startpos")) {
                    return ToEngineCommand{ .position = .{
                        .value = .startpos,
                        .moves = null,
                    } };
                } else if (std.mem.eql(u8, subCommand, "fen")) {
                    var fen_parts = std.ArrayList([]const u8).init(self.allocator);
                    defer fen_parts.deinit();

                    var curr = parser.next();
                    while (curr != null and !std.mem.eql(u8, curr.?, "moves")) {
                        try fen_parts.append(curr.?);
                        curr = parser.next();
                    }

                    const fen_joined = try std.mem.join(self.allocator, " ", fen_parts.items);

                    return ToEngineCommand{ .position = .{
                        .value = .{ .fen = fen_joined },
                        .moves = null,
                    } };
                } else {
                    return error.UnknownCommand;
                }
            },
            // .go => {
            //     return ToEngineCommand{ .go = .{} };
            // },
            .stop => {
                return ToEngineCommand.stop;
            },
            .ponderhit => {
                return ToEngineCommand.ponderhit;
            },
            .quit => {
                return ToEngineCommand.quit;
            },
            .display => {
                return ToEngineCommand.display;
            },
            .perft => {
                const depth = parser.next() orelse return error.UnexpectedEOF;

                const passedDepth = std.fmt.parseInt(u64, depth, 10) catch {
                    return error.InvalidArgument;
                };

                return ToEngineCommand{ .perft = passedDepth };
            },
            else => {
                // TODO: remove this once all commands are implemented
                return error.Unimplemented;
            },
        };
    }
};
