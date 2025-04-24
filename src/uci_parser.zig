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

    fn parseMoves(self: Self, parser: *std.mem.TokenIterator(u8, .any)) !?[][]const u8 {
        var moves_list = std.ArrayList([]const u8).init(self.allocator);
        var move = parser.next();
        while (move) |m| {
            try moves_list.append(m);
            move = parser.next();
        }

        return moves_list.toOwnedSlice() catch null;
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

                const is_startpos = std.mem.eql(u8, subCommand, "startpos");
                const is_fen = std.mem.eql(u8, subCommand, "fen");

                if (!is_startpos and !is_fen) return error.UnknownCommand;

                var moves: ?[][]const u8 = null;

                if (is_startpos) {
                    const next_tok = parser.next();
                    if (next_tok != null and std.mem.eql(u8, next_tok.?, "moves")) {
                        moves = try self.parseMoves(&parser);
                    }

                    return ToEngineCommand{ .position = .{
                        .value = .startpos,
                        .moves = moves,
                    } };
                }

                // "fen" case
                var fen_parts = std.ArrayList([]const u8).init(self.allocator);
                defer fen_parts.deinit();

                var curr = parser.next();
                while (curr != null and !std.mem.eql(u8, curr.?, "moves")) {
                    try fen_parts.append(curr.?);
                    curr = parser.next();
                }

                if (curr != null and std.mem.eql(u8, curr.?, "moves")) {
                    moves = try self.parseMoves(&parser);
                }

                const fen_joined = try std.mem.join(self.allocator, " ", fen_parts.items);

                return ToEngineCommand{ .position = .{
                    .value = .{ .fen = fen_joined },
                    .moves = moves,
                } };
            },
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
                const parsed_depth = std.fmt.parseInt(u64, depth, 10) catch return UciError.InvalidArgument;

                return ToEngineCommand{ .perft = parsed_depth };
            },
            .go => {
                var go_params: cmd.GoOptions = .{};

                while (parser.next()) |param| {
                    if (std.mem.eql(u8, param, "searchmoves")) {
                        var moves_list = std.ArrayList([]const u8).init(self.allocator);
                        while (parser.peek()) |peeked| {
                            if (std.mem.startsWith(u8, peeked, "ponder") or
                                std.mem.startsWith(u8, peeked, "wtime") or
                                std.mem.startsWith(u8, peeked, "btime") or
                                std.mem.startsWith(u8, peeked, "winc") or
                                std.mem.startsWith(u8, peeked, "binc") or
                                std.mem.startsWith(u8, peeked, "movestogo") or
                                std.mem.startsWith(u8, peeked, "depth") or
                                std.mem.startsWith(u8, peeked, "nodes") or
                                std.mem.startsWith(u8, peeked, "mate") or
                                std.mem.startsWith(u8, peeked, "movetime") or
                                std.mem.startsWith(u8, peeked, "infinite"))
                            {
                                break;
                            }
                            const move = parser.next() orelse break;
                            try moves_list.append(move);
                        }
                        go_params.search_moves = moves_list.toOwnedSlice() catch &[_][]const u8{};
                    } else if (std.mem.eql(u8, param, "ponder")) {
                        go_params.ponder = true;
                    } else if (std.mem.eql(u8, param, "wtime")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.wtime = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "btime")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.btime = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "winc")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.winc = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "binc")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.binc = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "movestogo")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.moves_to_go = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "depth")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.depth = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "nodes")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.nodes = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "mate")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.mate = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "movetime")) {
                        const value = parser.next() orelse return error.UnexpectedEOF;
                        go_params.move_time = std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
                    } else if (std.mem.eql(u8, param, "infinite")) {
                        go_params.infinite = true;
                    }
                }

                return ToEngineCommand{ .go = go_params };
            },
        };
    }
};
