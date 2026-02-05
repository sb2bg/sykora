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
        var moves_list = std.ArrayList([]const u8).empty;
        defer moves_list.deinit(self.allocator);
        var move = parser.next();
        while (move) |m| {
            try moves_list.append(self.allocator, m);
            move = parser.next();
        }

        return moves_list.toOwnedSlice(self.allocator) catch null;
    }

    /// Helper function to parse a u64 parameter from the token iterator
    fn parseU64Param(parser: *std.mem.TokenIterator(u8, .any)) UciError!u64 {
        const value = parser.next() orelse return error.UnexpectedEOF;
        return std.fmt.parseInt(u64, value, 10) catch return error.InvalidArgument;
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
                if (parser.next()) |next| {
                    if (!std.mem.eql(u8, next, "name")) {
                        return error.InvalidArgument;
                    }
                } else return error.UnexpectedEOF;

                var name_parts = std.ArrayList([]const u8).empty;
                defer name_parts.deinit(self.allocator);
                var curr = parser.next();
                while (curr != null and !std.mem.eql(u8, curr.?, "value")) {
                    try name_parts.append(self.allocator, curr.?);
                    curr = parser.next();
                }

                const name = try std.mem.join(self.allocator, " ", name_parts.items);

                if (curr == null) {
                    // we are a button
                    return ToEngineCommand{ .setoption = .{ .name = name, .value = null } };
                }

                if (!std.mem.eql(u8, curr.?, "value")) {
                    return error.UnknownCommand;
                }

                // TODO: check if value can have spaces
                const value = parser.next() orelse return error.UnexpectedEOF;
                return ToEngineCommand{ .setoption = .{ .name = name, .value = value } };
            },
            .ucinewgame => {
                return ToEngineCommand.ucinewgame;
            },
            .position => {
                const sub_command = parser.next() orelse return error.UnexpectedEOF;
                const is_startpos = std.mem.eql(u8, sub_command, "startpos");
                const is_fen = std.mem.eql(u8, sub_command, "fen");

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
                var fen_parts = std.ArrayList([]const u8).empty;
                defer fen_parts.deinit(self.allocator);

                var curr = parser.next();
                while (curr != null and !std.mem.eql(u8, curr.?, "moves")) {
                    try fen_parts.append(self.allocator, curr.?);
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

                // Check for optional mode argument
                const mode: cmd.PerftMode = if (parser.next()) |mode_str| blk: {
                    if (std.mem.eql(u8, mode_str, "stats")) {
                        break :blk .stats;
                    } else if (std.mem.eql(u8, mode_str, "divide")) {
                        break :blk .divide;
                    } else {
                        return error.InvalidArgument;
                    }
                } else .divide;

                return ToEngineCommand{ .perft = .{ .depth = parsed_depth, .mode = mode } };
            },
            .go => {
                var go_params: cmd.GoOptions = .{};

                while (parser.next()) |param| {
                    if (std.mem.eql(u8, param, "searchmoves")) {
                        var moves_list = std.ArrayList([]const u8).empty;
                        defer moves_list.deinit(self.allocator);
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
                            try moves_list.append(self.allocator, move);
                        }
                        go_params.search_moves = moves_list.toOwnedSlice(self.allocator) catch &[_][]const u8{};
                    } else if (std.mem.eql(u8, param, "ponder")) {
                        go_params.ponder = true;
                    } else if (std.mem.eql(u8, param, "wtime")) {
                        go_params.wtime = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "btime")) {
                        go_params.btime = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "winc")) {
                        go_params.winc = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "binc")) {
                        go_params.binc = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "movestogo")) {
                        go_params.moves_to_go = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "depth")) {
                        go_params.depth = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "nodes")) {
                        go_params.nodes = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "mate")) {
                        go_params.mate = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "movetime")) {
                        go_params.move_time = try parseU64Param(&parser);
                    } else if (std.mem.eql(u8, param, "infinite")) {
                        go_params.infinite = true;
                    }
                }

                return ToEngineCommand{ .go = go_params };
            },
        };
    }
};
