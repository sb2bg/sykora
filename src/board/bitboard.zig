const std = @import("std");
const UciError = @import("../uci/uci_error.zig").UciError;

pub const BitBoard = struct {
    const Self = @This();

    pub fn algebraicToBitboard(_: Self, _: []const u8) UciError!u64 {
        return 0;
    }

    pub fn makeMove(_: Self, _: u64) UciError!void {}

    pub fn reset(_: Self) UciError!void {}
};
