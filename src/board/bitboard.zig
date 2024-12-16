const std = @import("std");
const UciError = @import("../uci/uci_error.zig").UciError;

pub const BitBoard = struct {
    const Self = @This();

    pub fn reset(_: Self) UciError!void {}
};
