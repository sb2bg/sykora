const std = @import("std");
const uci = @import("uci/interface.zig");
const uciErr = @import("uci/uci_error.zig");
const UciError = uciErr.UciError;

pub fn main() void {
    tryMain() catch |err| {
        std.debug.print("Encountered an unrecoverable error while running Sykora.\n", .{});
        std.debug.print("\t|> {s}\n", .{uciErr.getErrorDescriptor(err)});
    };
}

fn tryMain() UciError!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var uci_interface = uci.Uci.init(std.io.getStdIn().reader().any(), std.io.getStdOut().writer().any());
    try uci_interface.run();
}
