const std = @import("std");
const uci = @import("interface.zig");
const Uci = uci.Uci;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
pub fn main() void {
    tryMain() catch |err| {
        std.log.err("Encountered an unrecoverable error while running Sykora.\n", .{});
        std.log.err("\t|> {s}\n", .{uciErr.getErrorDescriptor(err)});
    };
}

fn tryMain() UciError!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var uci_interface = try Uci.init(
        std.io.getStdIn().reader().any(),
        std.io.getStdOut().writer().any(),
        allocator,
    );
    try uci_interface.run();
}
