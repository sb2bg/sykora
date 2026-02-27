const std = @import("std");
const uci = @import("interface.zig");
const Uci = uci.Uci;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const gensfen = @import("gensfen.zig");

pub fn main() void {
    tryMain() catch |err| {
        std.log.err("Encountered an unrecoverable error while running Sykora.\n", .{});
        switch (err) {
            error.GensfenError => {},
            else => {
                std.log.err("\t|> {s}\n", .{uciErr.getErrorDescriptor(@errorCast(err))});
            },
        }
    };
}

const MainError = UciError || error{GensfenError};

fn tryMain() MainError!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check for subcommands via CLI args
    var args_iter = std.process.args();
    _ = args_iter.next(); // skip argv[0]

    if (args_iter.next()) |subcmd| {
        if (std.mem.eql(u8, subcmd, "gensfen")) {
            return runGensfen(&args_iter, allocator);
        }
        // "uci" or unrecognized → fall through to UCI mode
    }

    // Default: UCI mode
    var uci_interface = try Uci.init(
        std.fs.File.stdin(),
        std.fs.File.stdout(),
        allocator,
    );
    defer uci_interface.deinit();
    try uci_interface.run();
}

fn runGensfen(args_iter: *std.process.ArgIterator, allocator: std.mem.Allocator) MainError!void {
    // Collect remaining args
    var args_list = std.ArrayList([]const u8).empty;
    defer args_list.deinit(allocator);
    while (args_iter.next()) |arg| {
        args_list.append(allocator, arg) catch return error.GensfenError;
    }

    const opts = gensfen.parseArgs(args_list.items) catch {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "Usage: sykora gensfen --output FILE --games N --depth D [--random-plies N] [--seed N] [--sample-pct N]\n", .{}) catch "";
        std.fs.File.stderr().writeAll(msg) catch {};
        return error.GensfenError;
    };

    gensfen.run(opts, allocator) catch {
        return error.GensfenError;
    };
}
