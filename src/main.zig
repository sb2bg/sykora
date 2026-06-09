const std = @import("std");
const uci = @import("interface.zig");
const Uci = uci.Uci;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const gensfen = @import("gensfen.zig");
const nnue_check = @import("nnue_check.zig");

pub fn main() void {
    tryMain() catch |err| {
        std.log.err("Encountered an unrecoverable error while running Sykora.\n", .{});
        switch (err) {
            error.GensfenError => {},
            error.NnueCheckError => std.process.exit(1),
            else => {
                std.log.err("\t|> {s}\n", .{uciErr.getErrorDescriptor(@errorCast(err))});
            },
        }
    };
}

const MainError = UciError || error{GensfenError} || nnue_check.NnueCheckError;

fn tryMain() MainError!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check for subcommands via CLI args
    // Use argsAlloc for cross-platform compatibility (Windows needs allocator).
    const args = std.process.argsAlloc(allocator) catch return error.GensfenError;
    defer std.process.argsFree(allocator, args);

    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "gensfen")) {
            return runGensfenFromSlice(args[2..], allocator);
        }
        if (std.mem.eql(u8, args[1], "nnuecheck")) {
            return runNnueCheckFromSlice(args[2..], allocator);
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

fn runGensfenFromSlice(args: []const [:0]u8, allocator: std.mem.Allocator) MainError!void {
    // Convert [:0]u8 slices to []const u8 slices for parseArgs
    var arg_slices = std.ArrayList([]const u8).empty;
    defer arg_slices.deinit(allocator);
    for (args) |arg| {
        arg_slices.append(allocator, arg) catch return error.GensfenError;
    }

    const opts = gensfen.parseArgs(arg_slices.items) catch {
        std.log.err("Usage: sykora gensfen --output FILE --games N --depth D [--random-plies N] [--seed N] [--sample-pct N]\n", .{});
        return error.GensfenError;
    };

    gensfen.run(opts, allocator) catch {
        return error.GensfenError;
    };
}

fn runNnueCheckFromSlice(args: []const [:0]u8, allocator: std.mem.Allocator) MainError!void {
    var arg_slices = std.ArrayList([]const u8).empty;
    defer arg_slices.deinit(allocator);
    for (args) |arg| {
        arg_slices.append(allocator, arg) catch return error.NnueCheckError;
    }

    return nnue_check.run(arg_slices.items, allocator);
}
