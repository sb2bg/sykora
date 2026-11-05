const std = @import("std");
const uci = @import("interface.zig");
const Uci = uci.Uci;
const uciErr = @import("uci_error.zig");
const UciError = uciErr.UciError;
const gensfen = @import("gensfen.zig");
const nnue_check = @import("nnue_check.zig");
const bench = @import("bench.zig");
const tuning = @import("search/tuning.zig");

pub fn main() void {
    tryMain() catch |err| {
        std.log.err("Encountered an unrecoverable error while running Sykora.\n", .{});
        switch (err) {
            error.GensfenError => {},
            error.NnueCheckError => std.process.exit(1),
            error.BenchFailed => std.process.exit(1),
            else => {
                std.log.err("\t|> {s}\n", .{uciErr.getErrorDescriptor(@errorCast(err))});
            },
        }
    };
}

const MainError = UciError || error{GensfenError} || nnue_check.NnueCheckError || bench.BenchError;

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
        if (std.mem.eql(u8, args[1], "bench")) {
            // An optional depth gives longer, search-dominated runs for profiling.
            const depth = if (args.len > 2)
                std.fmt.parseInt(u64, args[2], 10) catch return error.BenchFailed
            else
                bench.DEFAULT_DEPTH;
            return bench.run(allocator, depth);
        }
        if (std.mem.eql(u8, args[1], "spsa")) {
            return printSpsaInput(args[2..]);
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

/// Prints OpenBench SPSA input for the named tunable groups (default: search).
fn printSpsaInput(args: []const [:0]u8) MainError!void {
    var groups: [@typeInfo(tuning.Group).@"enum".fields.len]tuning.Group = undefined;
    var count: usize = 0;
    for (args) |arg| {
        const group = std.meta.stringToEnum(tuning.Group, arg) orelse {
            std.log.err("Usage: sykora spsa [search] [time] [discrete] [mode]\n", .{});
            return error.InvalidArgument;
        };
        if (std.mem.indexOfScalar(tuning.Group, groups[0..count], group) == null) {
            groups[count] = group;
            count += 1;
        }
    }
    if (count == 0) {
        groups[0] = .search;
        count = 1;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    tuning.writeSpsaInput(&stdout_writer.interface, groups[0..count]) catch return error.IOError;
    stdout_writer.interface.flush() catch return error.IOError;
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
