/// sykora-tune: standalone batch evaluator for Texel tuning.
///
/// Usage:
///   ./zig-out/bin/sykora-tune [--params <file>]
///
/// Reads FEN strings from stdin (one per line), evaluates each position
/// from white's perspective using the HCE, and prints one integer score
/// per line to stdout.
const std = @import("std");
const Board = @import("bitboard.zig").Board;
const eval = @import("evaluation.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse --params <file> argument
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--params") and i + 1 < args.len) {
            i += 1;
            try eval.loadParams(args[i], allocator);
        }
    }

    const stdin = std.fs.File.stdin();
    const stdout = std.fs.File.stdout();

    var line_buf = std.ArrayListUnmanaged(u8){};
    defer line_buf.deinit(allocator);

    var score_buf: [32]u8 = undefined;

    // Read FEN strings line by line, evaluate each one
    while (true) {
        line_buf.clearRetainingCapacity();

        // Read one line byte by byte (mirrors interface.zig approach)
        while (true) {
            var byte: [1]u8 = undefined;
            const n = stdin.read(&byte) catch break;
            if (n == 0) {
                // EOF — process any partial line then exit
                if (line_buf.items.len == 0) return;
                break;
            }
            if (byte[0] == '\n') break;
            if (byte[0] == '\r') continue;
            line_buf.append(allocator, byte[0]) catch continue;
        }

        const fen = std.mem.trim(u8, line_buf.items, " \t");
        if (fen.len == 0) {
            if (line_buf.capacity == 0) return; // clean EOF on empty input
            continue;
        }

        var b = Board.fromFen(fen) catch continue;
        const score = eval.evaluateWhite(&b);

        const s = std.fmt.bufPrint(&score_buf, "{d}\n", .{score}) catch continue;
        stdout.writeAll(s) catch return;
    }
}
