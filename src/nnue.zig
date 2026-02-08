const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 4096;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

const MAGIC = "SYKNNUE1";
const FORMAT_VERSION: u16 = 1;

pub const LoadError = error{
    OutOfMemory,
    FileNotFound,
    AccessDenied,
    NotDir,
    IsDir,
    NameTooLong,
    IOError,
    InvalidNetwork,
    UnsupportedVersion,
    NetworkTooLarge,
};

/// Simple NNUE format used by Sykora (little-endian):
/// - 8 bytes  magic: "SYKNNUE1"
/// - u16      version: 1
/// - u16      hidden_size
/// - i32      output_bias
/// - i16[hidden_size]                accumulator biases
/// - i16[INPUT_SIZE * hidden_size]   input -> accumulator weights
/// - i16[2 * hidden_size]            output weights (stm half, nstm half)
pub const Network = struct {
    allocator: std.mem.Allocator,
    hidden_size: u16,
    biases: []i16,
    input_weights: []i16,
    output_weights: []i16,
    output_bias: i32,

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.biases);
        self.allocator.free(self.input_weights);
        self.allocator.free(self.output_weights);
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) LoadError!Network {
        var file = std.fs.cwd().openFile(path, .{}) catch |err| {
            return mapOpenError(err);
        };
        defer file.close();

        var file_buf: [4096]u8 = undefined;
        var reader = file.reader(&file_buf);
        var magic_buf: [8]u8 = undefined;
        reader.interface.readSliceAll(&magic_buf) catch |err| {
            return mapReadError(err);
        };

        if (!std.mem.eql(u8, magic_buf[0..], MAGIC)) {
            return error.InvalidNetwork;
        }

        const version = try readInt(u16, &reader.interface);
        if (version != FORMAT_VERSION) {
            return error.UnsupportedVersion;
        }

        const hidden_size_u16 = try readInt(u16, &reader.interface);
        const hidden_size: usize = @intCast(hidden_size_u16);
        if (hidden_size == 0) return error.InvalidNetwork;
        if (hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

        const output_bias = try readInt(i32, &reader.interface);

        const biases = try allocator.alloc(i16, hidden_size);
        errdefer allocator.free(biases);
        for (biases) |*v| {
            v.* = try readInt(i16, &reader.interface);
        }

        const input_len = INPUT_SIZE * hidden_size;
        const input_weights = try allocator.alloc(i16, input_len);
        errdefer allocator.free(input_weights);
        for (input_weights) |*w| {
            w.* = try readInt(i16, &reader.interface);
        }

        const output_len = 2 * hidden_size;
        const output_weights = try allocator.alloc(i16, output_len);
        errdefer allocator.free(output_weights);
        for (output_weights) |*w| {
            w.* = try readInt(i16, &reader.interface);
        }

        return Network{
            .allocator = allocator,
            .hidden_size = hidden_size_u16,
            .biases = biases,
            .input_weights = input_weights,
            .output_weights = output_weights,
            .output_bias = output_bias,
        };
    }
};

fn mapOpenError(err: anyerror) LoadError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.FileNotFound => error.FileNotFound,
        error.AccessDenied => error.AccessDenied,
        error.NotDir => error.NotDir,
        error.IsDir => error.IsDir,
        error.NameTooLong => error.NameTooLong,
        else => error.IOError,
    };
}

fn mapReadError(err: anyerror) LoadError {
    return switch (err) {
        error.EndOfStream => error.InvalidNetwork,
        else => error.IOError,
    };
}

fn readInt(comptime T: type, reader: *std.Io.Reader) LoadError!T {
    return reader.takeInt(T, .little) catch |err| {
        return mapReadError(err);
    };
}

inline fn flipVertical(square: u8) u8 {
    return square ^ 56;
}

inline fn oppositeColor(color: piece.Color) piece.Color {
    return if (color == .white) .black else .white;
}

fn featureIndex(
    perspective: piece.Color,
    square: u8,
    piece_type: piece.Type,
    color: piece.Color,
) usize {
    const sq = if (perspective == .white) square else flipVertical(square);
    const side = if (perspective == .white) color else oppositeColor(color);
    const side_idx: usize = @intFromEnum(side);
    const piece_idx: usize = @intFromEnum(piece_type);

    return side_idx * 6 * 64 + piece_idx * 64 + sq;
}

inline fn clampToQa(v: i32) i32 {
    if (v <= 0) return 0;
    if (v >= QA) return QA;
    return v;
}

/// Returns score from the side-to-move perspective, same convention as classical eval.
pub fn evaluate(net: *const Network, b: *Board, use_screlu: bool) i32 {
    const state = b.board;
    const hidden_size: usize = @intCast(net.hidden_size);

    var white_acc: [MAX_HIDDEN_SIZE]i32 = [_]i32{0} ** MAX_HIDDEN_SIZE;
    var black_acc: [MAX_HIDDEN_SIZE]i32 = [_]i32{0} ** MAX_HIDDEN_SIZE;

    for (0..hidden_size) |h| {
        const bias = net.biases[h];
        white_acc[h] = bias;
        black_acc[h] = bias;
    }

    inline for ([_]piece.Color{ .white, .black }) |color| {
        const color_bb = state.getColorBitboard(color);

        inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king }) |pt| {
            var bb = color_bb & state.getKindBitboard(pt);
            while (bb != 0) {
                const sq: u8 = @intCast(@ctz(bb));
                bb &= bb - 1;

                const white_feature = featureIndex(.white, sq, pt, color);
                const black_feature = featureIndex(.black, sq, pt, color);

                const white_base = white_feature * hidden_size;
                const black_base = black_feature * hidden_size;
                for (0..hidden_size) |h| {
                    white_acc[h] += net.input_weights[white_base + h];
                    black_acc[h] += net.input_weights[black_base + h];
                }
            }
        }
    }

    const stm_is_white = state.move == .white;
    var sum: i64 = 0;

    for (0..hidden_size) |h| {
        const us_raw = if (stm_is_white) white_acc[h] else black_acc[h];
        const them_raw = if (stm_is_white) black_acc[h] else white_acc[h];

        const us = clampToQa(us_raw);
        const them = clampToQa(them_raw);

        if (use_screlu) {
            // SCReLU(x) = clamp(x, 0, QA)^2. Matches Bullet simple/progression nets.
            sum += @as(i64, us) * @as(i64, us) * @as(i64, net.output_weights[h]);
            sum += @as(i64, them) * @as(i64, them) * @as(i64, net.output_weights[hidden_size + h]);
        } else {
            // CReLU(x) = clamp(x, 0, QA). Legacy SYK nets used this.
            sum += @as(i64, us) * @as(i64, net.output_weights[h]);
            sum += @as(i64, them) * @as(i64, net.output_weights[hidden_size + h]);
        }
    }

    if (use_screlu) {
        // SCReLU accumulates an extra QA factor relative to CReLU.
        sum = @divTrunc(sum, QA);
    }
    sum += net.output_bias;

    const score = @divTrunc(sum * SCALE, QA * QB);
    return @intCast(score);
}
