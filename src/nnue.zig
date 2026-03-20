const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const EMBEDDED_NET = @embedFile("net.sknnue");

pub const INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 512;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;
const MAX_NETWORK_BYTES = 2 * 1024 * 1024;

const MAGIC_V2 = "SYKNNUE2";
const FORMAT_VERSION: u16 = 2;

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

/// NNUE format used by Sykora (little-endian).
///
/// V2 (single layer):
///   magic "SYKNNUE2", version 2, hidden_size, activation_type, output_bias,
///   biases[hidden_size], input_weights[768*hidden_size], output_weights[2*hidden_size]
///
pub const Network = struct {
    allocator: std.mem.Allocator,
    hidden_size: u16,
    activation_type: u8, // 0 = ReLU, 1 = SCReLU
    biases: []i16,
    input_weights: []i16,
    output_weights: []i16, // shape [2 * hidden_size]
    output_bias: i32,

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.biases);
        self.allocator.free(self.input_weights);
        self.allocator.free(self.output_weights);
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
        if (data.len < 8) return error.InvalidNetwork;

        if (!std.mem.eql(u8, data[0..8], MAGIC_V2)) return error.InvalidNetwork;

        var pos: usize = 8;

        const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
        if (version != FORMAT_VERSION) return error.UnsupportedVersion;

        const hidden_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
        const hidden_size: usize = @intCast(hidden_size_u16);
        if (hidden_size == 0) return error.InvalidNetwork;
        if (hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

        if (pos >= data.len) return error.InvalidNetwork;
        const activation_type: u8 = data[pos];
        pos += 1;

        const output_bias = readBytesInt(i32, data, &pos) orelse return error.InvalidNetwork;

        const biases = try allocator.alloc(i16, hidden_size);
        errdefer allocator.free(biases);
        for (biases) |*v| {
            v.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
        }

        const input_len = INPUT_SIZE * hidden_size;
        const input_weights = try allocator.alloc(i16, input_len);
        errdefer allocator.free(input_weights);
        for (input_weights) |*w| {
            w.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
        }

        const output_len = 2 * hidden_size;
        const output_weights = try allocator.alloc(i16, output_len);
        errdefer allocator.free(output_weights);
        for (output_weights) |*w| {
            w.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
        }

        return Network{
            .allocator = allocator,
            .hidden_size = hidden_size_u16,
            .activation_type = activation_type,
            .biases = biases,
            .input_weights = input_weights,
            .output_weights = output_weights,
            .output_bias = output_bias,
        };
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) LoadError!Network {
        var file = std.fs.cwd().openFile(path, .{}) catch |err| {
            return mapOpenError(err);
        };
        defer file.close();

        const stat = file.stat() catch return error.IOError;
        if (stat.size > MAX_NETWORK_BYTES) return error.NetworkTooLarge;

        const data = file.readToEndAlloc(allocator, MAX_NETWORK_BYTES) catch |err| {
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.IOError,
            };
        };
        defer allocator.free(data);
        return loadFromBytes(allocator, data);
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

fn readBytesInt(comptime T: type, data: []const u8, pos: *usize) ?T {
    const size = @sizeOf(T);
    if (pos.* + size > data.len) return null;
    const bytes = data[pos.*..][0..size];
    pos.* += size;
    return std.mem.readInt(T, bytes, .little);
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

// ─── Incremental Accumulator Infrastructure ───

pub const AccumulatorPair = struct {
    white: [MAX_HIDDEN_SIZE]i32,
    black: [MAX_HIDDEN_SIZE]i32,
};

/// Full recompute of accumulators from board state (used at search root).
pub fn initAccumulators(net: *const Network, b: *Board) AccumulatorPair {
    const state = b.board;
    const hidden_size: usize = @intCast(net.hidden_size);

    var acc = AccumulatorPair{
        .white = [_]i32{0} ** MAX_HIDDEN_SIZE,
        .black = [_]i32{0} ** MAX_HIDDEN_SIZE,
    };

    for (0..hidden_size) |h| {
        const bias = net.biases[h];
        acc.white[h] = bias;
        acc.black[h] = bias;
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
                    acc.white[h] += net.input_weights[white_base + h];
                    acc.black[h] += net.input_weights[black_base + h];
                }
            }
        }
    }

    return acc;
}

/// Apply a single feature delta (add or subtract) to an accumulator pair.
inline fn applyDelta(
    net: *const Network,
    acc: *AccumulatorPair,
    sq: u8,
    pt: piece.Type,
    color: piece.Color,
    add: bool,
) void {
    const hidden_size: usize = @intCast(net.hidden_size);
    const white_feature = featureIndex(.white, sq, pt, color);
    const black_feature = featureIndex(.black, sq, pt, color);
    const white_base = white_feature * hidden_size;
    const black_base = black_feature * hidden_size;

    if (add) {
        for (0..hidden_size) |h| {
            acc.white[h] += net.input_weights[white_base + h];
            acc.black[h] += net.input_weights[black_base + h];
        }
    } else {
        for (0..hidden_size) |h| {
            acc.white[h] -= net.input_weights[white_base + h];
            acc.black[h] -= net.input_weights[black_base + h];
        }
    }
}

/// Incremental accumulator update after a move.
/// Copies `prev` into `result`, then applies feature deltas.
pub fn updateAccumulators(
    net: *const Network,
    prev: *const AccumulatorPair,
    result: *AccumulatorPair,
    from_sq: u8,
    to_sq: u8,
    moved_piece: piece.Type,
    moved_color: piece.Color,
    captured_piece: ?piece.Type,
    capture_sq: ?u8,
    promotion: ?piece.Type,
    is_castling: bool,
    rook_from: ?u8,
    rook_to: ?u8,
) void {
    const hidden_size: usize = @intCast(net.hidden_size);

    // Copy previous accumulator
    @memcpy(result.white[0..hidden_size], prev.white[0..hidden_size]);
    @memcpy(result.black[0..hidden_size], prev.black[0..hidden_size]);

    // Remove piece from origin
    applyDelta(net, result, from_sq, moved_piece, moved_color, false);

    // Add piece at destination (or promoted piece)
    const final_piece = promotion orelse moved_piece;
    applyDelta(net, result, to_sq, final_piece, moved_color, true);

    // Remove captured piece if any
    if (captured_piece) |cp| {
        const opp = oppositeColor(moved_color);
        applyDelta(net, result, capture_sq.?, cp, opp, false);
    }

    // Castling: move the rook too
    if (is_castling) {
        if (rook_from) |rf| {
            applyDelta(net, result, rf, .rook, moved_color, false);
            applyDelta(net, result, rook_to.?, .rook, moved_color, true);
        }
    }
}

/// Evaluate using pre-computed accumulators (activation + output layer only).
/// Returns score from the side-to-move perspective.
pub fn evaluateFromAccumulators(
    net: *const Network,
    acc: *const AccumulatorPair,
    stm_is_white: bool,
) i32 {
    const hidden_size: usize = @intCast(net.hidden_size);
    const use_screlu = net.activation_type == 1;

    var sum: i64 = 0;

    for (0..hidden_size) |h| {
        const us_raw = if (stm_is_white) acc.white[h] else acc.black[h];
        const them_raw = if (stm_is_white) acc.black[h] else acc.white[h];

        const us = clampToQa(us_raw);
        const them = clampToQa(them_raw);

        if (use_screlu) {
            sum += @as(i64, us) * @as(i64, us) * @as(i64, net.output_weights[h]);
            sum += @as(i64, them) * @as(i64, them) * @as(i64, net.output_weights[hidden_size + h]);
        } else {
            sum += @as(i64, us) * @as(i64, net.output_weights[h]);
            sum += @as(i64, them) * @as(i64, net.output_weights[hidden_size + h]);
        }
    }

    if (use_screlu) {
        sum = @divTrunc(sum, QA);
    }
    sum += net.output_bias;

    const score = @divTrunc(sum * SCALE, QA * QB);
    return @intCast(score);
}

/// Returns score from the side-to-move perspective, same convention as classical eval.
/// This is the full-recompute path (non-incremental). Kept for fallback/gensfen use.
pub fn evaluate(net: *const Network, b: *Board) i32 {
    const acc = initAccumulators(net, b);
    const stm_is_white = b.board.move == .white;
    return evaluateFromAccumulators(net, &acc, stm_is_white);
}
