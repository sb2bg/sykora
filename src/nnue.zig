const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const EMBEDDED_NET = @embedFile("net.sknnue");

pub const INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 4096;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

const MAGIC_V3 = "SYKNNUE3";
const MAGIC_V2 = "SYKNNUE2";
const FORMAT_VERSION: u16 = 3;

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
/// V3 (two layers):
///   magic "SYKNNUE3", version 3, hidden_size, l2_size, activation_type, output_bias,
///   biases[hidden_size], input_weights[768*hidden_size],
///   l2_weights[2*hidden_size*l2_size], l2_biases[l2_size], output_weights[l2_size]
pub const Network = struct {
    allocator: std.mem.Allocator,
    hidden_size: u16,
    l2_size: u16, // 0 = single layer (V2 compat)
    activation_type: u8, // 0 = ReLU, 1 = SCReLU
    biases: []i16,
    input_weights: []i16,
    l2_weights: ?[]i16, // L1 concat -> L2, shape [2 * hidden_size * l2_size]
    l2_biases: ?[]i16, // L2 biases, shape [l2_size]
    output_weights: []i16, // shape [2*hidden_size] if l2_size==0, [l2_size] if l2_size>0
    output_bias: i32,

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.biases);
        self.allocator.free(self.input_weights);
        if (self.l2_weights) |w| self.allocator.free(w);
        if (self.l2_biases) |b| self.allocator.free(b);
        self.allocator.free(self.output_weights);
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
        if (data.len < 8) return error.InvalidNetwork;

        const is_v3 = std.mem.eql(u8, data[0..8], MAGIC_V3);
        const is_v2 = std.mem.eql(u8, data[0..8], MAGIC_V2);
        if (!is_v3 and !is_v2) return error.InvalidNetwork;

        var pos: usize = 8;

        const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
        if (is_v3 and version != 3) return error.UnsupportedVersion;
        if (is_v2 and version != 2) return error.UnsupportedVersion;

        const hidden_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
        const hidden_size: usize = @intCast(hidden_size_u16);
        if (hidden_size == 0) return error.InvalidNetwork;
        if (hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

        // V3 has l2_size after hidden_size
        const l2_size_u16: u16 = if (is_v3)
            readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork
        else
            0;
        const l2_size: usize = @intCast(l2_size_u16);

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

        // L2 weights/biases (V3 only)
        var l2_weights: ?[]i16 = null;
        var l2_biases: ?[]i16 = null;
        if (l2_size > 0) {
            const l2w = try allocator.alloc(i16, 2 * hidden_size * l2_size);
            errdefer allocator.free(l2w);
            for (l2w) |*w| {
                w.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
            }
            l2_weights = l2w;

            const l2b = try allocator.alloc(i16, l2_size);
            errdefer allocator.free(l2b);
            for (l2b) |*v| {
                v.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
            }
            l2_biases = l2b;
        }

        const output_len: usize = if (l2_size > 0) l2_size else 2 * hidden_size;
        const output_weights = try allocator.alloc(i16, output_len);
        errdefer allocator.free(output_weights);
        for (output_weights) |*w| {
            w.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
        }

        return Network{
            .allocator = allocator,
            .hidden_size = hidden_size_u16,
            .l2_size = l2_size_u16,
            .activation_type = activation_type,
            .biases = biases,
            .input_weights = input_weights,
            .l2_weights = l2_weights,
            .l2_biases = l2_biases,
            .output_weights = output_weights,
            .output_bias = output_bias,
        };
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

        const is_v3 = std.mem.eql(u8, magic_buf[0..], MAGIC_V3);
        const is_v2 = std.mem.eql(u8, magic_buf[0..], MAGIC_V2);
        if (!is_v3 and !is_v2) {
            return error.InvalidNetwork;
        }

        const version = try readInt(u16, &reader.interface);
        if (is_v3 and version != 3) return error.UnsupportedVersion;
        if (is_v2 and version != 2) return error.UnsupportedVersion;

        const hidden_size_u16 = try readInt(u16, &reader.interface);
        const hidden_size: usize = @intCast(hidden_size_u16);
        if (hidden_size == 0) return error.InvalidNetwork;
        if (hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

        // V3 has l2_size after hidden_size
        const l2_size_u16: u16 = if (is_v3)
            try readInt(u16, &reader.interface)
        else
            0;
        const l2_size: usize = @intCast(l2_size_u16);

        const activation_type: u8 = try readInt(u8, &reader.interface);

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

        // L2 weights/biases (V3 only)
        var l2_weights: ?[]i16 = null;
        var l2_biases: ?[]i16 = null;
        if (l2_size > 0) {
            const l2w = try allocator.alloc(i16, 2 * hidden_size * l2_size);
            errdefer allocator.free(l2w);
            for (l2w) |*w| {
                w.* = try readInt(i16, &reader.interface);
            }
            l2_weights = l2w;

            const l2b = try allocator.alloc(i16, l2_size);
            errdefer allocator.free(l2b);
            for (l2b) |*v| {
                v.* = try readInt(i16, &reader.interface);
            }
            l2_biases = l2b;
        }

        const output_len: usize = if (l2_size > 0) l2_size else 2 * hidden_size;
        const output_weights = try allocator.alloc(i16, output_len);
        errdefer allocator.free(output_weights);
        for (output_weights) |*w| {
            w.* = try readInt(i16, &reader.interface);
        }

        return Network{
            .allocator = allocator,
            .hidden_size = hidden_size_u16,
            .l2_size = l2_size_u16,
            .activation_type = activation_type,
            .biases = biases,
            .input_weights = input_weights,
            .l2_weights = l2_weights,
            .l2_biases = l2_biases,
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

fn readBytesInt(comptime T: type, data: []const u8, pos: *usize) ?T {
    const size = @sizeOf(T);
    if (pos.* + size > data.len) return null;
    const bytes = data[pos.*..][0..size];
    pos.* += size;
    return std.mem.readInt(T, bytes, .little);
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
    const l2_size: usize = @intCast(net.l2_size);
    const use_screlu = net.activation_type == 1;

    if (l2_size > 0) {
        return evaluateWithL2(net, acc, stm_is_white, hidden_size, l2_size, use_screlu);
    }

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

/// Two-layer evaluation: L1 activations -> L2 forward -> output
fn evaluateWithL2(
    net: *const Network,
    acc: *const AccumulatorPair,
    stm_is_white: bool,
    hidden_size: usize,
    l2_size: usize,
    use_screlu: bool,
) i32 {
    const l2_weights = net.l2_weights.?;
    const l2_biases = net.l2_biases.?;

    // Compute L1 activations (SCReLU/ReLU on accumulator values)
    // Input to L2 is [stm_activations | nstm_activations], length = 2 * hidden_size
    // L2 weights are stored as [2 * hidden_size * l2_size] row-major:
    //   for each L2 neuron j: weights[j * 2*hidden_size .. (j+1) * 2*hidden_size]

    // Precompute L1 SCReLU activations: [stm | nstm], length 2 * hidden_size
    // For SCReLU: store clamped^2 (QA^2 scale). For ReLU: store clamped (QA scale).
    // Bullet stores affine(2*hl_size, l2_size) weights as [input, output] = [2*hl_size, l2_size]
    var l1_acts: [2 * MAX_HIDDEN_SIZE]i32 = undefined;
    for (0..hidden_size) |h| {
        const clamped = clampToQa(if (stm_is_white) acc.white[h] else acc.black[h]);
        l1_acts[h] = if (use_screlu) clamped * clamped else clamped;
    }
    for (0..hidden_size) |h| {
        const clamped = clampToQa(if (stm_is_white) acc.black[h] else acc.white[h]);
        l1_acts[hidden_size + h] = if (use_screlu) clamped * clamped else clamped;
    }

    // L2 matmul
    var l2_acc: [256]i32 = undefined;
    const total_inputs = 2 * hidden_size;
    for (0..l2_size) |j| {
        var dot: i64 = 0;
        for (0..total_inputs) |i| {
            dot += @as(i64, l1_acts[i]) * @as(i64, l2_weights[i * l2_size + j]);
        }
        // SCReLU: dot at QA^2 * QB, /QA -> QA*QB. Add bias (QA) * QB. /QB -> QA.
        if (use_screlu) {
            dot = @divTrunc(dot, QA);
        }
        dot += @as(i64, l2_biases[j]) * QB;
        l2_acc[j] = @intCast(@divTrunc(dot, QB));
    }

    // L2 SCReLU + output layer (same pattern as single-layer)
    // l2_acc is at QA scale. Clamp to [0, QA], square -> QA^2.
    // Output weights at QB -> QA^2 * QB per term.
    var sum: i64 = 0;
    for (0..l2_size) |j| {
        const activated = clampToQa(l2_acc[j]);
        if (use_screlu) {
            sum += @as(i64, activated) * @as(i64, activated) * @as(i64, net.output_weights[j]);
        } else {
            sum += @as(i64, activated) * @as(i64, net.output_weights[j]);
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
