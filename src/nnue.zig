const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const EMBEDDED_NET = @embedFile("net.sknnue");

pub const LEGACY_INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 512;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;
const MAX_NETWORK_BYTES = 8 * 1024 * 1024;

const MAGIC_V2 = "SYKNNUE2";
const MAGIC_V3 = "SYKNNUE3";
const FORMAT_VERSION_V2: u16 = 2;
const FORMAT_VERSION_V3: u16 = 3;

pub const FeatureSet = enum(u8) {
    legacy_psqt = 0,
    king_buckets_mirrored = 1,
};

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
    feature_set: FeatureSet,
    bucket_count: u16,
    bucket_layout: [64]u8,
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

        if (std.mem.eql(u8, data[0..8], MAGIC_V2)) {
            return loadFromBytesV2(allocator, data);
        }
        if (std.mem.eql(u8, data[0..8], MAGIC_V3)) {
            return loadFromBytesV3(allocator, data);
        }
        return error.InvalidNetwork;
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

    pub fn inputSize(self: *const Network) usize {
        return switch (self.feature_set) {
            .legacy_psqt => LEGACY_INPUT_SIZE,
            .king_buckets_mirrored => LEGACY_INPUT_SIZE * @as(usize, @intCast(self.bucket_count)),
        };
    }
};

fn loadFromBytesV2(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
    var pos: usize = 8;

    const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (version != FORMAT_VERSION_V2) return error.UnsupportedVersion;

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

    const input_len = LEGACY_INPUT_SIZE * hidden_size;
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
        .feature_set = .legacy_psqt,
        .bucket_count = 1,
        .bucket_layout = [_]u8{0} ** 64,
        .biases = biases,
        .input_weights = input_weights,
        .output_weights = output_weights,
        .output_bias = output_bias,
    };
}

fn loadFromBytesV3(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
    var pos: usize = 8;

    const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (version != FORMAT_VERSION_V3) return error.UnsupportedVersion;

    if (pos >= data.len) return error.InvalidNetwork;
    const feature_set = std.meta.intToEnum(FeatureSet, data[pos]) catch return error.InvalidNetwork;
    pos += 1;

    const hidden_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const hidden_size: usize = @intCast(hidden_size_u16);
    if (hidden_size == 0) return error.InvalidNetwork;
    if (hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

    if (pos >= data.len) return error.InvalidNetwork;
    const activation_type: u8 = data[pos];
    pos += 1;

    const bucket_count = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (feature_set == .legacy_psqt and bucket_count != 1) return error.InvalidNetwork;
    if (feature_set == .king_buckets_mirrored and bucket_count == 0) return error.InvalidNetwork;

    var bucket_layout = [_]u8{0} ** 64;
    for (&bucket_layout) |*entry| {
        if (pos >= data.len) return error.InvalidNetwork;
        entry.* = data[pos];
        pos += 1;
    }
    if (feature_set == .king_buckets_mirrored) {
        for (bucket_layout) |entry| {
            if (entry >= bucket_count) return error.InvalidNetwork;
        }
    }

    const output_bias = readBytesInt(i32, data, &pos) orelse return error.InvalidNetwork;

    const biases = try allocator.alloc(i16, hidden_size);
    errdefer allocator.free(biases);
    for (biases) |*v| {
        v.* = readBytesInt(i16, data, &pos) orelse return error.InvalidNetwork;
    }

    const input_size = switch (feature_set) {
        .legacy_psqt => LEGACY_INPUT_SIZE,
        .king_buckets_mirrored => LEGACY_INPUT_SIZE * @as(usize, @intCast(bucket_count)),
    };
    const input_len = input_size * hidden_size;
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
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .biases = biases,
        .input_weights = input_weights,
        .output_weights = output_weights,
        .output_bias = output_bias,
    };
}

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
    net: *const Network,
    perspective: piece.Color,
    square: u8,
    piece_type: piece.Type,
    color: piece.Color,
    perspective_king_sq: u8,
) usize {
    var sq = if (perspective == .white) square else flipVertical(square);
    const king_sq = if (perspective == .white) perspective_king_sq else flipVertical(perspective_king_sq);
    const side = if (perspective == .white) color else oppositeColor(color);
    const side_idx: usize = @intFromEnum(side);
    const piece_idx: usize = @intFromEnum(piece_type);

    if (net.feature_set == .legacy_psqt) {
        return side_idx * 6 * 64 + piece_idx * 64 + sq;
    }

    if ((king_sq % 8) > 3) {
        sq ^= 7;
    }
    const bucket_offset = LEGACY_INPUT_SIZE * @as(usize, net.bucket_layout[king_sq]);
    return bucket_offset + side_idx * 6 * 64 + piece_idx * 64 + sq;
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

const SimdWeightVec = @Vector(8, i16);
const SimdAccVec = @Vector(8, i32);
const SIMD_LANES = @typeInfo(SimdAccVec).vector.len;
const OutputAccVec = @Vector(4, i32);
const OutputWeightVec = @Vector(4, i16);
const OutputWideVec = @Vector(4, i64);
const OUTPUT_SIMD_LANES = @typeInfo(OutputAccVec).vector.len;

inline fn clampOutputVector(v: OutputAccVec) OutputAccVec {
    const zero: OutputAccVec = @splat(0);
    const qa: OutputAccVec = @splat(QA);
    return @min(@max(v, zero), qa);
}

inline fn outputChunkDot(
    comptime use_screlu: bool,
    values: []const i32,
    weights: []const i16,
    start: usize,
) i64 {
    const value_ptr: *align(1) const OutputAccVec = @ptrCast(&values[start]);
    const weight_ptr: *align(1) const OutputWeightVec = @ptrCast(&weights[start]);
    const clamped = clampOutputVector(value_ptr.*);
    const weight_i32: OutputAccVec = @intCast(weight_ptr.*);
    const activated: OutputWideVec = if (use_screlu) blk: {
        const widened: OutputWideVec = @intCast(clamped);
        break :blk widened * widened;
    } else @intCast(clamped);
    const weight_i64: OutputWideVec = @intCast(weight_i32);
    return @reduce(.Add, activated * weight_i64);
}

inline fn perspectiveMirrored(king_sq: u8) bool {
    return (king_sq % 8) > 3;
}

inline fn perspectiveKingSquare(perspective: piece.Color, king_sq: u8) u8 {
    return if (perspective == .white) king_sq else flipVertical(king_sq);
}

inline fn perspectiveLayoutChanged(
    net: *const Network,
    perspective: piece.Color,
    old_king_sq: u8,
    new_king_sq: u8,
) bool {
    if (net.feature_set != .king_buckets_mirrored) return false;
    const old_perspective_king_sq = perspectiveKingSquare(perspective, old_king_sq);
    const new_perspective_king_sq = perspectiveKingSquare(perspective, new_king_sq);
    return net.bucket_layout[old_perspective_king_sq] != net.bucket_layout[new_perspective_king_sq] or
        perspectiveMirrored(old_perspective_king_sq) != perspectiveMirrored(new_perspective_king_sq);
}

inline fn initAccumulatorBiases(dest: []i32, biases: []const i16) void {
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const bias_ptr: *align(1) const SimdWeightVec = @ptrCast(&biases[h]);
        const dest_ptr: *align(1) SimdAccVec = @ptrCast(&dest[h]);
        dest_ptr.* = @intCast(bias_ptr.*);
    }

    while (h < dest.len) : (h += 1) {
        dest[h] = biases[h];
    }
}

inline fn applyFeatureSlice(comptime add: bool, dest: []i32, weights: []const i16) void {
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const weight_ptr: *align(1) const SimdWeightVec = @ptrCast(&weights[h]);
        const dest_ptr: *align(1) SimdAccVec = @ptrCast(&dest[h]);
        const weight_vec: SimdAccVec = @intCast(weight_ptr.*);
        dest_ptr.* = if (add) dest_ptr.* + weight_vec else dest_ptr.* - weight_vec;
    }

    while (h < dest.len) : (h += 1) {
        if (add) {
            dest[h] += weights[h];
        } else {
            dest[h] -= weights[h];
        }
    }
}

fn initPerspectiveAccumulator(
    net: *const Network,
    b: *Board,
    perspective: piece.Color,
    dest: []i32,
) void {
    const state = b.board;
    const hidden_size: usize = @intCast(net.hidden_size);
    const perspective_king_sq: u8 = switch (perspective) {
        .white => @intCast(@ctz(state.getColorBitboard(.white) & state.getKindBitboard(.king))),
        .black => @intCast(@ctz(state.getColorBitboard(.black) & state.getKindBitboard(.king))),
    };

    initAccumulatorBiases(dest, net.biases[0..hidden_size]);

    inline for ([_]piece.Color{ .white, .black }) |color| {
        const color_bb = state.getColorBitboard(color);

        inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king }) |pt| {
            var bb = color_bb & state.getKindBitboard(pt);
            while (bb != 0) {
                const sq: u8 = @intCast(@ctz(bb));
                bb &= bb - 1;

                const feature = featureIndex(net, perspective, sq, pt, color, perspective_king_sq);
                const base = feature * hidden_size;
                applyFeatureSlice(true, dest, net.input_weights[base .. base + hidden_size]);
            }
        }
    }
}

/// Full recompute of accumulators from board state (used at search root).
pub fn initAccumulators(net: *const Network, b: *Board) AccumulatorPair {
    var acc = AccumulatorPair{
        .white = [_]i32{0} ** MAX_HIDDEN_SIZE,
        .black = [_]i32{0} ** MAX_HIDDEN_SIZE,
    };

    initPerspectiveAccumulator(net, b, .white, acc.white[0..@intCast(net.hidden_size)]);
    initPerspectiveAccumulator(net, b, .black, acc.black[0..@intCast(net.hidden_size)]);

    return acc;
}

/// Apply a single feature delta (add or subtract) to an accumulator pair.
inline fn applyDelta(
    net: *const Network,
    acc: *AccumulatorPair,
    sq: u8,
    pt: piece.Type,
    color: piece.Color,
    white_king_sq: u8,
    black_king_sq: u8,
    add: bool,
) void {
    const hidden_size: usize = @intCast(net.hidden_size);
    const white_feature = featureIndex(net, .white, sq, pt, color, white_king_sq);
    const black_feature = featureIndex(net, .black, sq, pt, color, black_king_sq);
    const white_base = white_feature * hidden_size;
    const black_base = black_feature * hidden_size;

    if (add) {
        applyFeatureSlice(true, acc.white[0..hidden_size], net.input_weights[white_base .. white_base + hidden_size]);
        applyFeatureSlice(true, acc.black[0..hidden_size], net.input_weights[black_base .. black_base + hidden_size]);
    } else {
        applyFeatureSlice(false, acc.white[0..hidden_size], net.input_weights[white_base .. white_base + hidden_size]);
        applyFeatureSlice(false, acc.black[0..hidden_size], net.input_weights[black_base .. black_base + hidden_size]);
    }
}

/// Incremental accumulator update after a move.
/// Copies `prev` into `result`, then applies feature deltas.
pub fn updateAccumulators(
    net: *const Network,
    b: *Board,
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
    const state = b.board;
    const white_king_sq: u8 = @intCast(@ctz(state.getColorBitboard(.white) & state.getKindBitboard(.king)));
    const black_king_sq: u8 = @intCast(@ctz(state.getColorBitboard(.black) & state.getKindBitboard(.king)));

    // Copy previous accumulator
    @memcpy(result.white[0..hidden_size], prev.white[0..hidden_size]);
    @memcpy(result.black[0..hidden_size], prev.black[0..hidden_size]);

    // Remove piece from origin
    applyDelta(net, result, from_sq, moved_piece, moved_color, white_king_sq, black_king_sq, false);

    // Add piece at destination (or promoted piece)
    const final_piece = promotion orelse moved_piece;
    applyDelta(net, result, to_sq, final_piece, moved_color, white_king_sq, black_king_sq, true);

    // Remove captured piece if any
    if (captured_piece) |cp| {
        const opp = oppositeColor(moved_color);
        applyDelta(net, result, capture_sq.?, cp, opp, white_king_sq, black_king_sq, false);
    }

    // Castling: move the rook too
    if (is_castling) {
        if (rook_from) |rf| {
            applyDelta(net, result, rf, .rook, moved_color, white_king_sq, black_king_sq, false);
            applyDelta(net, result, rook_to.?, .rook, moved_color, white_king_sq, black_king_sq, true);
        }
    }

    if (net.feature_set == .king_buckets_mirrored and moved_piece == .king) {
        switch (moved_color) {
            .white => {
                if (perspectiveLayoutChanged(net, .white, from_sq, to_sq)) {
                    initPerspectiveAccumulator(net, b, .white, result.white[0..hidden_size]);
                }
            },
            .black => {
                if (perspectiveLayoutChanged(net, .black, from_sq, to_sq)) {
                    initPerspectiveAccumulator(net, b, .black, result.black[0..hidden_size]);
                }
            },
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
    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];
    const us_weights = net.output_weights[0..hidden_size];
    const them_weights = net.output_weights[hidden_size .. hidden_size + hidden_size];

    var sum: i64 = 0;
    var h: usize = 0;

    if (use_screlu) {
        while (h + OUTPUT_SIMD_LANES <= hidden_size) : (h += OUTPUT_SIMD_LANES) {
            sum += outputChunkDot(true, us_acc, us_weights, h);
            sum += outputChunkDot(true, them_acc, them_weights, h);
        }
    } else {
        while (h + OUTPUT_SIMD_LANES <= hidden_size) : (h += OUTPUT_SIMD_LANES) {
            sum += outputChunkDot(false, us_acc, us_weights, h);
            sum += outputChunkDot(false, them_acc, them_weights, h);
        }
    }

    while (h < hidden_size) : (h += 1) {
        const us = clampToQa(us_acc[h]);
        const them = clampToQa(them_acc[h]);

        if (use_screlu) {
            sum += @as(i64, us) * @as(i64, us) * @as(i64, us_weights[h]);
            sum += @as(i64, them) * @as(i64, them) * @as(i64, them_weights[h]);
        } else {
            sum += @as(i64, us) * @as(i64, us_weights[h]);
            sum += @as(i64, them) * @as(i64, them_weights[h]);
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
