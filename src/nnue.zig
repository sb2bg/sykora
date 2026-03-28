const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const EMBEDDED_NET = @embedFile("net.sknnue");

pub const LEGACY_INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 2048;
pub const Q0: i32 = 255;
pub const Q: i32 = 64;
pub const SCALE: i32 = 400;
const MAX_NETWORK_BYTES = 64 * 1024 * 1024;

const MAGIC_V3 = "SYKNNUE3";
const MAGIC_V4 = "SYKNNUE4";
const FORMAT_VERSION_V3: u16 = 3;
const FORMAT_VERSION_V4: u16 = 4;

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
pub const Network = struct {
    pub const V3Head = struct {
        activation_type: u8, // 0 = ReLU, 1 = SCReLU
        output_weights: []i16, // [2 * H]
        output_bias: i32,
    };

    pub const V4Head = struct {
        activation_type: u8, // 0 = ReLU, 1 = SCReLU
        q0: u16,
        q: u16,
        scale: u16,
        output_weights: []i16, // [2 * H]
        output_bias: i32,
    };

    allocator: std.mem.Allocator,
    feature_set: FeatureSet,
    bucket_count: u8,
    bucket_layout: [64]u8,
    ft_hidden_size: u16,
    ft_biases: []i16,
    ft_weights: []i16,
    head: union(enum) {
        v3: V3Head,
        v4: V4Head,
    },

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.ft_biases);
        self.allocator.free(self.ft_weights);
        switch (self.head) {
            .v3 => |v3| {
                self.allocator.free(v3.output_weights);
            },
            .v4 => |v4| {
                self.allocator.free(v4.output_weights);
            },
        }
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
        if (data.len < 8) return error.InvalidNetwork;

        if (std.mem.eql(u8, data[0..8], MAGIC_V3)) {
            return loadFromBytesV3(allocator, data);
        }
        if (std.mem.eql(u8, data[0..8], MAGIC_V4)) {
            return loadFromBytesV4(allocator, data);
        }
        return error.UnsupportedVersion;
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
            .king_buckets_mirrored => LEGACY_INPUT_SIZE * @as(usize, self.bucket_count),
        };
    }
};

fn allocAndReadInts(
    comptime T: type,
    allocator: std.mem.Allocator,
    data: []const u8,
    pos: *usize,
    len: usize,
) LoadError![]T {
    const values = allocator.alloc(T, len) catch |err| {
        return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    errdefer allocator.free(values);

    for (values) |*value| {
        value.* = readBytesInt(T, data, pos) orelse return error.InvalidNetwork;
    }
    return values;
}

fn checkedMulU64(a: u64, b: u64) ?u64 {
    return std.math.mul(u64, a, b) catch null;
}

fn checkedAddU64(a: u64, b: u64) ?u64 {
    return std.math.add(u64, a, b) catch null;
}

fn computeV4PayloadBytes(
    input_size: usize,
    ft_hidden_size: usize,
) ?u64 {
    var total: u64 = 0;

    const ft_bias_bytes = checkedMulU64(@as(u64, @intCast(ft_hidden_size)), @sizeOf(i16)) orelse return null;
    total = checkedAddU64(total, ft_bias_bytes) orelse return null;

    const ft_weight_count = checkedMulU64(@as(u64, @intCast(input_size)), @as(u64, @intCast(ft_hidden_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(ft_weight_count, @sizeOf(i16)) orelse return null) orelse return null;

    total = checkedAddU64(total, @sizeOf(i32)) orelse return null;
    const out_weight_count = checkedMulU64(2, @as(u64, @intCast(ft_hidden_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(out_weight_count, @sizeOf(i16)) orelse return null) orelse return null;

    return total;
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
    if (hidden_size == 0 or hidden_size > MAX_HIDDEN_SIZE) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const activation_type = data[pos];
    pos += 1;
    if (activation_type > 1) return error.InvalidNetwork;

    const bucket_count_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (bucket_count_u16 == 0 or bucket_count_u16 > 255) return error.InvalidNetwork;
    if (feature_set == .legacy_psqt and bucket_count_u16 != 1) return error.InvalidNetwork;
    const bucket_count: u8 = @intCast(bucket_count_u16);

    var bucket_layout = [_]u8{0} ** 64;
    for (&bucket_layout) |*entry| {
        if (pos >= data.len) return error.InvalidNetwork;
        entry.* = data[pos];
        pos += 1;
        if (entry.* >= bucket_count) return error.InvalidNetwork;
    }

    const output_bias = readBytesInt(i32, data, &pos) orelse return error.InvalidNetwork;
    const ft_biases = try allocAndReadInts(i16, allocator, data, &pos, hidden_size);
    errdefer allocator.free(ft_biases);

    const input_size = switch (feature_set) {
        .legacy_psqt => LEGACY_INPUT_SIZE,
        .king_buckets_mirrored => LEGACY_INPUT_SIZE * @as(usize, bucket_count),
    };
    const ft_weights = try allocAndReadInts(i16, allocator, data, &pos, input_size * hidden_size);
    errdefer allocator.free(ft_weights);

    const output_weights = try allocAndReadInts(i16, allocator, data, &pos, 2 * hidden_size);
    errdefer allocator.free(output_weights);

    if (pos != data.len) return error.InvalidNetwork;

    return Network{
        .allocator = allocator,
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .ft_hidden_size = hidden_size_u16,
        .ft_biases = ft_biases,
        .ft_weights = ft_weights,
        .head = .{
            .v3 = .{
                .activation_type = activation_type,
                .output_weights = output_weights,
                .output_bias = output_bias,
            },
        },
    };
}

fn loadFromBytesV4(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
    var pos: usize = 8;

    const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (version != FORMAT_VERSION_V4) return error.UnsupportedVersion;

    if (pos >= data.len) return error.InvalidNetwork;
    const feature_set = std.meta.intToEnum(FeatureSet, data[pos]) catch return error.InvalidNetwork;
    pos += 1;
    if (feature_set != .king_buckets_mirrored) return error.InvalidNetwork;

    const ft_hidden_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const ft_hidden_size: usize = @intCast(ft_hidden_size_u16);
    if (ft_hidden_size == 0) return error.InvalidNetwork;
    if (ft_hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

    if (pos >= data.len) return error.InvalidNetwork;
    const activation_type = data[pos];
    pos += 1;
    if (activation_type > 1) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const bucket_count = data[pos];
    pos += 1;
    if (bucket_count == 0) return error.InvalidNetwork;

    const q0 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const q = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const scale = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (q0 == 0 or q == 0 or scale == 0) return error.InvalidNetwork;

    var bucket_layout = [_]u8{0} ** 64;
    for (&bucket_layout) |*entry| {
        if (pos >= data.len) return error.InvalidNetwork;
        entry.* = data[pos];
        pos += 1;
        if (entry.* >= bucket_count) return error.InvalidNetwork;
    }

    const input_size = LEGACY_INPUT_SIZE * @as(usize, bucket_count);
    const payload_size = computeV4PayloadBytes(
        input_size,
        ft_hidden_size,
    ) orelse return error.InvalidNetwork;
    const expected_size = checkedAddU64(@as(u64, @intCast(pos)), payload_size) orelse return error.InvalidNetwork;
    if (expected_size != data.len) return error.InvalidNetwork;

    const output_bias = readBytesInt(i32, data, &pos) orelse return error.InvalidNetwork;
    const ft_biases = try allocAndReadInts(i16, allocator, data, &pos, ft_hidden_size);
    errdefer allocator.free(ft_biases);

    const ft_weights = try allocAndReadInts(i16, allocator, data, &pos, input_size * ft_hidden_size);
    errdefer allocator.free(ft_weights);

    const output_weights = try allocAndReadInts(i16, allocator, data, &pos, 2 * ft_hidden_size);
    errdefer allocator.free(output_weights);

    if (pos != data.len) return error.InvalidNetwork;

    return Network{
        .allocator = allocator,
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .ft_hidden_size = ft_hidden_size_u16,
        .ft_biases = ft_biases,
        .ft_weights = ft_weights,
        .head = .{
            .v4 = .{
                .activation_type = activation_type,
                .q0 = q0,
                .q = q,
                .scale = scale,
                .output_weights = output_weights,
                .output_bias = output_bias,
            },
        },
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

inline fn clampToActivationRange(v: i32, max_value: i32) i32 {
    if (v <= 0) return 0;
    if (v >= max_value) return max_value;
    return v;
}

inline fn divRoundNearestNonNeg(x: i64, d: i64) i64 {
    return @divTrunc(x + @divTrunc(d, 2), d);
}

inline fn divRoundNearestSigned(x: i64, d: i64) i64 {
    return if (x >= 0)
        @divTrunc(x + @divTrunc(d, 2), d)
    else
        -@divTrunc((-x) + @divTrunc(d, 2), d);
}

// ─── Incremental Accumulator Infrastructure ───

pub const AccumulatorPair = struct {
    white: [MAX_HIDDEN_SIZE]i32,
    black: [MAX_HIDDEN_SIZE]i32,
};

const SimdWeightVec = @Vector(8, i16);
const SimdAccVec = @Vector(8, i32);
const SIMD_LANES = @typeInfo(SimdAccVec).vector.len;

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

inline fn applyFeatureSlicesAddSub(dest: []i32, add_weights: []const i16, sub_weights: []const i16) void {
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const add_ptr: *align(1) const SimdWeightVec = @ptrCast(&add_weights[h]);
        const sub_ptr: *align(1) const SimdWeightVec = @ptrCast(&sub_weights[h]);
        const dest_ptr: *align(1) SimdAccVec = @ptrCast(&dest[h]);
        dest_ptr.* = dest_ptr.* + @as(SimdAccVec, @intCast(add_ptr.*)) - @as(SimdAccVec, @intCast(sub_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] += add_weights[h];
        dest[h] -= sub_weights[h];
    }
}

inline fn applyFeatureSlicesAddSubSub(
    dest: []i32,
    add_weights: []const i16,
    sub_a_weights: []const i16,
    sub_b_weights: []const i16,
) void {
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const add_ptr: *align(1) const SimdWeightVec = @ptrCast(&add_weights[h]);
        const sub_a_ptr: *align(1) const SimdWeightVec = @ptrCast(&sub_a_weights[h]);
        const sub_b_ptr: *align(1) const SimdWeightVec = @ptrCast(&sub_b_weights[h]);
        const dest_ptr: *align(1) SimdAccVec = @ptrCast(&dest[h]);
        dest_ptr.* = dest_ptr.* + @as(SimdAccVec, @intCast(add_ptr.*)) - @as(SimdAccVec, @intCast(sub_a_ptr.*)) - @as(SimdAccVec, @intCast(sub_b_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] += add_weights[h];
        dest[h] -= sub_a_weights[h];
        dest[h] -= sub_b_weights[h];
    }
}

inline fn applyFeatureSlicesAddAddSubSub(
    dest: []i32,
    add_a_weights: []const i16,
    add_b_weights: []const i16,
    sub_a_weights: []const i16,
    sub_b_weights: []const i16,
) void {
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const add_a_ptr: *align(1) const SimdWeightVec = @ptrCast(&add_a_weights[h]);
        const add_b_ptr: *align(1) const SimdWeightVec = @ptrCast(&add_b_weights[h]);
        const sub_a_ptr: *align(1) const SimdWeightVec = @ptrCast(&sub_a_weights[h]);
        const sub_b_ptr: *align(1) const SimdWeightVec = @ptrCast(&sub_b_weights[h]);
        const dest_ptr: *align(1) SimdAccVec = @ptrCast(&dest[h]);
        dest_ptr.* = dest_ptr.* + @as(SimdAccVec, @intCast(add_a_ptr.*)) + @as(SimdAccVec, @intCast(add_b_ptr.*)) - @as(SimdAccVec, @intCast(sub_a_ptr.*)) - @as(SimdAccVec, @intCast(sub_b_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] += add_a_weights[h];
        dest[h] += add_b_weights[h];
        dest[h] -= sub_a_weights[h];
        dest[h] -= sub_b_weights[h];
    }
}

fn initPerspectiveAccumulator(
    net: *const Network,
    b: *Board,
    perspective: piece.Color,
    dest: []i32,
) void {
    const state = b.board;
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const perspective_king_sq: u8 = switch (perspective) {
        .white => @intCast(@ctz(state.getColorBitboard(.white) & state.getKindBitboard(.king))),
        .black => @intCast(@ctz(state.getColorBitboard(.black) & state.getKindBitboard(.king))),
    };

    initAccumulatorBiases(dest, net.ft_biases[0..hidden_size]);

    inline for ([_]piece.Color{ .white, .black }) |color| {
        const color_bb = state.getColorBitboard(color);

        inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king }) |pt| {
            var bb = color_bb & state.getKindBitboard(pt);
            while (bb != 0) {
                const sq: u8 = @intCast(@ctz(bb));
                bb &= bb - 1;

                const feature = featureIndex(net, perspective, sq, pt, color, perspective_king_sq);
                const base = feature * hidden_size;
                applyFeatureSlice(true, dest, net.ft_weights[base .. base + hidden_size]);
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

    initPerspectiveAccumulator(net, b, .white, acc.white[0..@intCast(net.ft_hidden_size)]);
    initPerspectiveAccumulator(net, b, .black, acc.black[0..@intCast(net.ft_hidden_size)]);

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
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const white_feature = featureIndex(net, .white, sq, pt, color, white_king_sq);
    const black_feature = featureIndex(net, .black, sq, pt, color, black_king_sq);
    const white_base = white_feature * hidden_size;
    const black_base = black_feature * hidden_size;

    if (add) {
        applyFeatureSlice(true, acc.white[0..hidden_size], net.ft_weights[white_base .. white_base + hidden_size]);
        applyFeatureSlice(true, acc.black[0..hidden_size], net.ft_weights[black_base .. black_base + hidden_size]);
    } else {
        applyFeatureSlice(false, acc.white[0..hidden_size], net.ft_weights[white_base .. white_base + hidden_size]);
        applyFeatureSlice(false, acc.black[0..hidden_size], net.ft_weights[black_base .. black_base + hidden_size]);
    }
}

inline fn perspectiveAccumulatorSlice(
    acc: *AccumulatorPair,
    perspective: piece.Color,
    hidden_size: usize,
) []i32 {
    return switch (perspective) {
        .white => acc.white[0..hidden_size],
        .black => acc.black[0..hidden_size],
    };
}

inline fn perspectiveFeatureWeights(
    net: *const Network,
    perspective: piece.Color,
    king_sq: u8,
    sq: u8,
    pt: piece.Type,
    color: piece.Color,
    hidden_size: usize,
) []const i16 {
    const feature = featureIndex(net, perspective, sq, pt, color, king_sq);
    const base = feature * hidden_size;
    return net.ft_weights[base .. base + hidden_size];
}

inline fn applyPerspectiveAddSub(
    net: *const Network,
    acc: *AccumulatorPair,
    perspective: piece.Color,
    king_sq: u8,
    add_sq: u8,
    add_pt: piece.Type,
    add_color: piece.Color,
    sub_sq: u8,
    sub_pt: piece.Type,
    sub_color: piece.Color,
    hidden_size: usize,
) void {
    const dest = perspectiveAccumulatorSlice(acc, perspective, hidden_size);
    const add_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_sq, add_pt, add_color, hidden_size);
    const sub_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_sq, sub_pt, sub_color, hidden_size);
    applyFeatureSlicesAddSub(dest, add_weights, sub_weights);
}

inline fn applyPerspectiveAddSubSub(
    net: *const Network,
    acc: *AccumulatorPair,
    perspective: piece.Color,
    king_sq: u8,
    add_sq: u8,
    add_pt: piece.Type,
    add_color: piece.Color,
    sub_a_sq: u8,
    sub_a_pt: piece.Type,
    sub_a_color: piece.Color,
    sub_b_sq: u8,
    sub_b_pt: piece.Type,
    sub_b_color: piece.Color,
    hidden_size: usize,
) void {
    const dest = perspectiveAccumulatorSlice(acc, perspective, hidden_size);
    const add_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_sq, add_pt, add_color, hidden_size);
    const sub_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_a_sq, sub_a_pt, sub_a_color, hidden_size);
    const sub_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_b_sq, sub_b_pt, sub_b_color, hidden_size);
    applyFeatureSlicesAddSubSub(dest, add_weights, sub_a_weights, sub_b_weights);
}

inline fn applyPerspectiveAddAddSubSub(
    net: *const Network,
    acc: *AccumulatorPair,
    perspective: piece.Color,
    king_sq: u8,
    add_a_sq: u8,
    add_a_pt: piece.Type,
    add_a_color: piece.Color,
    add_b_sq: u8,
    add_b_pt: piece.Type,
    add_b_color: piece.Color,
    sub_a_sq: u8,
    sub_a_pt: piece.Type,
    sub_a_color: piece.Color,
    sub_b_sq: u8,
    sub_b_pt: piece.Type,
    sub_b_color: piece.Color,
    hidden_size: usize,
) void {
    const dest = perspectiveAccumulatorSlice(acc, perspective, hidden_size);
    const add_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_a_sq, add_a_pt, add_a_color, hidden_size);
    const add_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_b_sq, add_b_pt, add_b_color, hidden_size);
    const sub_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_a_sq, sub_a_pt, sub_a_color, hidden_size);
    const sub_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_b_sq, sub_b_pt, sub_b_color, hidden_size);
    applyFeatureSlicesAddAddSubSub(dest, add_a_weights, add_b_weights, sub_a_weights, sub_b_weights);
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
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const state = b.board;
    const white_king_sq: u8 = @intCast(@ctz(state.getColorBitboard(.white) & state.getKindBitboard(.king)));
    const black_king_sq: u8 = @intCast(@ctz(state.getColorBitboard(.black) & state.getKindBitboard(.king)));
    const refresh_white = net.feature_set == .king_buckets_mirrored and
        moved_piece == .king and
        moved_color == .white and
        perspectiveLayoutChanged(net, .white, from_sq, to_sq);
    const refresh_black = net.feature_set == .king_buckets_mirrored and
        moved_piece == .king and
        moved_color == .black and
        perspectiveLayoutChanged(net, .black, from_sq, to_sq);
    const final_piece = promotion orelse moved_piece;
    const opp_color = oppositeColor(moved_color);

    if (!refresh_white) {
        @memcpy(result.white[0..hidden_size], prev.white[0..hidden_size]);
    }
    if (!refresh_black) {
        @memcpy(result.black[0..hidden_size], prev.black[0..hidden_size]);
    }

    const applyPerspective = struct {
        inline fn run(
            net_: *const Network,
            result_: *AccumulatorPair,
            perspective: piece.Color,
            king_sq: u8,
            hidden_size_: usize,
            from_sq_: u8,
            to_sq_: u8,
            moved_piece_: piece.Type,
            moved_color_: piece.Color,
            final_piece_: piece.Type,
            captured_piece_: ?piece.Type,
            capture_sq_: ?u8,
            opp_color_: piece.Color,
            is_castling_: bool,
            rook_from_: ?u8,
            rook_to_: ?u8,
        ) void {
            if (is_castling_) {
                if (rook_from_) |rf| {
                    applyPerspectiveAddAddSubSub(
                        net_,
                        result_,
                        perspective,
                        king_sq,
                        to_sq_,
                        final_piece_,
                        moved_color_,
                        rook_to_.?,
                        .rook,
                        moved_color_,
                        from_sq_,
                        moved_piece_,
                        moved_color_,
                        rf,
                        .rook,
                        moved_color_,
                        hidden_size_,
                    );
                } else {
                    applyPerspectiveAddSub(
                        net_,
                        result_,
                        perspective,
                        king_sq,
                        to_sq_,
                        final_piece_,
                        moved_color_,
                        from_sq_,
                        moved_piece_,
                        moved_color_,
                        hidden_size_,
                    );
                }
            } else if (captured_piece_) |cp| {
                applyPerspectiveAddSubSub(
                    net_,
                    result_,
                    perspective,
                    king_sq,
                    to_sq_,
                    final_piece_,
                    moved_color_,
                    from_sq_,
                    moved_piece_,
                    moved_color_,
                    capture_sq_.?,
                    cp,
                    opp_color_,
                    hidden_size_,
                );
            } else {
                applyPerspectiveAddSub(
                    net_,
                    result_,
                    perspective,
                    king_sq,
                    to_sq_,
                    final_piece_,
                    moved_color_,
                    from_sq_,
                    moved_piece_,
                    moved_color_,
                    hidden_size_,
                );
            }
        }
    }.run;

    if (!refresh_white) {
        applyPerspective(
            net,
            result,
            .white,
            white_king_sq,
            hidden_size,
            from_sq,
            to_sq,
            moved_piece,
            moved_color,
            final_piece,
            captured_piece,
            capture_sq,
            opp_color,
            is_castling,
            rook_from,
            rook_to,
        );
    }
    if (!refresh_black) {
        applyPerspective(
            net,
            result,
            .black,
            black_king_sq,
            hidden_size,
            from_sq,
            to_sq,
            moved_piece,
            moved_color,
            final_piece,
            captured_piece,
            capture_sq,
            opp_color,
            is_castling,
            rook_from,
            rook_to,
        );
    }

    if (refresh_white) {
        initPerspectiveAccumulator(net, b, .white, result.white[0..hidden_size]);
    }
    if (refresh_black) {
        initPerspectiveAccumulator(net, b, .black, result.black[0..hidden_size]);
    }
}

fn evaluateV4FromAccumulators(
    net: *const Network,
    head: *const Network.V4Head,
    acc: *const AccumulatorPair,
    stm_is_white: bool,
) i32 {
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const q0: i32 = head.q0;
    const q: i32 = head.q;
    const scale: i32 = head.scale;
    const use_screlu = head.activation_type == 1;
    const final_den: i64 = @as(i64, q0) * @as(i64, q);

    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];

    var sum: i64 = 0;
    for (0..hidden_size) |idx| {
        const us = clampToActivationRange(us_acc[idx], q0);
        const them = clampToActivationRange(them_acc[idx], q0);

        if (use_screlu) {
            sum += @as(i64, us) * @as(i64, us) * @as(i64, head.output_weights[idx]);
            sum += @as(i64, them) * @as(i64, them) * @as(i64, head.output_weights[hidden_size + idx]);
        } else {
            sum += @as(i64, us) * @as(i64, head.output_weights[idx]);
            sum += @as(i64, them) * @as(i64, head.output_weights[hidden_size + idx]);
        }
    }

    if (use_screlu) {
        sum = divRoundNearestSigned(sum, q0);
    }
    sum += head.output_bias;
    return @intCast(divRoundNearestSigned(sum * scale, final_den));
}

fn evaluateV3FromAccumulators(
    net: *const Network,
    head: *const Network.V3Head,
    acc: *const AccumulatorPair,
    stm_is_white: bool,
) i32 {
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const use_screlu = head.activation_type == 1;
    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];

    var sum: i64 = 0;
    for (0..hidden_size) |idx| {
        const us = clampToActivationRange(us_acc[idx], Q0);
        const them = clampToActivationRange(them_acc[idx], Q0);

        if (use_screlu) {
            sum += @as(i64, us) * @as(i64, us) * @as(i64, head.output_weights[idx]);
            sum += @as(i64, them) * @as(i64, them) * @as(i64, head.output_weights[hidden_size + idx]);
        } else {
            sum += @as(i64, us) * @as(i64, head.output_weights[idx]);
            sum += @as(i64, them) * @as(i64, head.output_weights[hidden_size + idx]);
        }
    }

    if (use_screlu) {
        sum = @divTrunc(sum, Q0);
    }
    sum += head.output_bias;
    return @intCast(@divTrunc(sum * SCALE, Q0 * Q));
}

/// Evaluate using pre-computed accumulators (activation + head only).
/// Returns score from the side-to-move perspective.
pub fn evaluateFromAccumulators(
    net: *const Network,
    acc: *const AccumulatorPair,
    b: *Board,
) i32 {
    const stm_is_white = b.board.move == .white;
    return switch (net.head) {
        .v3 => |*head| evaluateV3FromAccumulators(net, head, acc, stm_is_white),
        .v4 => |*head| evaluateV4FromAccumulators(net, head, acc, stm_is_white),
    };
}

/// Returns score from the side-to-move perspective, same convention as classical eval.
/// This is the full-recompute path (non-incremental). Kept for fallback/gensfen use.
pub fn evaluate(net: *const Network, b: *Board) i32 {
    const acc = initAccumulators(net, b);
    return evaluateFromAccumulators(net, &acc, b);
}
