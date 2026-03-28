const std = @import("std");
const board = @import("bitboard.zig");
const Board = board.Board;
const piece = @import("piece.zig");

pub const EMBEDDED_NET = @embedFile("net.sknnue");

pub const LEGACY_INPUT_SIZE: usize = 768; // 2 colors * 6 piece types * 64 squares
pub const MAX_HIDDEN_SIZE: usize = 2048;
pub const MAX_DENSE_L1_SIZE: usize = 64;
pub const MAX_DENSE_L2_SIZE: usize = 64;
pub const Q0: i32 = 255;
pub const Q1: i32 = 128;
pub const Q: i32 = 64;
pub const SCALE: i32 = 400;
const MAX_NETWORK_BYTES = 64 * 1024 * 1024;

const MAGIC_V4 = "SYKNNUE4";
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
    pub const V4Head = struct {
        dense_l1_size: u16,
        dense_l2_size: u16,
        output_bucket_count: u8,
        q0: u16,
        q1: u16,
        q: u16,
        scale: u16,
        l1_biases: []i32,
        l1_weights: []i8, // [S * L1 * H]
        l2_biases: []i32,
        l2_weights: []i8, // [S * L2 * (2 * L1)]
        out_biases: []i32,
        out_weights: []i8, // [S * L2]
    };

    allocator: std.mem.Allocator,
    feature_set: FeatureSet,
    bucket_count: u8,
    bucket_layout: [64]u8,
    ft_hidden_size: u16,
    ft_biases: []i16,
    ft_weights: []i16,
    v4: V4Head,

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.ft_biases);
        self.allocator.free(self.ft_weights);
        self.allocator.free(self.v4.l1_biases);
        self.allocator.free(self.v4.l1_weights);
        self.allocator.free(self.v4.l2_biases);
        self.allocator.free(self.v4.l2_weights);
        self.allocator.free(self.v4.out_biases);
        self.allocator.free(self.v4.out_weights);
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
        if (data.len < 8) return error.InvalidNetwork;

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
    dense_l1_size: usize,
    dense_l2_size: usize,
    output_bucket_count: usize,
) ?u64 {
    var total: u64 = 0;
    const dense_expand = checkedMulU64(2, @as(u64, @intCast(dense_l1_size))) orelse return null;

    const ft_bias_bytes = checkedMulU64(@as(u64, @intCast(ft_hidden_size)), @sizeOf(i16)) orelse return null;
    total = checkedAddU64(total, ft_bias_bytes) orelse return null;

    const ft_weight_count = checkedMulU64(@as(u64, @intCast(input_size)), @as(u64, @intCast(ft_hidden_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(ft_weight_count, @sizeOf(i16)) orelse return null) orelse return null;

    const l1_bias_count = checkedMulU64(@as(u64, @intCast(output_bucket_count)), @as(u64, @intCast(dense_l1_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(l1_bias_count, @sizeOf(i32)) orelse return null) orelse return null;

    const l1_weight_count = checkedMulU64(l1_bias_count, @as(u64, @intCast(ft_hidden_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(l1_weight_count, @sizeOf(i8)) orelse return null) orelse return null;

    const l2_bias_count = checkedMulU64(@as(u64, @intCast(output_bucket_count)), @as(u64, @intCast(dense_l2_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(l2_bias_count, @sizeOf(i32)) orelse return null) orelse return null;

    const l2_weight_count = checkedMulU64(l2_bias_count, dense_expand) orelse return null;
    total = checkedAddU64(total, checkedMulU64(l2_weight_count, @sizeOf(i8)) orelse return null) orelse return null;

    total = checkedAddU64(total, checkedMulU64(@as(u64, @intCast(output_bucket_count)), @sizeOf(i32)) orelse return null) orelse return null;

    const out_weight_count = checkedMulU64(@as(u64, @intCast(output_bucket_count)), @as(u64, @intCast(dense_l2_size))) orelse return null;
    total = checkedAddU64(total, checkedMulU64(out_weight_count, @sizeOf(i8)) orelse return null) orelse return null;

    return total;
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
    if (ft_hidden_size == 0 or (ft_hidden_size % 2) != 0) return error.InvalidNetwork;
    if (ft_hidden_size > MAX_HIDDEN_SIZE) return error.NetworkTooLarge;

    const dense_l1_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const dense_l1_size: usize = @intCast(dense_l1_size_u16);
    if (dense_l1_size == 0 or dense_l1_size > MAX_DENSE_L1_SIZE) return error.InvalidNetwork;

    const dense_l2_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const dense_l2_size: usize = @intCast(dense_l2_size_u16);
    if (dense_l2_size == 0 or dense_l2_size > MAX_DENSE_L2_SIZE) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const output_bucket_count = data[pos];
    pos += 1;
    if (output_bucket_count == 0) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const bucket_count = data[pos];
    pos += 1;
    if (bucket_count == 0) return error.InvalidNetwork;

    const q0 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const q1 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const q = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const scale = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (q0 == 0 or q1 == 0 or q == 0 or scale == 0) return error.InvalidNetwork;
    if ((@as(u32, q0) * @as(u32, q1)) % @as(u32, q) != 0) return error.InvalidNetwork;

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
        dense_l1_size,
        dense_l2_size,
        output_bucket_count,
    ) orelse return error.InvalidNetwork;
    const expected_size = checkedAddU64(@as(u64, @intCast(pos)), payload_size) orelse return error.InvalidNetwork;
    if (expected_size != data.len) return error.InvalidNetwork;

    const ft_biases = try allocAndReadInts(i16, allocator, data, &pos, ft_hidden_size);
    errdefer allocator.free(ft_biases);

    const ft_weights = try allocAndReadInts(i16, allocator, data, &pos, input_size * ft_hidden_size);
    errdefer allocator.free(ft_weights);

    const output_bucket_count_usize: usize = output_bucket_count;
    const dense_expand = 2 * dense_l1_size;
    const l1_biases = try allocAndReadInts(i32, allocator, data, &pos, output_bucket_count_usize * dense_l1_size);
    errdefer allocator.free(l1_biases);

    const l1_weights = try allocAndReadInts(
        i8,
        allocator,
        data,
        &pos,
        output_bucket_count_usize * dense_l1_size * ft_hidden_size,
    );
    errdefer allocator.free(l1_weights);

    const l2_biases = try allocAndReadInts(i32, allocator, data, &pos, output_bucket_count_usize * dense_l2_size);
    errdefer allocator.free(l2_biases);

    const l2_weights = try allocAndReadInts(
        i8,
        allocator,
        data,
        &pos,
        output_bucket_count_usize * dense_l2_size * dense_expand,
    );
    errdefer allocator.free(l2_weights);

    const out_biases = try allocAndReadInts(i32, allocator, data, &pos, output_bucket_count_usize);
    errdefer allocator.free(out_biases);

    const out_weights = try allocAndReadInts(i8, allocator, data, &pos, output_bucket_count_usize * dense_l2_size);
    errdefer allocator.free(out_weights);

    if (pos != data.len) return error.InvalidNetwork;

    return Network{
        .allocator = allocator,
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .ft_hidden_size = ft_hidden_size_u16,
        .ft_biases = ft_biases,
        .ft_weights = ft_weights,
        .v4 = .{
            .dense_l1_size = dense_l1_size_u16,
            .dense_l2_size = dense_l2_size_u16,
            .output_bucket_count = output_bucket_count,
            .q0 = q0,
            .q1 = q1,
            .q = q,
            .scale = scale,
            .l1_biases = l1_biases,
            .l1_weights = l1_weights,
            .l2_biases = l2_biases,
            .l2_weights = l2_weights,
            .out_biases = out_biases,
            .out_weights = out_weights,
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

inline fn outputStackIndex(board_state: anytype, output_bucket_count: usize) usize {
    const piece_count: usize = @popCount(board_state.occupied());
    const non_king_piece_count = if (piece_count >= 2) piece_count - 2 else 0;
    return @min(output_bucket_count - 1, non_king_piece_count / 4);
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
    board_state: anytype,
) i32 {
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const half_hidden = hidden_size / 2;
    const q0: i32 = head.q0;
    const q1: i32 = head.q1;
    const q: i32 = head.q;
    const scale: i32 = head.scale;
    const l1_size: usize = head.dense_l1_size;
    const l2_size: usize = head.dense_l2_size;
    const dense_expand = 2 * l1_size;
    const stack_index = outputStackIndex(board_state, head.output_bucket_count);
    const l1_stack_base = stack_index * l1_size;
    const l1_weight_base = l1_stack_base * hidden_size;
    const l2_stack_base = stack_index * l2_size;
    const l2_weight_base = l2_stack_base * dense_expand;
    const out_weight_base = stack_index * l2_size;
    const r1_den: i64 = @divExact(@as(i64, q0) * @as(i64, q1), @as(i64, q));
    const r2_den: i64 = @as(i64, q) * @as(i64, q);

    var pooled: [MAX_HIDDEN_SIZE]u8 = undefined;
    var t1: [MAX_DENSE_L1_SIZE]i32 = undefined;
    var expanded: [2 * MAX_DENSE_L1_SIZE]i32 = undefined;
    var a2: [MAX_DENSE_L2_SIZE]i32 = undefined;
    if (hidden_size > pooled.len or l1_size > t1.len or dense_expand > expanded.len or l2_size > a2.len) unreachable;

    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];

    for (0..half_hidden) |idx| {
        const us_left = clampToActivationRange(us_acc[idx], q0);
        const us_right = clampToActivationRange(us_acc[half_hidden + idx], q0);
        const them_left = clampToActivationRange(them_acc[idx], q0);
        const them_right = clampToActivationRange(them_acc[half_hidden + idx], q0);

        pooled[idx] = @intCast(divRoundNearestNonNeg(@as(i64, us_left) * @as(i64, us_right), q0));
        pooled[half_hidden + idx] = @intCast(divRoundNearestNonNeg(@as(i64, them_left) * @as(i64, them_right), q0));
    }

    for (0..l1_size) |out_idx| {
        var z: i64 = head.l1_biases[l1_stack_base + out_idx];
        const weight_row = head.l1_weights[l1_weight_base + out_idx * hidden_size ..][0..hidden_size];

        for (0..hidden_size) |h| {
            z += @as(i64, pooled[h]) * @as(i64, weight_row[h]);
        }
        const rescaled_l1 = divRoundNearestSigned(z, r1_den);
        t1[out_idx] = clampToActivationRange(@intCast(rescaled_l1), q);
    }

    for (0..l1_size) |idx| {
        expanded[idx] = t1[idx] * q;
        expanded[l1_size + idx] = t1[idx] * t1[idx];
    }

    for (0..l2_size) |out_idx| {
        var z: i64 = head.l2_biases[l2_stack_base + out_idx];
        const weight_row = head.l2_weights[l2_weight_base + out_idx * dense_expand ..][0..dense_expand];
        for (0..dense_expand) |in_idx| {
            z += @as(i64, expanded[in_idx]) * @as(i64, weight_row[in_idx]);
        }
        const rescaled_l2 = divRoundNearestSigned(z, r2_den);
        a2[out_idx] = clampToActivationRange(@intCast(rescaled_l2), q);
    }

    var z3: i64 = head.out_biases[stack_index];
    const out_weights = head.out_weights[out_weight_base..][0..l2_size];
    for (0..l2_size) |in_idx| {
        z3 += @as(i64, a2[in_idx]) * @as(i64, out_weights[in_idx]);
    }

    return @intCast(divRoundNearestSigned(z3 * scale, r2_den));
}

/// Evaluate using pre-computed accumulators (activation + head only).
/// Returns score from the side-to-move perspective.
pub fn evaluateFromAccumulators(
    net: *const Network,
    acc: *const AccumulatorPair,
    b: *Board,
) i32 {
    const stm_is_white = b.board.move == .white;
    return evaluateV4FromAccumulators(net, &net.v4, acc, stm_is_white, b.board);
}

/// Returns score from the side-to-move perspective, same convention as classical eval.
/// This is the full-recompute path (non-incremental). Kept for fallback/gensfen use.
pub fn evaluate(net: *const Network, b: *Board) i32 {
    const acc = initAccumulators(net, b);
    return evaluateFromAccumulators(net, &acc, b);
}
