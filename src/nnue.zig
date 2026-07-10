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

const MAGIC_V6 = "SYKNNUE6";
const FORMAT_VERSION_V6: u16 = 6;
const MAGIC_V7 = "SYKNNUE7";
const FORMAT_VERSION_V7: u16 = 7;
const V7_HEADER_BYTES: usize = 160;
const V7_SECTION_ENTRY_BYTES: usize = 48;
const V7_HASH_OFFSET: usize = 114;
const MAX_V7_SECTIONS: usize = 32;
const MAX_DENSE_SIZE: usize = 256;

pub const Architecture = enum(u16) {
    legacy_linear = 0,
    pairwise_mlp = 1,
};

pub const FeatureSet = enum(u8) {
    legacy_psqt = 0,
    king_buckets_mirrored = 1,
};

pub const OutputBucketScheme = enum(u8) {
    single = 0,
    material_popcount = 1,
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
    AccumulatorBoundsExceeded,
};

/// NNUE format used by Sykora (little-endian).
pub const Network = struct {
    allocator: std.mem.Allocator,
    architecture: Architecture,
    feature_set: FeatureSet,
    bucket_count: u8, // input king buckets
    bucket_layout: [64]u8,
    ft_hidden_size: u16,
    activation_type: u8, // 0 = ReLU, 1 = SCReLU
    output_bucket_count: u8,
    output_bucket_scheme: OutputBucketScheme,
    q0: u16,
    pool_quant: u16,
    q: u16,
    scale: u16,
    dense1_size: u16,
    dense2_size: u16,
    ft_biases: []i16, // [H]
    ft_weights: []i16, // [I * H]
    output_biases: ?[]i32, // V6: [O]
    output_weights: ?[]i16, // V6: [O * 2 * H], bucket-major
    l1_biases: ?[]i32, // V7: [O, D1]
    l1_weights: ?[]i8, // V7: [O, H, D1]
    l2_biases: ?[]i32, // V7: [O, D2]
    l2_weights: ?[]i8, // V7: [O, 2*D1, D2]
    v7_output_biases: ?[]i32, // V7: [O]
    v7_output_weights: ?[]i8, // V7: [O, D2]

    pub fn deinit(self: *Network) void {
        self.allocator.free(self.ft_biases);
        self.allocator.free(self.ft_weights);
        if (self.output_biases) |values| self.allocator.free(values);
        if (self.output_weights) |values| self.allocator.free(values);
        if (self.l1_biases) |values| self.allocator.free(values);
        if (self.l1_weights) |values| self.allocator.free(values);
        if (self.l2_biases) |values| self.allocator.free(values);
        if (self.l2_weights) |values| self.allocator.free(values);
        if (self.v7_output_biases) |values| self.allocator.free(values);
        if (self.v7_output_weights) |values| self.allocator.free(values);
    }

    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
        if (data.len < 8) return error.InvalidNetwork;
        if (std.mem.eql(u8, data[0..8], MAGIC_V6)) return loadFromBytesV6(allocator, data);
        if (std.mem.eql(u8, data[0..8], MAGIC_V7)) return loadFromBytesV7(allocator, data);
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
        return LEGACY_INPUT_SIZE * @as(usize, self.bucket_count);
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

fn computeV6PayloadBytes(
    input_size: usize,
    ft_hidden_size: usize,
    output_bucket_count: usize,
) ?u64 {
    var total: u64 = 0;

    const hidden_size_u64: u64 = @intCast(ft_hidden_size);
    const ft_bias_bytes = checkedMulU64(hidden_size_u64, @sizeOf(i16)) orelse return null;
    total = checkedAddU64(total, ft_bias_bytes) orelse return null;

    const ft_weight_count = checkedMulU64(@as(u64, @intCast(input_size)), hidden_size_u64) orelse return null;
    total = checkedAddU64(total, checkedMulU64(ft_weight_count, @sizeOf(i16)) orelse return null) orelse return null;

    const bias_bytes = checkedMulU64(@as(u64, @intCast(output_bucket_count)), @sizeOf(i32)) orelse return null;
    total = checkedAddU64(total, bias_bytes) orelse return null;

    const single_head_weight_count = checkedMulU64(2, hidden_size_u64) orelse return null;
    const output_weight_count = checkedMulU64(
        @as(u64, @intCast(output_bucket_count)),
        single_head_weight_count,
    ) orelse return null;
    total = checkedAddU64(total, checkedMulU64(output_weight_count, @sizeOf(i16)) orelse return null) orelse return null;

    return total;
}

/// Production accumulators are i16. Prove at load time that no reachable
/// accumulator value can leave i16 range: for every hidden unit and king
/// bucket, |bias| plus the sum of the `ACC_BOUND_FEATURES` largest absolute
/// weights among the bucket's rows must fit in i16. A position activates at
/// most 32 features per perspective, all from one bucket; the fused add/sub
/// update kernels evaluate adds before subs, so intermediates can transiently
/// hold up to 2 extra rows — hence 34. Nets that fail are rejected rather
/// than silently evaluated with wrapped or saturated sums.
const ACC_BOUND_FEATURES: usize = 34;

fn validateI16AccumulatorBounds(
    allocator: std.mem.Allocator,
    ft_biases: []const i16,
    ft_weights: []const i16,
    hidden_size: usize,
    bucket_count: usize,
) LoadError!void {
    const tops = allocator.alloc(u16, hidden_size * ACC_BOUND_FEATURES) catch return error.OutOfMemory;
    defer allocator.free(tops);
    const mins = allocator.alloc(u16, hidden_size) catch return error.OutOfMemory;
    defer allocator.free(mins);

    for (0..bucket_count) |bucket| {
        @memset(tops, 0);
        @memset(mins, 0);
        const bucket_base = bucket * LEGACY_INPUT_SIZE * hidden_size;
        for (0..LEGACY_INPUT_SIZE) |feature| {
            const row = ft_weights[bucket_base + feature * hidden_size ..][0..hidden_size];
            for (row, mins, 0..) |weight, current_min, h| {
                const magnitude: u16 = @abs(weight);
                if (magnitude <= current_min) continue;
                const unit_tops = tops[h * ACC_BOUND_FEATURES ..][0..ACC_BOUND_FEATURES];
                for (unit_tops) |*slot| {
                    if (slot.* == current_min) {
                        slot.* = magnitude;
                        break;
                    }
                }
                var new_min: u16 = std.math.maxInt(u16);
                for (unit_tops) |slot| new_min = @min(new_min, slot);
                mins[h] = new_min;
            }
        }
        for (0..hidden_size) |h| {
            var bound: i64 = @abs(ft_biases[h]);
            for (tops[h * ACC_BOUND_FEATURES ..][0..ACC_BOUND_FEATURES]) |magnitude| {
                bound += magnitude;
            }
            if (bound > std.math.maxInt(i16)) return error.AccumulatorBoundsExceeded;
        }
    }
}

fn loadFromBytesV6(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
    var pos: usize = 8;

    const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (version != FORMAT_VERSION_V6) return error.UnsupportedVersion;

    if (pos >= data.len) return error.InvalidNetwork;
    const feature_set = std.meta.intToEnum(FeatureSet, data[pos]) catch return error.InvalidNetwork;
    pos += 1;
    if (feature_set != .king_buckets_mirrored) return error.InvalidNetwork;

    const ft_hidden_size_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const ft_hidden_size: usize = @intCast(ft_hidden_size_u16);
    if (ft_hidden_size == 0 or ft_hidden_size > MAX_HIDDEN_SIZE) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const activation_type = data[pos];
    pos += 1;
    if (activation_type > 1) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const bucket_count = data[pos];
    pos += 1;
    if (bucket_count == 0) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const output_bucket_count = data[pos];
    pos += 1;
    if (output_bucket_count == 0) return error.InvalidNetwork;

    if (pos >= data.len) return error.InvalidNetwork;
    const output_bucket_scheme = std.meta.intToEnum(OutputBucketScheme, data[pos]) catch return error.InvalidNetwork;
    pos += 1;
    switch (output_bucket_scheme) {
        .single => if (output_bucket_count != 1) return error.InvalidNetwork,
        .material_popcount => if (32 % @as(usize, output_bucket_count) != 0) return error.InvalidNetwork,
    }

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
    const payload_size = computeV6PayloadBytes(
        input_size,
        ft_hidden_size,
        output_bucket_count,
    ) orelse return error.InvalidNetwork;
    const expected_size = checkedAddU64(@as(u64, @intCast(pos)), payload_size) orelse return error.InvalidNetwork;
    if (expected_size != data.len) return error.InvalidNetwork;

    // Payload order: output_biases, ft_biases, ft_weights, output_weights.
    const output_biases = try allocAndReadInts(i32, allocator, data, &pos, output_bucket_count);
    errdefer allocator.free(output_biases);

    const ft_biases = try allocAndReadInts(i16, allocator, data, &pos, ft_hidden_size);
    errdefer allocator.free(ft_biases);

    const ft_weights = try allocAndReadInts(i16, allocator, data, &pos, input_size * ft_hidden_size);
    errdefer allocator.free(ft_weights);

    const output_weights = try allocAndReadInts(i16, allocator, data, &pos, @as(usize, output_bucket_count) * 2 * ft_hidden_size);
    errdefer allocator.free(output_weights);

    if (pos != data.len) return error.InvalidNetwork;

    try validateI16AccumulatorBounds(allocator, ft_biases, ft_weights, ft_hidden_size, bucket_count);

    return Network{
        .allocator = allocator,
        .architecture = .legacy_linear,
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .ft_hidden_size = ft_hidden_size_u16,
        .activation_type = activation_type,
        .output_bucket_count = output_bucket_count,
        .output_bucket_scheme = output_bucket_scheme,
        .q0 = q0,
        .pool_quant = 0,
        .q = q,
        .scale = scale,
        .dense1_size = 0,
        .dense2_size = 0,
        .ft_biases = ft_biases,
        .ft_weights = ft_weights,
        .output_biases = output_biases,
        .output_weights = output_weights,
        .l1_biases = null,
        .l1_weights = null,
        .l2_biases = null,
        .l2_weights = null,
        .v7_output_biases = null,
        .v7_output_weights = null,
    };
}

const V7Section = struct {
    id: u16,
    element_type: u8,
    rank: u8,
    dimensions: [4]u32,
    offset: usize,
    byte_length: usize,
};

fn bytesAreZero(bytes: []const u8) bool {
    for (bytes) |value| {
        if (value != 0) return false;
    }
    return true;
}

fn v7TypeSize(element_type: u8) ?usize {
    return switch (element_type) {
        1, 2, 5 => 1,
        3 => 2,
        4 => 4,
        else => null,
    };
}

fn findV7Section(sections: []const V7Section, id: u16) ?V7Section {
    for (sections) |section| {
        if (section.id == id) return section;
    }
    return null;
}

fn requireV7Section(
    sections: []const V7Section,
    id: u16,
    element_type: u8,
    dimensions: []const u32,
) LoadError!V7Section {
    const section = findV7Section(sections, id) orelse return error.InvalidNetwork;
    if (section.element_type != element_type or section.rank != dimensions.len) return error.InvalidNetwork;
    for (dimensions, 0..) |dimension, index| {
        if (section.dimensions[index] != dimension) return error.InvalidNetwork;
    }
    return section;
}

fn allocV7SectionInts(
    comptime T: type,
    allocator: std.mem.Allocator,
    data: []const u8,
    section: V7Section,
) LoadError![]T {
    if (section.byte_length % @sizeOf(T) != 0) return error.InvalidNetwork;
    var pos = section.offset;
    const values = try allocAndReadInts(T, allocator, data, &pos, section.byte_length / @sizeOf(T));
    if (pos != section.offset + section.byte_length) {
        allocator.free(values);
        return error.InvalidNetwork;
    }
    return values;
}

fn loadFromBytesV7(allocator: std.mem.Allocator, data: []const u8) LoadError!Network {
    if (data.len < V7_HEADER_BYTES) return error.InvalidNetwork;
    var pos: usize = 8;

    const version = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const header_bytes = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const section_count_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const section_entry_bytes = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const flags = readBytesInt(u32, data, &pos) orelse return error.InvalidNetwork;
    if (version != FORMAT_VERSION_V7) return error.UnsupportedVersion;
    if (header_bytes != V7_HEADER_BYTES or section_entry_bytes != V7_SECTION_ENTRY_BYTES or flags != 0) {
        return error.InvalidNetwork;
    }

    const architecture = std.meta.intToEnum(
        Architecture,
        readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork,
    ) catch return error.InvalidNetwork;
    if (architecture != .pairwise_mlp) return error.InvalidNetwork;
    const feature_set_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (feature_set_u16 > 255) return error.InvalidNetwork;
    const feature_set = std.meta.intToEnum(FeatureSet, @as(u8, @intCast(feature_set_u16))) catch return error.InvalidNetwork;
    if (feature_set != .king_buckets_mirrored) return error.InvalidNetwork;

    const bucket_count_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const output_bucket_count_u16 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const hidden_size = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const dense1_size = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const dense2_size = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (bucket_count_u16 == 0 or bucket_count_u16 > 255 or output_bucket_count_u16 != 8) {
        return error.InvalidNetwork;
    }
    if (hidden_size == 0 or hidden_size > MAX_HIDDEN_SIZE or hidden_size % 2 != 0) {
        return error.InvalidNetwork;
    }
    if (dense1_size == 0 or dense1_size > MAX_DENSE_SIZE or dense2_size == 0 or dense2_size > MAX_DENSE_SIZE) {
        return error.InvalidNetwork;
    }

    if (pos + 8 > data.len) return error.InvalidNetwork;
    const ft_activation = data[pos];
    const pooling = data[pos + 1];
    const dense1_activation = data[pos + 2];
    const dense2_activation = data[pos + 3];
    const output_selector = data[pos + 4];
    pos += 5;
    if (ft_activation != 0 or pooling != 1 or dense1_activation != 1 or dense2_activation != 1 or output_selector != 1) {
        return error.InvalidNetwork;
    }
    if (!bytesAreZero(data[pos .. pos + 3])) return error.InvalidNetwork;
    pos += 3;

    const q0 = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const pool_quant = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const q = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    const scale = readBytesInt(u16, data, &pos) orelse return error.InvalidNetwork;
    if (q0 != Q0 or pool_quant != 128 or q != Q or scale != SCALE) return error.InvalidNetwork;

    const bucket_count: u8 = @intCast(bucket_count_u16);
    var bucket_layout = [_]u8{0} ** 64;
    var max_bucket: u8 = 0;
    for (&bucket_layout) |*entry| {
        if (pos >= data.len) return error.InvalidNetwork;
        entry.* = data[pos];
        pos += 1;
        if (entry.* >= bucket_count) return error.InvalidNetwork;
        max_bucket = @max(max_bucket, entry.*);
    }
    if (@as(u16, max_bucket) + 1 != bucket_count_u16) return error.InvalidNetwork;

    if (pos + 32 + 14 != V7_HEADER_BYTES) return error.InvalidNetwork;
    const expected_hash = data[pos .. pos + 32];
    pos += 32;
    if (!bytesAreZero(data[pos .. pos + 14])) return error.InvalidNetwork;
    pos += 14;
    if (pos != V7_HEADER_BYTES) return error.InvalidNetwork;

    var actual_hash: [32]u8 = undefined;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(data[0..V7_HASH_OFFSET]);
    hasher.update(&([_]u8{0} ** 32));
    hasher.update(data[V7_HASH_OFFSET + 32 ..]);
    hasher.final(&actual_hash);
    if (!std.mem.eql(u8, expected_hash, &actual_hash)) return error.InvalidNetwork;

    const section_count: usize = section_count_u16;
    if (section_count != 8 or section_count > MAX_V7_SECTIONS) return error.InvalidNetwork;
    const table_bytes = std.math.mul(usize, section_count, V7_SECTION_ENTRY_BYTES) catch return error.InvalidNetwork;
    const table_end = std.math.add(usize, V7_HEADER_BYTES, table_bytes) catch return error.InvalidNetwork;
    if (table_end > data.len) return error.InvalidNetwork;

    var section_storage: [MAX_V7_SECTIONS]V7Section = undefined;
    const sections = section_storage[0..section_count];
    var previous_end = std.mem.alignForward(usize, table_end, 64);
    if (previous_end > data.len or !bytesAreZero(data[table_end..previous_end])) return error.InvalidNetwork;

    for (sections, 0..) |*section, index| {
        var entry_pos = V7_HEADER_BYTES + index * V7_SECTION_ENTRY_BYTES;
        const id = readBytesInt(u16, data, &entry_pos) orelse return error.InvalidNetwork;
        if (entry_pos + 2 > data.len) return error.InvalidNetwork;
        const element_type = data[entry_pos];
        const rank = data[entry_pos + 1];
        entry_pos += 2;
        const section_flags = readBytesInt(u32, data, &entry_pos) orelse return error.InvalidNetwork;
        if (rank == 0 or rank > 4 or section_flags != 1) return error.InvalidNetwork;
        var dimensions = [_]u32{1} ** 4;
        for (&dimensions) |*dimension| {
            dimension.* = readBytesInt(u32, data, &entry_pos) orelse return error.InvalidNetwork;
        }
        for (dimensions[0..rank]) |dimension| {
            if (dimension == 0) return error.InvalidNetwork;
        }
        for (dimensions[rank..]) |dimension| {
            if (dimension != 1) return error.InvalidNetwork;
        }
        const file_offset_u64 = readBytesInt(u64, data, &entry_pos) orelse return error.InvalidNetwork;
        const byte_length_u64 = readBytesInt(u64, data, &entry_pos) orelse return error.InvalidNetwork;
        const crc32 = readBytesInt(u32, data, &entry_pos) orelse return error.InvalidNetwork;
        const reserved = readBytesInt(u32, data, &entry_pos) orelse return error.InvalidNetwork;
        if (reserved != 0 or entry_pos != V7_HEADER_BYTES + (index + 1) * V7_SECTION_ENTRY_BYTES) {
            return error.InvalidNetwork;
        }
        const file_offset = std.math.cast(usize, file_offset_u64) orelse return error.InvalidNetwork;
        const byte_length = std.math.cast(usize, byte_length_u64) orelse return error.InvalidNetwork;
        if (file_offset % 64 != 0 or file_offset < previous_end) return error.InvalidNetwork;
        const section_end = std.math.add(usize, file_offset, byte_length) catch return error.InvalidNetwork;
        if (section_end > data.len or !bytesAreZero(data[previous_end..file_offset])) return error.InvalidNetwork;

        var expected_length = v7TypeSize(element_type) orelse return error.InvalidNetwork;
        for (dimensions[0..rank]) |dimension| {
            expected_length = std.math.mul(usize, expected_length, dimension) catch return error.InvalidNetwork;
        }
        if (expected_length != byte_length) return error.InvalidNetwork;
        if (std.hash.crc.Crc32.hash(data[file_offset..section_end]) != crc32) return error.InvalidNetwork;
        for (sections[0..index]) |prior| {
            if (prior.id == id) return error.InvalidNetwork;
        }
        section.* = .{
            .id = id,
            .element_type = element_type,
            .rank = rank,
            .dimensions = dimensions,
            .offset = file_offset,
            .byte_length = byte_length,
        };
        previous_end = section_end;
    }
    if (previous_end != data.len) return error.InvalidNetwork;

    const h32: u32 = hidden_size;
    const d1_32: u32 = dense1_size;
    const d2_32: u32 = dense2_size;
    const o32: u32 = output_bucket_count_u16;
    const input_size_u32 = std.math.mul(u32, 768, bucket_count_u16) catch return error.InvalidNetwork;
    const ft_bias_section = try requireV7Section(sections, 1, 3, &.{h32});
    const ft_weight_section = try requireV7Section(sections, 2, 3, &.{ input_size_u32, h32 });
    const l1_bias_section = try requireV7Section(sections, 10, 4, &.{ o32, d1_32 });
    const l1_weight_section = try requireV7Section(sections, 11, 1, &.{ o32, h32, d1_32 });
    const l2_bias_section = try requireV7Section(sections, 12, 4, &.{ o32, d2_32 });
    const l2_weight_section = try requireV7Section(sections, 13, 1, &.{ o32, 2 * d1_32, d2_32 });
    const output_bias_section = try requireV7Section(sections, 14, 4, &.{o32});
    const output_weight_section = try requireV7Section(sections, 15, 1, &.{ o32, d2_32 });

    const ft_biases = try allocV7SectionInts(i16, allocator, data, ft_bias_section);
    errdefer allocator.free(ft_biases);
    const ft_weights = try allocV7SectionInts(i16, allocator, data, ft_weight_section);
    errdefer allocator.free(ft_weights);
    const l1_biases = try allocV7SectionInts(i32, allocator, data, l1_bias_section);
    errdefer allocator.free(l1_biases);
    const l1_weights = try allocV7SectionInts(i8, allocator, data, l1_weight_section);
    errdefer allocator.free(l1_weights);
    const l2_biases = try allocV7SectionInts(i32, allocator, data, l2_bias_section);
    errdefer allocator.free(l2_biases);
    const l2_weights = try allocV7SectionInts(i8, allocator, data, l2_weight_section);
    errdefer allocator.free(l2_weights);
    const output_biases = try allocV7SectionInts(i32, allocator, data, output_bias_section);
    errdefer allocator.free(output_biases);
    const output_weights = try allocV7SectionInts(i8, allocator, data, output_weight_section);
    errdefer allocator.free(output_weights);

    // The SIMD registered tail accumulates into i32 lanes. Reject otherwise
    // structurally valid files whose biases could overflow those lanes.
    if (dense1_size == 16 and dense2_size == 32) {
        const l1_margin = @as(i64, hidden_size) * 127 * 127;
        for (l1_biases) |bias| {
            const value: i64 = bias;
            if (value < std.math.minInt(i32) + l1_margin or value > std.math.maxInt(i32) - l1_margin) {
                return error.InvalidNetwork;
            }
        }
        const l2_margin: i64 = 32 * 64 * 127;
        for (l2_biases) |bias| {
            const value: i64 = bias;
            if (value < std.math.minInt(i32) + l2_margin or value > std.math.maxInt(i32) - l2_margin) {
                return error.InvalidNetwork;
            }
        }
    }

    try validateI16AccumulatorBounds(allocator, ft_biases, ft_weights, hidden_size, bucket_count);

    return .{
        .allocator = allocator,
        .architecture = architecture,
        .feature_set = feature_set,
        .bucket_count = bucket_count,
        .bucket_layout = bucket_layout,
        .ft_hidden_size = hidden_size,
        .activation_type = 0,
        .output_bucket_count = @intCast(output_bucket_count_u16),
        .output_bucket_scheme = .material_popcount,
        .q0 = q0,
        .pool_quant = pool_quant,
        .q = q,
        .scale = scale,
        .dense1_size = dense1_size,
        .dense2_size = dense2_size,
        .ft_biases = ft_biases,
        .ft_weights = ft_weights,
        .output_biases = null,
        .output_weights = null,
        .l1_biases = l1_biases,
        .l1_weights = l1_weights,
        .l2_biases = l2_biases,
        .l2_weights = l2_weights,
        .v7_output_biases = output_biases,
        .v7_output_weights = output_weights,
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

/// Production accumulators are i16 (`AccumulatorPair`); the loader proves
/// per-hidden-unit bounds so no reachable position can overflow them.
/// `AccumulatorPairWide` is the i32 reference backend kept for the parity
/// gate: with in-range weights the two must stay bit-identical.
pub fn AccumulatorPairT(comptime T: type) type {
    return struct {
        pub const Elem = T;

        white: [MAX_HIDDEN_SIZE]T,
        black: [MAX_HIDDEN_SIZE]T,

        pub inline fn slice(self: *@This(), perspective: piece.Color, hidden_size: usize) []T {
            return switch (perspective) {
                .white => self.white[0..hidden_size],
                .black => self.black[0..hidden_size],
            };
        }

        pub inline fn sliceConst(self: *const @This(), perspective: piece.Color, hidden_size: usize) []const T {
            return switch (perspective) {
                .white => self.white[0..hidden_size],
                .black => self.black[0..hidden_size],
            };
        }
    };
}

pub const AccumulatorPair = AccumulatorPairT(i16);
pub const AccumulatorPairWide = AccumulatorPairT(i32);

const SIMD_LANES = std.simd.suggestVectorLength(i16) orelse 8;
const WeightVec = @Vector(SIMD_LANES, i16);
const I32Vec = @Vector(SIMD_LANES, i32);
const I64Vec = @Vector(SIMD_LANES, i64);

inline fn AccVecOf(comptime Slice: type) type {
    return @Vector(SIMD_LANES, std.meta.Elem(Slice));
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

inline fn initAccumulatorBiases(dest: anytype, biases: []const i16) void {
    const AccVec = AccVecOf(@TypeOf(dest));
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const bias_ptr: *align(1) const WeightVec = @ptrCast(&biases[h]);
        const dest_ptr: *align(1) AccVec = @ptrCast(&dest[h]);
        dest_ptr.* = @intCast(bias_ptr.*);
    }

    while (h < dest.len) : (h += 1) {
        dest[h] = biases[h];
    }
}

inline fn applyFeatureSlice(comptime add: bool, dest: anytype, weights: []const i16) void {
    const AccVec = AccVecOf(@TypeOf(dest));
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const weight_ptr: *align(1) const WeightVec = @ptrCast(&weights[h]);
        const dest_ptr: *align(1) AccVec = @ptrCast(&dest[h]);
        const weight_vec: AccVec = @intCast(weight_ptr.*);
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

inline fn applyFeatureSlicesFromPrevAddSub(
    dest: anytype,
    prev: anytype,
    add_weights: []const i16,
    sub_weights: []const i16,
) void {
    const AccVec = AccVecOf(@TypeOf(dest));
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const prev_ptr: *align(1) const AccVec = @ptrCast(&prev[h]);
        const add_ptr: *align(1) const WeightVec = @ptrCast(&add_weights[h]);
        const sub_ptr: *align(1) const WeightVec = @ptrCast(&sub_weights[h]);
        const dest_ptr: *align(1) AccVec = @ptrCast(&dest[h]);
        dest_ptr.* = prev_ptr.* + @as(AccVec, @intCast(add_ptr.*)) - @as(AccVec, @intCast(sub_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] = prev[h] + add_weights[h] - sub_weights[h];
    }
}

inline fn applyFeatureSlicesFromPrevAddSubSub(
    dest: anytype,
    prev: anytype,
    add_weights: []const i16,
    sub_a_weights: []const i16,
    sub_b_weights: []const i16,
) void {
    const AccVec = AccVecOf(@TypeOf(dest));
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const prev_ptr: *align(1) const AccVec = @ptrCast(&prev[h]);
        const add_ptr: *align(1) const WeightVec = @ptrCast(&add_weights[h]);
        const sub_a_ptr: *align(1) const WeightVec = @ptrCast(&sub_a_weights[h]);
        const sub_b_ptr: *align(1) const WeightVec = @ptrCast(&sub_b_weights[h]);
        const dest_ptr: *align(1) AccVec = @ptrCast(&dest[h]);
        dest_ptr.* = prev_ptr.* + @as(AccVec, @intCast(add_ptr.*)) - @as(AccVec, @intCast(sub_a_ptr.*)) - @as(AccVec, @intCast(sub_b_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] = prev[h] + add_weights[h] - sub_a_weights[h] - sub_b_weights[h];
    }
}

inline fn applyFeatureSlicesFromPrevAddAddSubSub(
    dest: anytype,
    prev: anytype,
    add_a_weights: []const i16,
    add_b_weights: []const i16,
    sub_a_weights: []const i16,
    sub_b_weights: []const i16,
) void {
    const AccVec = AccVecOf(@TypeOf(dest));
    var h: usize = 0;
    while (h + SIMD_LANES <= dest.len) : (h += SIMD_LANES) {
        const prev_ptr: *align(1) const AccVec = @ptrCast(&prev[h]);
        const add_a_ptr: *align(1) const WeightVec = @ptrCast(&add_a_weights[h]);
        const add_b_ptr: *align(1) const WeightVec = @ptrCast(&add_b_weights[h]);
        const sub_a_ptr: *align(1) const WeightVec = @ptrCast(&sub_a_weights[h]);
        const sub_b_ptr: *align(1) const WeightVec = @ptrCast(&sub_b_weights[h]);
        const dest_ptr: *align(1) AccVec = @ptrCast(&dest[h]);
        dest_ptr.* = prev_ptr.* + @as(AccVec, @intCast(add_a_ptr.*)) + @as(AccVec, @intCast(add_b_ptr.*)) - @as(AccVec, @intCast(sub_a_ptr.*)) - @as(AccVec, @intCast(sub_b_ptr.*));
    }

    while (h < dest.len) : (h += 1) {
        dest[h] = prev[h] + add_a_weights[h] + add_b_weights[h] - sub_a_weights[h] - sub_b_weights[h];
    }
}

inline fn clampVecToActivationRange(values: I32Vec, max_value: i32) I32Vec {
    const zero: I32Vec = @splat(0);
    const max_vec: I32Vec = @splat(max_value);
    return @min(@max(values, zero), max_vec);
}

inline fn reduceProductToI64(values: I32Vec, weights: I32Vec) i64 {
    const product = values * weights;
    return @reduce(.Add, @as(I64Vec, @intCast(product)));
}

inline fn activatedDot(
    acc_values: anytype,
    weights: []const i16,
    hidden_size: usize,
    activation_type: u8,
    q0: i32,
) i64 {
    const AccVec = AccVecOf(@TypeOf(acc_values));
    const use_screlu = activation_type == 1;
    var sum: i64 = 0;
    var h: usize = 0;

    while (h + SIMD_LANES <= hidden_size) : (h += SIMD_LANES) {
        const acc_ptr: *align(1) const AccVec = @ptrCast(&acc_values[h]);
        const weight_ptr: *align(1) const WeightVec = @ptrCast(&weights[h]);
        const clamped = clampVecToActivationRange(@intCast(acc_ptr.*), q0);
        const activated = if (use_screlu) clamped * clamped else clamped;
        sum += reduceProductToI64(activated, @intCast(weight_ptr.*));
    }

    while (h < hidden_size) : (h += 1) {
        const v = clampToActivationRange(acc_values[h], q0);
        if (use_screlu) {
            sum += @as(i64, v) * @as(i64, v) * @as(i64, weights[h]);
        } else {
            sum += @as(i64, v) * @as(i64, weights[h]);
        }
    }

    return sum;
}

fn initPerspectiveAccumulator(
    net: *const Network,
    b: *Board,
    perspective: piece.Color,
    dest: anytype,
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

fn initAccumulatorsT(comptime T: type, net: *const Network, b: *Board) AccumulatorPairT(T) {
    var acc = AccumulatorPairT(T){
        .white = [_]T{0} ** MAX_HIDDEN_SIZE,
        .black = [_]T{0} ** MAX_HIDDEN_SIZE,
    };

    initPerspectiveAccumulator(net, b, .white, acc.white[0..@intCast(net.ft_hidden_size)]);
    initPerspectiveAccumulator(net, b, .black, acc.black[0..@intCast(net.ft_hidden_size)]);

    return acc;
}

/// Full recompute of accumulators from board state (used at search root).
pub fn initAccumulators(net: *const Network, b: *Board) AccumulatorPair {
    return initAccumulatorsT(i16, net, b);
}

/// Full recompute into the i32 reference backend (parity checking only).
pub fn initAccumulatorsWide(net: *const Network, b: *Board) AccumulatorPairWide {
    return initAccumulatorsT(i32, net, b);
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

inline fn applyPerspectiveFromPrevAddSub(
    net: *const Network,
    prev: anytype,
    result: anytype,
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
    const dest = result.slice(perspective, hidden_size);
    const prev_slice = prev.sliceConst(perspective, hidden_size);
    const add_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_sq, add_pt, add_color, hidden_size);
    const sub_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_sq, sub_pt, sub_color, hidden_size);
    applyFeatureSlicesFromPrevAddSub(dest, prev_slice, add_weights, sub_weights);
}

inline fn applyPerspectiveFromPrevAddSubSub(
    net: *const Network,
    prev: anytype,
    result: anytype,
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
    const dest = result.slice(perspective, hidden_size);
    const prev_slice = prev.sliceConst(perspective, hidden_size);
    const add_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_sq, add_pt, add_color, hidden_size);
    const sub_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_a_sq, sub_a_pt, sub_a_color, hidden_size);
    const sub_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_b_sq, sub_b_pt, sub_b_color, hidden_size);
    applyFeatureSlicesFromPrevAddSubSub(dest, prev_slice, add_weights, sub_a_weights, sub_b_weights);
}

inline fn applyPerspectiveFromPrevAddAddSubSub(
    net: *const Network,
    prev: anytype,
    result: anytype,
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
    const dest = result.slice(perspective, hidden_size);
    const prev_slice = prev.sliceConst(perspective, hidden_size);
    const add_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_a_sq, add_a_pt, add_a_color, hidden_size);
    const add_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, add_b_sq, add_b_pt, add_b_color, hidden_size);
    const sub_a_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_a_sq, sub_a_pt, sub_a_color, hidden_size);
    const sub_b_weights = perspectiveFeatureWeights(net, perspective, king_sq, sub_b_sq, sub_b_pt, sub_b_color, hidden_size);
    applyFeatureSlicesFromPrevAddAddSubSub(dest, prev_slice, add_a_weights, add_b_weights, sub_a_weights, sub_b_weights);
}

/// Incremental accumulator update after a move.
/// Writes `result = prev + feature deltas` for each unchanged perspective.
/// `prev`/`result` may point at `AccumulatorPair` (production, i16) or
/// `AccumulatorPairWide` (i32 reference backend).
pub fn updateAccumulators(
    net: *const Network,
    b: *Board,
    prev: anytype,
    result: anytype,
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

    const applyPerspective = struct {
        inline fn run(
            net_: *const Network,
            prev_: anytype,
            result_: anytype,
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
                    applyPerspectiveFromPrevAddAddSubSub(
                        net_,
                        prev_,
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
                    applyPerspectiveFromPrevAddSub(
                        net_,
                        prev_,
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
                applyPerspectiveFromPrevAddSubSub(
                    net_,
                    prev_,
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
                applyPerspectiveFromPrevAddSub(
                    net_,
                    prev_,
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
            prev,
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
            prev,
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

inline fn outputBucket(net: *const Network, b: *Board) usize {
    switch (net.output_bucket_scheme) {
        .single => return 0,
        .material_popcount => {
            const o: usize = net.output_bucket_count;
            const n = @popCount(b.board.occupied()); // includes kings, 2..32
            const divisor = (32 + o - 1) / o;
            const non_king = if (n >= 2) n - 2 else 0;
            return @min(non_king / divisor, o - 1);
        },
    }
}

fn evaluateLinearFromAccumulators(
    net: *const Network,
    acc: anytype,
    b: *Board,
) i32 {
    const stm_is_white = b.board.move == .white;
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const q0: i32 = net.q0;
    const q: i32 = net.q;
    const scale: i32 = net.scale;
    const use_screlu = net.activation_type == 1;
    const final_den: i64 = @as(i64, q0) * @as(i64, q);

    const bucket = outputBucket(net, b);
    const weights_base = bucket * 2 * hidden_size;
    const all_weights = net.output_weights orelse unreachable;
    const weights = all_weights[weights_base .. weights_base + 2 * hidden_size];

    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];

    var sum = activatedDot(us_acc, weights[0..hidden_size], hidden_size, net.activation_type, q0) +
        activatedDot(them_acc, weights[hidden_size .. 2 * hidden_size], hidden_size, net.activation_type, q0);

    if (use_screlu) {
        sum = divRoundNearestSigned(sum, q0);
    }
    sum += (net.output_biases orelse unreachable)[bucket];
    return @intCast(divRoundNearestSigned(sum * scale, final_den));
}

const V7Dense1Vec = @Vector(16, i32);
const V7Dense1WeightVec = @Vector(16, i8);
const V7Dense2Vec = @Vector(32, i32);
const V7Dense2WeightVec = @Vector(32, i8);

fn evaluatePairwiseMlpTail16x32(
    net: *const Network,
    pooled: []const i32,
    bucket: usize,
) i32 {
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const q: i64 = net.q;
    const pool_quant: i64 = net.pool_quant;
    const l1_biases = net.l1_biases orelse unreachable;
    const l1_weights = net.l1_weights orelse unreachable;
    const l2_biases = net.l2_biases orelse unreachable;
    const l2_weights = net.l2_weights orelse unreachable;
    const output_biases = net.v7_output_biases orelse unreachable;
    const output_weights = net.v7_output_weights orelse unreachable;

    const l1_bias_ptr: *align(1) const V7Dense1Vec = @ptrCast(&l1_biases[bucket * 16]);
    var l1_sums = l1_bias_ptr.*;
    for (0..hidden_size) |input| {
        const weight_index = (bucket * hidden_size + input) * 16;
        const weight_ptr: *align(1) const V7Dense1WeightVec = @ptrCast(&l1_weights[weight_index]);
        const input_vec: V7Dense1Vec = @splat(pooled[input]);
        l1_sums += input_vec * @as(V7Dense1Vec, @intCast(weight_ptr.*));
    }

    var dense1_activated: [32]i32 = undefined;
    for (0..16) |output| {
        const value = divRoundNearestSigned(l1_sums[output], pool_quant);
        dense1_activated[output] = @intCast(@min(@max(value, 0), q));
        const squared = divRoundNearestNonNeg(value * value, q);
        dense1_activated[16 + output] = @intCast(@min(squared, q));
    }

    const l2_bias_ptr: *align(1) const V7Dense2Vec = @ptrCast(&l2_biases[bucket * 32]);
    var l2_sums = l2_bias_ptr.*;
    for (0..32) |input| {
        const weight_index = (bucket * 32 + input) * 32;
        const weight_ptr: *align(1) const V7Dense2WeightVec = @ptrCast(&l2_weights[weight_index]);
        const input_vec: V7Dense2Vec = @splat(dense1_activated[input]);
        l2_sums += input_vec * @as(V7Dense2Vec, @intCast(weight_ptr.*));
    }

    var dense2_activated: [32]i32 = undefined;
    for (0..32) |output| {
        const value = divRoundNearestSigned(l2_sums[output], q);
        const clipped = @min(@max(value, 0), q);
        dense2_activated[output] = @intCast(divRoundNearestNonNeg(clipped * clipped, q));
    }

    const activation_vec: V7Dense2Vec = dense2_activated;
    const output_weight_ptr: *align(1) const V7Dense2WeightVec = @ptrCast(&output_weights[bucket * 32]);
    const products = activation_vec * @as(V7Dense2Vec, @intCast(output_weight_ptr.*));
    const sum = @as(i64, output_biases[bucket]) + @as(i64, @reduce(.Add, products));
    return @intCast(divRoundNearestSigned(sum * net.scale, q * q));
}

fn evaluatePairwiseMlpFromAccumulators(
    net: *const Network,
    acc: anytype,
    b: *Board,
) i32 {
    const stm_is_white = b.board.move == .white;
    const hidden_size: usize = @intCast(net.ft_hidden_size);
    const half = hidden_size / 2;
    const dense1_size: usize = @intCast(net.dense1_size);
    const dense2_size: usize = @intCast(net.dense2_size);
    const q: i64 = net.q;
    const pool_quant: i64 = net.pool_quant;
    const bucket = outputBucket(net, b);

    const us_acc = if (stm_is_white) acc.white[0..hidden_size] else acc.black[0..hidden_size];
    const them_acc = if (stm_is_white) acc.black[0..hidden_size] else acc.white[0..hidden_size];
    const AccVec = AccVecOf(@TypeOf(us_acc));
    var pooled: [MAX_HIDDEN_SIZE]i32 = undefined;
    var index: usize = 0;
    const pool_divisor: I32Vec = @splat(512);
    while (index + SIMD_LANES <= half) : (index += SIMD_LANES) {
        const us_a_ptr: *align(1) const AccVec = @ptrCast(&us_acc[index]);
        const us_b_ptr: *align(1) const AccVec = @ptrCast(&us_acc[index + half]);
        const them_a_ptr: *align(1) const AccVec = @ptrCast(&them_acc[index]);
        const them_b_ptr: *align(1) const AccVec = @ptrCast(&them_acc[index + half]);
        const us_output_ptr: *align(1) I32Vec = @ptrCast(&pooled[index]);
        const them_output_ptr: *align(1) I32Vec = @ptrCast(&pooled[index + half]);
        const us_a = clampVecToActivationRange(@intCast(us_a_ptr.*), net.q0);
        const us_b = clampVecToActivationRange(@intCast(us_b_ptr.*), net.q0);
        const them_a = clampVecToActivationRange(@intCast(them_a_ptr.*), net.q0);
        const them_b = clampVecToActivationRange(@intCast(them_b_ptr.*), net.q0);
        us_output_ptr.* = @divTrunc(us_a * us_b, pool_divisor);
        them_output_ptr.* = @divTrunc(them_a * them_b, pool_divisor);
    }
    while (index < half) : (index += 1) {
        const us_a = clampToActivationRange(us_acc[index], net.q0);
        const us_b = clampToActivationRange(us_acc[index + half], net.q0);
        const them_a = clampToActivationRange(them_acc[index], net.q0);
        const them_b = clampToActivationRange(them_acc[index + half], net.q0);
        pooled[index] = @divTrunc(us_a * us_b, 512);
        pooled[index + half] = @divTrunc(them_a * them_b, 512);
    }

    if (dense1_size == 16 and dense2_size == 32) {
        return evaluatePairwiseMlpTail16x32(net, pooled[0..hidden_size], bucket);
    }

    const l1_biases = net.l1_biases orelse unreachable;
    const l1_weights = net.l1_weights orelse unreachable;
    var dense1_activated: [2 * MAX_DENSE_SIZE]i32 = undefined;
    for (0..dense1_size) |output| {
        var sum: i64 = l1_biases[bucket * dense1_size + output];
        for (0..hidden_size) |input| {
            const weight_index = (bucket * hidden_size + input) * dense1_size + output;
            sum += @as(i64, pooled[input]) * @as(i64, l1_weights[weight_index]);
        }
        const value = divRoundNearestSigned(sum, pool_quant);
        dense1_activated[output] = @intCast(@min(@max(value, 0), q));
        const squared = divRoundNearestNonNeg(value * value, q);
        dense1_activated[dense1_size + output] = @intCast(@min(squared, q));
    }

    const l2_biases = net.l2_biases orelse unreachable;
    const l2_weights = net.l2_weights orelse unreachable;
    var dense2_activated: [MAX_DENSE_SIZE]i32 = undefined;
    for (0..dense2_size) |output| {
        var sum: i64 = l2_biases[bucket * dense2_size + output];
        for (0..2 * dense1_size) |input| {
            const weight_index = (bucket * 2 * dense1_size + input) * dense2_size + output;
            sum += @as(i64, dense1_activated[input]) * @as(i64, l2_weights[weight_index]);
        }
        const value = divRoundNearestSigned(sum, q);
        const clipped = @min(@max(value, 0), q);
        dense2_activated[output] = @intCast(divRoundNearestNonNeg(clipped * clipped, q));
    }

    const output_biases = net.v7_output_biases orelse unreachable;
    const output_weights = net.v7_output_weights orelse unreachable;
    var sum: i64 = output_biases[bucket];
    for (0..dense2_size) |input| {
        sum += @as(i64, dense2_activated[input]) *
            @as(i64, output_weights[bucket * dense2_size + input]);
    }
    return @intCast(divRoundNearestSigned(sum * net.scale, q * q));
}

/// Evaluate using pre-computed accumulators (activation + head only).
/// Returns score from the side-to-move perspective. `acc` may point at
/// `AccumulatorPair` or `AccumulatorPairWide`; results are bit-identical
/// for any net that passes the load-time accumulator bound check.
pub fn evaluateFromAccumulators(
    net: *const Network,
    acc: anytype,
    b: *Board,
) i32 {
    return switch (net.architecture) {
        .legacy_linear => evaluateLinearFromAccumulators(net, acc, b),
        .pairwise_mlp => evaluatePairwiseMlpFromAccumulators(net, acc, b),
    };
}

/// Returns score from the side-to-move perspective, same convention as classical eval.
/// This is the full-recompute path (non-incremental). Kept for fallback/gensfen use.
pub fn evaluate(net: *const Network, b: *Board) i32 {
    const acc = initAccumulators(net, b);
    return evaluateFromAccumulators(net, &acc, b);
}
