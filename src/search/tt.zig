const std = @import("std");
const board = @import("../bitboard.zig");
const Move = board.Move;

pub const TTEntryBound = enum(u2) {
    exact,
    lower_bound,
    upper_bound,
};

const TTEntryMetadata = packed struct(u8) {
    bound: TTEntryBound,
    age: u6,
};

const NO_STATIC_EVAL: i16 = std.math.minInt(i16);

pub const TTEntry = struct {
    hash: u64,
    score: i32,
    best_move: Move,
    depth: u8,
    metadata: TTEntryMetadata,
    static_eval: i16,

    pub fn init() TTEntry {
        return TTEntry{
            .hash = 0,
            .score = 0,
            .best_move = Move.init(0, 0, null),
            .depth = 0,
            .metadata = .{ .bound = .exact, .age = 0 },
            .static_eval = NO_STATIC_EVAL,
        };
    }

    pub inline fn bound(self: TTEntry) TTEntryBound {
        return self.metadata.bound;
    }

    pub inline fn staticEval(self: TTEntry) ?i32 {
        return if (self.static_eval == NO_STATIC_EVAL) null else @as(i32, self.static_eval);
    }
};

const BUCKET_SIZE: usize = 4;

const AtomicTTEntry = struct {
    // Storing hash XOR payload lets a racing reader validate that both atomic
    // words belong to the same snapshot. Mixed snapshots become misses.
    hash_xor_data: std.atomic.Value(u64),
    data: std.atomic.Value(u64),

    fn init() AtomicTTEntry {
        return .{
            .hash_xor_data = std.atomic.Value(u64).init(0),
            .data = std.atomic.Value(u64).init(0),
        };
    }
};

const TTBucket = struct {
    entries: [BUCKET_SIZE]AtomicTTEntry,

    fn init() TTBucket {
        return TTBucket{
            .entries = [_]AtomicTTEntry{AtomicTTEntry.init()} ** BUCKET_SIZE,
        };
    }
};

comptime {
    // Four entries fit exactly in one 64-byte cache line. Keep these assertions
    // beside the layout so adding a field cannot silently undo that property.
    std.debug.assert(@sizeOf(AtomicTTEntry) == 16);
    std.debug.assert(@sizeOf(TTBucket) == 64);
}

pub const TranspositionTable = struct {
    const Self = @This();

    buckets: []TTBucket,
    num_buckets: usize,
    bucket_divisor_reciprocal: u64,
    bucket_mask: usize,
    current_age: u6,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size_mb: usize) !Self {
        const bucket_size = @sizeOf(TTBucket);
        const num_buckets = (size_mb * 1024 * 1024) / bucket_size;
        const bucket_divisor_reciprocal = divisorReciprocal(@intCast(num_buckets));
        const bucket_mask = if (std.math.isPowerOfTwo(num_buckets)) num_buckets - 1 else 0;
        const buckets = try allocator.alloc(TTBucket, num_buckets);

        for (buckets) |*bucket| {
            bucket.* = TTBucket.init();
        }

        return Self{
            .buckets = buckets,
            .num_buckets = num_buckets,
            .bucket_divisor_reciprocal = bucket_divisor_reciprocal,
            .bucket_mask = bucket_mask,
            .current_age = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buckets);
    }

    pub fn clear(self: *Self) void {
        for (self.buckets) |*bucket| {
            bucket.* = TTBucket.init();
        }
        self.current_age = 0;
    }

    pub fn nextAge(self: *Self) void {
        self.current_age +%= 1;
    }

    /// Retained for the UCI/search interface; entries are always lock-free.
    pub fn setConcurrent(self: *Self, concurrent: bool) void {
        _ = self;
        _ = concurrent;
    }

    pub fn resize(self: *Self, new_size_mb: usize) !void {
        self.allocator.free(self.buckets);
        const bucket_size = @sizeOf(TTBucket);
        const num_buckets = (new_size_mb * 1024 * 1024) / bucket_size;
        const buckets = try self.allocator.alloc(TTBucket, num_buckets);
        for (buckets) |*bucket| {
            bucket.* = TTBucket.init();
        }
        self.buckets = buckets;
        self.num_buckets = num_buckets;
        self.bucket_divisor_reciprocal = divisorReciprocal(@intCast(num_buckets));
        self.bucket_mask = if (std.math.isPowerOfTwo(num_buckets)) num_buckets - 1 else 0;
        self.current_age = 0;
    }

    inline fn bucketIndex(self: *Self, hash: u64) usize {
        if (self.bucket_mask != 0) {
            return @intCast(hash & self.bucket_mask);
        }
        return @intCast(fastModulo(
            hash,
            @intCast(self.num_buckets),
            self.bucket_divisor_reciprocal,
        ));
    }

    pub fn probe(self: *Self, hash: u64) ?TTEntry {
        const idx = self.bucketIndex(hash);
        return probeBucket(&self.buckets[idx], hash);
    }

    inline fn probeBucket(bucket: *const TTBucket, hash: u64) ?TTEntry {
        for (&bucket.entries) |entry| {
            const data = entry.data.load(.monotonic);
            const hash_xor_data = entry.hash_xor_data.load(.monotonic);
            if (hash_xor_data ^ data == hash) {
                return unpackEntry(hash, data);
            }
        }
        return null;
    }

    pub fn store(
        self: *Self,
        hash: u64,
        depth: u8,
        score: i32,
        bound: TTEntryBound,
        best_move: Move,
        static_eval: ?i32,
    ) void {
        const idx = self.bucketIndex(hash);
        self.storeInBucket(&self.buckets[idx], hash, depth, score, bound, best_move, static_eval);
    }

    inline fn storeInBucket(
        self: *Self,
        bucket: *TTBucket,
        hash: u64,
        depth: u8,
        score: i32,
        bound: TTEntryBound,
        best_move: Move,
        static_eval: ?i32,
    ) void {
        // Check if this hash already exists in the bucket — always update same-hash entry
        for (&bucket.entries) |*entry| {
            const current = loadEntry(entry);
            if (current.hash == hash) {
                var updated = current;
                if (depth >= current.depth or current.metadata.age != self.current_age) {
                    updated.depth = depth;
                    updated.score = score;
                    updated.metadata = .{ .bound = bound, .age = self.current_age };
                }
                if (best_move.from() != 0 or best_move.to() != 0) {
                    updated.best_move = best_move;
                }
                if (static_eval) |value| {
                    updated.static_eval = encodeStaticEval(value);
                }
                writeEntry(entry, updated);
                return;
            }
        }

        // Find best replacement victim: prefer empty → stale age → shallowest depth
        var victim_idx: usize = 0;
        var victim_score: i32 = replacementScore(loadEntry(&bucket.entries[0]), self.current_age);
        for (1..BUCKET_SIZE) |i| {
            const s = replacementScore(loadEntry(&bucket.entries[i]), self.current_age);
            if (s < victim_score) {
                victim_score = s;
                victim_idx = i;
            }
        }

        writeEntry(&bucket.entries[victim_idx], .{
            .hash = hash,
            .score = score,
            .best_move = best_move,
            .depth = depth,
            .metadata = .{ .bound = bound, .age = self.current_age },
            .static_eval = encodeStaticEval(static_eval),
        });
    }

    /// Lower score = more replaceable. Empty slots get lowest score.
    inline fn replacementScore(entry: TTEntry, current_age: u6) i32 {
        if (entry.hash == 0) return -1000; // Empty — most replaceable
        var score: i32 = @as(i32, entry.depth);
        if (entry.metadata.age != current_age) score -= 256; // Stale — very replaceable
        return score;
    }
};

inline fn encodeStaticEval(value: ?i32) i16 {
    const score = value orelse return NO_STATIC_EVAL;
    if (score <= NO_STATIC_EVAL or score > std.math.maxInt(i16)) return NO_STATIC_EVAL;
    return @intCast(score);
}

inline fn packEntryData(entry: TTEntry) u64 {
    std.debug.assert(entry.score >= std.math.minInt(i16) and entry.score <= std.math.maxInt(i16));
    const score: i16 = @intCast(entry.score);
    const score_bits: u16 = @bitCast(score);
    const metadata: u8 = @bitCast(entry.metadata);
    const static_eval_bits: u16 = @bitCast(entry.static_eval);
    return @as(u64, score_bits) |
        (@as(u64, entry.best_move.data) << 16) |
        (@as(u64, entry.depth) << 32) |
        (@as(u64, metadata) << 40) |
        (@as(u64, static_eval_bits) << 48);
}

inline fn unpackEntry(hash: u64, data: u64) TTEntry {
    const score_bits: u16 = @truncate(data);
    const score: i16 = @bitCast(score_bits);
    const metadata_bits: u8 = @truncate(data >> 40);
    const static_eval_bits: u16 = @truncate(data >> 48);
    return .{
        .hash = hash,
        .score = score,
        .best_move = .{ .data = @truncate(data >> 16) },
        .depth = @truncate(data >> 32),
        .metadata = @bitCast(metadata_bits),
        .static_eval = @bitCast(static_eval_bits),
    };
}

inline fn loadEntry(entry: *const AtomicTTEntry) TTEntry {
    const data = entry.data.load(.monotonic);
    const hash_xor_data = entry.hash_xor_data.load(.monotonic);
    return unpackEntry(hash_xor_data ^ data, data);
}

inline fn writeEntry(entry: *AtomicTTEntry, value: TTEntry) void {
    const data = packEntryData(value);
    entry.data.store(data, .monotonic);
    entry.hash_xor_data.store(value.hash ^ data, .release);
}

inline fn divisorReciprocal(divisor: u64) u64 {
    if (divisor <= 1) return 0;
    return @intCast((@as(u128, 1) << 64) / divisor);
}

inline fn fastModulo(value: u64, divisor: u64, reciprocal: u64) u64 {
    if (divisor == 1) return 0;

    const quotient: u64 = @truncate((@as(u128, value) * reciprocal) >> 64);
    var remainder = value - quotient * divisor;
    if (remainder >= divisor) remainder -= divisor;
    return remainder;
}

test "packed TT payload round-trips every field" {
    const original = TTEntry{
        .hash = 0xD4E1_9A72_5BC3_08F6,
        .score = -32_000,
        .best_move = Move.init(7, 63, .queen),
        .depth = 255,
        .metadata = .{ .bound = .lower_bound, .age = 63 },
        .static_eval = 2345,
    };
    const decoded = unpackEntry(original.hash, packEntryData(original));

    try std.testing.expectEqual(original.hash, decoded.hash);
    try std.testing.expectEqual(original.score, decoded.score);
    try std.testing.expectEqual(original.best_move.data, decoded.best_move.data);
    try std.testing.expectEqual(original.depth, decoded.depth);
    try std.testing.expectEqual(original.metadata.bound, decoded.metadata.bound);
    try std.testing.expectEqual(original.metadata.age, decoded.metadata.age);
    try std.testing.expectEqual(original.static_eval, decoded.static_eval);
}

test "lock-free TT stores and probes packed entries" {
    var tt = try TranspositionTable.init(std.testing.allocator, 1);
    defer tt.deinit();

    const hash: u64 = 0x1234_5678_9ABC_DEF0;
    const best_move = Move.init(12, 28, .knight);
    tt.store(hash, 17, -1234, .upper_bound, best_move, 321);

    const entry = tt.probe(hash).?;
    try std.testing.expectEqual(hash, entry.hash);
    try std.testing.expectEqual(@as(u8, 17), entry.depth);
    try std.testing.expectEqual(@as(i32, -1234), entry.score);
    try std.testing.expectEqual(TTEntryBound.upper_bound, entry.bound());
    try std.testing.expectEqual(best_move.data, entry.best_move.data);
    try std.testing.expectEqual(@as(?i32, 321), entry.staticEval());
    try std.testing.expect(tt.probe(hash ^ 1) == null);
}

test "fast modulo matches integer remainder" {
    const divisors = [_]u64{
        1,
        2,
        3,
        7,
        64,
        97,
        10_922,
        1_398_101,
        std.math.maxInt(u32),
        std.math.maxInt(u64) - 1,
        std.math.maxInt(u64),
    };
    const values = [_]u64{
        0,
        1,
        2,
        63,
        64,
        65,
        std.math.maxInt(u32),
        std.math.maxInt(u64) / 2,
        std.math.maxInt(u64) - 1,
        std.math.maxInt(u64),
    };

    for (divisors) |divisor| {
        const reciprocal = divisorReciprocal(divisor);
        for (values) |value| {
            try std.testing.expectEqual(value % divisor, fastModulo(value, divisor, reciprocal));
        }
    }

    var prng = std.Random.DefaultPrng.init(0x5A17_C0DE);
    const random = prng.random();
    for (0..10_000) |_| {
        const divisor = random.intRangeAtMost(u64, 1, std.math.maxInt(u32));
        const value = random.int(u64);
        const reciprocal = divisorReciprocal(divisor);
        try std.testing.expectEqual(value % divisor, fastModulo(value, divisor, reciprocal));
    }
}
