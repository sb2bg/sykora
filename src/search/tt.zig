const std = @import("std");
const board = @import("../bitboard.zig");
const Move = board.Move;

pub const TTEntryBound = enum(u8) {
    exact,
    lower_bound,
    upper_bound,
};

pub const TTEntry = struct {
    hash: u64,
    depth: u8,
    score: i32,
    bound: TTEntryBound,
    best_move: Move,
    age: u8,

    pub fn init() TTEntry {
        return TTEntry{
            .hash = 0,
            .depth = 0,
            .score = 0,
            .bound = .exact,
            .best_move = Move.init(0, 0, null),
            .age = 0,
        };
    }
};

const BUCKET_SIZE: usize = 4;

const TTBucket = struct {
    entries: [BUCKET_SIZE]TTEntry,

    fn init() TTBucket {
        return TTBucket{
            .entries = [_]TTEntry{TTEntry.init()} ** BUCKET_SIZE,
        };
    }
};

pub const TranspositionTable = struct {
    const Self = @This();

    buckets: []TTBucket,
    num_buckets: usize,
    current_age: u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size_mb: usize) !Self {
        const bucket_size = @sizeOf(TTBucket);
        const num_buckets = (size_mb * 1024 * 1024) / bucket_size;
        const buckets = try allocator.alloc(TTBucket, num_buckets);

        for (buckets) |*bucket| {
            bucket.* = TTBucket.init();
        }

        return Self{
            .buckets = buckets,
            .num_buckets = num_buckets,
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
        self.current_age = 0;
    }

    inline fn bucketIndex(self: *Self, hash: u64) usize {
        return @as(usize, @intCast(hash % @as(u64, @intCast(self.num_buckets))));
    }

    pub fn probe(self: *Self, hash: u64) ?*TTEntry {
        const idx = self.bucketIndex(hash);
        const bucket = &self.buckets[idx];
        for (&bucket.entries) |*entry| {
            if (entry.hash == hash) {
                return entry;
            }
        }
        return null;
    }

    pub fn store(self: *Self, hash: u64, depth: u8, score: i32, bound: TTEntryBound, best_move: Move) void {
        const idx = self.bucketIndex(hash);
        const bucket = &self.buckets[idx];

        // Check if this hash already exists in the bucket — always update same-hash entry
        for (&bucket.entries) |*entry| {
            if (entry.hash == hash) {
                if (depth >= entry.depth or entry.age != self.current_age) {
                    entry.depth = depth;
                    entry.score = score;
                    entry.bound = bound;
                    entry.age = self.current_age;
                }
                if (best_move.from() != 0 or best_move.to() != 0) {
                    entry.best_move = best_move;
                }
                return;
            }
        }

        // Find best replacement victim: prefer empty → stale age → shallowest depth
        var victim_idx: usize = 0;
        var victim_score: i32 = replacementScore(&bucket.entries[0], self.current_age);
        for (1..BUCKET_SIZE) |i| {
            const s = replacementScore(&bucket.entries[i], self.current_age);
            if (s < victim_score) {
                victim_score = s;
                victim_idx = i;
            }
        }

        const victim = &bucket.entries[victim_idx];
        victim.hash = hash;
        victim.depth = depth;
        victim.score = score;
        victim.bound = bound;
        victim.best_move = best_move;
        victim.age = self.current_age;
    }

    /// Lower score = more replaceable. Empty slots get lowest score.
    inline fn replacementScore(entry: *const TTEntry, current_age: u8) i32 {
        if (entry.hash == 0) return -1000; // Empty — most replaceable
        var score: i32 = @as(i32, entry.depth);
        if (entry.age != current_age) score -= 256; // Stale — very replaceable
        return score;
    }
};
