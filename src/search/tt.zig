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

pub const TranspositionTable = struct {
    const Self = @This();

    entries: []TTEntry,
    size: usize,
    current_age: u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size_mb: usize) !Self {
        const entry_size = @sizeOf(TTEntry);
        const num_entries = (size_mb * 1024 * 1024) / entry_size;
        const entries = try allocator.alloc(TTEntry, num_entries);

        for (entries) |*entry| {
            entry.* = TTEntry.init();
        }

        return Self{
            .entries = entries,
            .size = num_entries,
            .current_age = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.entries);
    }

    pub fn clear(self: *Self) void {
        for (self.entries) |*entry| {
            entry.* = TTEntry.init();
        }
        self.current_age = 0;
    }

    pub fn nextAge(self: *Self) void {
        self.current_age +%= 1;
    }

    pub fn resize(self: *Self, new_size_mb: usize) !void {
        self.allocator.free(self.entries);
        const entry_size = @sizeOf(TTEntry);
        const num_entries = (new_size_mb * 1024 * 1024) / entry_size;
        const entries = try self.allocator.alloc(TTEntry, num_entries);
        for (entries) |*entry| {
            entry.* = TTEntry.init();
        }
        self.entries = entries;
        self.size = num_entries;
        self.current_age = 0;
    }

    pub fn index(self: *Self, hash: u64) usize {
        return @as(usize, @intCast(hash % @as(u64, @intCast(self.size))));
    }

    pub fn probe(self: *Self, hash: u64) ?*TTEntry {
        const idx = self.index(hash);
        if (self.entries[idx].hash == hash) {
            return &self.entries[idx];
        }
        return null;
    }

    pub fn store(self: *Self, hash: u64, depth: u8, score: i32, bound: TTEntryBound, best_move: Move) void {
        const idx = self.index(hash);
        const entry = &self.entries[idx];

        const replace = entry.hash == 0 or
            entry.age != self.current_age or
            (entry.hash == hash and depth >= entry.depth) or
            (entry.hash != hash and depth > entry.depth);

        if (replace) {
            entry.hash = hash;
            entry.depth = depth;
            entry.score = score;
            entry.bound = bound;
            entry.best_move = best_move;
            entry.age = self.current_age;
        } else if (entry.hash == hash and best_move.from() != 0) {
            entry.best_move = best_move;
        }
    }
};
