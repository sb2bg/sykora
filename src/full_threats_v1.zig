const std = @import("std");
const bitboard = @import("bitboard.zig");
const BitBoard = bitboard.BitBoard;
const piece = @import("piece.zig");

pub const scheme_id: u16 = 1;
pub const feature_count: usize = 60_720;
pub const features_per_attacker_colour: usize = 30_360;
pub const max_active: usize = 240;
pub const packing_hash = [32]u8{
    0x96, 0x45, 0x91, 0xed, 0xbe, 0x85, 0x6c, 0x9f,
    0x90, 0x69, 0x4d, 0xcb, 0xfa, 0xbe, 0x42, 0xd5,
    0x8b, 0x01, 0x1a, 0x46, 0x9e, 0x32, 0x75, 0xa8,
    0xaa, 0xa9, 0xe4, 0x24, 0x9b, 0x21, 0x98, 0x8a,
};

const geometry_counts = [5]usize{ 132, 336, 560, 896, 1456 };
const target_slot_counts = [5]usize{ 6, 10, 8, 8, 10 };
const type_bases = [5]usize{ 0, 792, 4152, 8632, 15800 };

pub const RelativeColour = enum(u1) {
    ours = 0,
    theirs = 1,
};

const GeometryIndices = [2][5][64][64]i16;
const GeometryPair = struct { source: u8 = 0, target: u8 = 0 };
const GeometryPairs = [2][5][1456]GeometryPair;

fn validGeometry(
    attacker_colour: RelativeColour,
    attacker_type: piece.Type,
    source: usize,
    target: usize,
) bool {
    if (source == target) return false;
    const source_rank: i8 = @intCast(source / 8);
    const source_file: i8 = @intCast(source % 8);
    const target_rank: i8 = @intCast(target / 8);
    const target_file: i8 = @intCast(target % 8);
    const dr = target_rank - source_rank;
    const df = target_file - source_file;
    const adr: u8 = @intCast(@abs(dr));
    const adf: u8 = @intCast(@abs(df));

    return switch (attacker_type) {
        .pawn => blk: {
            if (source_rank == 0 or source_rank == 7) break :blk false;
            const forward: i8 = if (attacker_colour == .ours) 1 else -1;
            break :blk dr == forward and adf <= 1;
        },
        .knight => (adr == 1 and adf == 2) or (adr == 2 and adf == 1),
        .bishop => adr == adf,
        .rook => dr == 0 or df == 0,
        .queen => adr == adf or dr == 0 or df == 0,
        .king => false,
    };
}

fn initGeometryIndices() GeometryIndices {
    @setEvalBranchQuota(2_000_000);
    var result = [_][5][64][64]i16{[_][64][64]i16{[_][64]i16{[_]i16{-1} ** 64} ** 64} ** 5} ** 2;
    for (0..2) |colour_index| {
        const colour: RelativeColour = @enumFromInt(colour_index);
        for (0..5) |type_index| {
            const attacker_type: piece.Type = @enumFromInt(type_index);
            var geometry: i16 = 0;
            for (0..64) |source| {
                for (0..64) |target| {
                    if (!validGeometry(colour, attacker_type, source, target)) continue;
                    result[colour_index][type_index][source][target] = geometry;
                    geometry += 1;
                }
            }
            if (geometry != geometry_counts[type_index]) {
                @compileError("full_threats_v1 geometry count mismatch");
            }
        }
    }
    return result;
}

const geometry_indices = initGeometryIndices();

fn initGeometryPairs() GeometryPairs {
    @setEvalBranchQuota(2_000_000);
    var result = [_][5][1456]GeometryPair{[_][1456]GeometryPair{[_]GeometryPair{.{}} ** 1456} ** 5} ** 2;
    for (0..2) |colour_index| {
        const colour: RelativeColour = @enumFromInt(colour_index);
        for (0..5) |type_index| {
            const attacker_type: piece.Type = @enumFromInt(type_index);
            var geometry: usize = 0;
            for (0..64) |source| {
                for (0..64) |target| {
                    if (!validGeometry(colour, attacker_type, source, target)) continue;
                    result[colour_index][type_index][geometry] = .{
                        .source = @intCast(source),
                        .target = @intCast(target),
                    };
                    geometry += 1;
                }
            }
        }
    }
    return result;
}

const geometry_pairs = initGeometryPairs();

fn targetTypeRank(attacker_type: piece.Type, target_type: piece.Type) ?usize {
    return switch (attacker_type) {
        .pawn => switch (target_type) {
            .pawn => 0,
            .knight => 1,
            .rook => 2,
            else => null,
        },
        .knight, .queen => switch (target_type) {
            .pawn => 0,
            .knight => 1,
            .bishop => 2,
            .rook => 3,
            .queen => 4,
            else => null,
        },
        .bishop, .rook => switch (target_type) {
            .pawn => 0,
            .knight => 1,
            .bishop => 2,
            .rook => 3,
            else => null,
        },
        .king => null,
    };
}

fn targetTypeFromRank(attacker_type: piece.Type, rank: usize) piece.Type {
    return switch (attacker_type) {
        .pawn => ([_]piece.Type{ .pawn, .knight, .rook })[rank],
        .knight, .queen => ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen })[rank],
        .bishop, .rook => ([_]piece.Type{ .pawn, .knight, .bishop, .rook })[rank],
        .king => unreachable,
    };
}

fn structurallyReachable(
    attacker_colour: RelativeColour,
    attacker_type: piece.Type,
    source: u8,
    target: u8,
    target_colour: RelativeColour,
    target_type: piece.Type,
) bool {
    if (target_type == .pawn and (target / 8 == 0 or target / 8 == 7)) return false;
    if (attacker_type == target_type) {
        const enemy_pair = attacker_colour != target_colour;
        const friendly_symmetric = attacker_colour == target_colour and attacker_type != .pawn;
        if ((enemy_pair or friendly_symmetric) and source >= target) return false;
    }
    return true;
}

pub fn encode(
    attacker_colour: RelativeColour,
    attacker_type: piece.Type,
    source: u8,
    target: u8,
    target_colour: RelativeColour,
    target_type: piece.Type,
) ?u16 {
    if (attacker_type == .king or target_type == .king) return null;
    const type_index: usize = @intFromEnum(attacker_type);
    const geometry = geometry_indices[@intFromEnum(attacker_colour)][type_index][source][target];
    if (geometry < 0) return null;
    const type_rank = targetTypeRank(attacker_type, target_type) orelse return null;
    if (!structurallyReachable(
        attacker_colour,
        attacker_type,
        source,
        target,
        target_colour,
        target_type,
    )) return null;

    const types_per_colour = target_slot_counts[type_index] / 2;
    const slot = @as(usize, @intFromEnum(target_colour)) * types_per_colour + type_rank;
    const feature = @as(usize, @intFromEnum(attacker_colour)) * features_per_attacker_colour +
        type_bases[type_index] + @as(usize, @intCast(geometry)) * target_slot_counts[type_index] + slot;
    std.debug.assert(feature < feature_count);
    return @intCast(feature);
}

pub fn isReachableId(feature: u16) bool {
    const feature_index: usize = feature;
    if (feature_index >= feature_count) return false;
    const attacker_colour_index = feature_index / features_per_attacker_colour;
    const attacker_colour: RelativeColour = @enumFromInt(attacker_colour_index);
    const local = feature_index % features_per_attacker_colour;
    for (0..5) |type_index| {
        const region_size = geometry_counts[type_index] * target_slot_counts[type_index];
        if (local < type_bases[type_index] or local >= type_bases[type_index] + region_size) continue;
        const attacker_type: piece.Type = @enumFromInt(type_index);
        const within = local - type_bases[type_index];
        const geometry = within / target_slot_counts[type_index];
        const slot = within % target_slot_counts[type_index];
        const types_per_colour = target_slot_counts[type_index] / 2;
        const target_colour: RelativeColour = @enumFromInt(slot / types_per_colour);
        const target_type = targetTypeFromRank(attacker_type, slot % types_per_colour);
        const pair = geometry_pairs[attacker_colour_index][type_index][geometry];
        return structurallyReachable(
            attacker_colour,
            attacker_type,
            pair.source,
            pair.target,
            target_colour,
            target_type,
        );
    }
    unreachable;
}

fn orientedSquare(square: u8, perspective: piece.Color, king_square: u8) u8 {
    var oriented = if (perspective == .white) square else square ^ 56;
    const oriented_king = if (perspective == .white) king_square else king_square ^ 56;
    if (oriented_king % 8 > 3) oriented ^= 7;
    return oriented;
}

fn relativeColour(actual: piece.Color, perspective: piece.Color) RelativeColour {
    return if (actual == perspective) .ours else .theirs;
}

const PieceAt = struct {
    colour: piece.Color,
    kind: piece.Type,
};

fn pieceAt(state: *const BitBoard, square: u8) ?PieceAt {
    const bit = @as(u64, 1) << @intCast(square);
    const colour: piece.Color = if (state.getColorBitboard(.white) & bit != 0)
        .white
    else if (state.getColorBitboard(.black) & bit != 0)
        .black
    else
        return null;
    inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen, .king }) |kind| {
        if (state.getKindBitboard(kind) & bit != 0) return .{ .colour = colour, .kind = kind };
    }
    return null;
}

fn occupiedTargets(state: *const BitBoard, square: u8, attacker: PieceAt) u64 {
    const occupied = state.occupied();
    const sq: u6 = @intCast(square);
    return switch (attacker.kind) {
        .pawn => blk: {
            var result = bitboard.getPawnAttacks(sq, attacker.colour) & occupied;
            const step: i16 = if (attacker.colour == .white) 8 else -8;
            const forward: i16 = @as(i16, square) + step;
            if (forward >= 0 and forward < 64) {
                const forward_bit = @as(u64, 1) << @intCast(forward);
                if (occupied & forward_bit != 0) result |= forward_bit;
            }
            break :blk result;
        },
        .knight => bitboard.getKnightAttacks(sq) & occupied,
        .bishop => bitboard.getBishopAttacks(sq, occupied) & occupied,
        .rook => bitboard.getRookAttacks(sq, occupied) & occupied,
        .queen => bitboard.getQueenAttacks(sq, occupied) & occupied,
        .king => 0,
    };
}

fn insertionSort(values: []u16) void {
    var index: usize = 1;
    while (index < values.len) : (index += 1) {
        const value = values[index];
        var insert = index;
        while (insert > 0 and values[insert - 1] > value) : (insert -= 1) {
            values[insert] = values[insert - 1];
        }
        values[insert] = value;
    }
}

/// Enumerate sorted, unique active IDs. The 240-entry bound follows from at
/// most 30 non-king attackers and at most eight occupied targets per attacker.
pub fn enumerate(
    state: *const BitBoard,
    perspective: piece.Color,
    storage: *[max_active]u16,
) []const u16 {
    const king_bb = state.getColorBitboard(perspective) & state.getKindBitboard(.king);
    std.debug.assert(king_bb != 0);
    const king_square: u8 = @intCast(@ctz(king_bb));
    var count: usize = 0;

    inline for ([_]piece.Color{ .white, .black }) |attacker_colour| {
        const colour_bb = state.getColorBitboard(attacker_colour);
        inline for ([_]piece.Type{ .pawn, .knight, .bishop, .rook, .queen }) |attacker_type| {
            var attackers = colour_bb & state.getKindBitboard(attacker_type);
            while (attackers != 0) {
                const source: u8 = @intCast(@ctz(attackers));
                attackers &= attackers - 1;
                const attacker = PieceAt{ .colour = attacker_colour, .kind = attacker_type };
                var targets = occupiedTargets(state, source, attacker);
                while (targets != 0) {
                    const target: u8 = @intCast(@ctz(targets));
                    targets &= targets - 1;
                    const target_piece = pieceAt(state, target) orelse unreachable;
                    if (target_piece.kind == .king) continue;
                    const feature = encode(
                        relativeColour(attacker_colour, perspective),
                        attacker_type,
                        orientedSquare(source, perspective, king_square),
                        orientedSquare(target, perspective, king_square),
                        relativeColour(target_piece.colour, perspective),
                        target_piece.kind,
                    ) orelse continue;
                    std.debug.assert(count < storage.len);
                    storage[count] = feature;
                    count += 1;
                }
            }
        }
    }

    insertionSort(storage[0..count]);
    if (count == 0) return storage[0..0];
    var unique: usize = 1;
    for (storage[1..count]) |value| {
        if (value == storage[unique - 1]) continue;
        storage[unique] = value;
        unique += 1;
    }
    return storage[0..unique];
}

test "full_threats_v1 start position golden IDs" {
    var board = try bitboard.Board.fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    var white_storage: [max_active]u16 = undefined;
    var black_storage: [max_active]u16 = undefined;
    const white = enumerate(&board.board, .white, &white_storage);
    const black = enumerate(&board.board, .black, &black_storage);
    try std.testing.expectEqual(@as(usize, 28), white.len);
    try std.testing.expectEqualSlices(u16, white, black);
    try std.testing.expectEqualSlices(
        u16,
        &.{ 812, 1002, 4264, 4272, 4432, 4440, 8633, 8688 },
        white[0..8],
    );
}

test "full_threats_v1 covers colour orientation and deduplicates symmetric pairs" {
    var board = try bitboard.Board.fromFen("8/8/3q4/2b1r3/3N4/2P1P3/4R3/4K2k w - - 0 1");
    var white_storage: [max_active]u16 = undefined;
    var black_storage: [max_active]u16 = undefined;
    const white = enumerate(&board.board, .white, &white_storage);
    const black = enumerate(&board.board, .black, &black_storage);
    try std.testing.expectEqual(@as(usize, 10), white.len);
    try std.testing.expectEqual(@as(usize, 10), black.len);
    try std.testing.expectEqualSlices(u16, &.{ 193, 217, 2215, 9928 }, white[0..4]);
    try std.testing.expectEqualSlices(u16, &.{ 6221, 11714, 11748, 20353 }, black[0..4]);
}

test "full_threats_v1 frozen space has 51130 reachable IDs" {
    var reachable: usize = 0;
    for (0..feature_count) |feature| {
        if (isReachableId(@intCast(feature))) reachable += 1;
    }
    try std.testing.expectEqual(@as(usize, 51_130), reachable);
}
