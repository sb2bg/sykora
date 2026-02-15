const pieceInfo = @import("../piece.zig");

const KNIGHT_ATTACKS = initKnightAttacks();
const KING_ATTACKS = initKingAttacks();
const WHITE_PAWN_ATTACKS = initWhitePawnAttacks();
const BLACK_PAWN_ATTACKS = initBlackPawnAttacks();

// Magic bitboard constants for rook attacks
const ROOK_MAGICS = initRookMagics();
const ROOK_MASKS = initRookMasks();
const ROOK_SHIFTS = initRookShifts();
const ROOK_ATTACKS = initRookAttacks();

// Magic bitboard constants for bishop attacks
const BISHOP_MAGICS = initBishopMagics();
const BISHOP_MASKS = initBishopMasks();
const BISHOP_SHIFTS = initBishopShifts();
const BISHOP_ATTACKS = initBishopAttacks();

fn initKnightAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (file > 0 and square < 48) result |= sq << 15;
        if (file < 7 and square < 48) result |= sq << 17;
        if (file > 1 and square < 56) result |= sq << 6;
        if (file < 6 and square < 56) result |= sq << 10;
        if (file > 0 and square >= 16) result |= sq >> 17;
        if (file < 7 and square >= 16) result |= sq >> 15;
        if (file > 1 and square >= 8) result |= sq >> 10;
        if (file < 6 and square >= 8) result |= sq >> 6;

        attacks[square] = result;
    }
    return attacks;
}

fn initKingAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (square < 56) result |= sq << 8;
        if (square >= 8) result |= sq >> 8;
        if (file > 0) result |= sq >> 1;
        if (file < 7) result |= sq << 1;
        if (file > 0 and square < 56) result |= sq << 7;
        if (file < 7 and square < 56) result |= sq << 9;
        if (file > 0 and square >= 8) result |= sq >> 9;
        if (file < 7 and square >= 8) result |= sq >> 7;

        attacks[square] = result;
    }
    return attacks;
}

fn initWhitePawnAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (file > 0 and square < 56) result |= sq << 7; // Up-left
        if (file < 7 and square < 56) result |= sq << 9; // Up-right

        attacks[square] = result;
    }
    return attacks;
}

fn initBlackPawnAttacks() [64]u64 {
    @setEvalBranchQuota(10000);
    var attacks: [64]u64 = undefined;
    for (0..64) |square| {
        const sq: u64 = @as(u64, 1) << @intCast(square);
        const file = square % 8;
        var result: u64 = 0;

        if (file > 0 and square >= 8) result |= sq >> 9; // Down-left
        if (file < 7 and square >= 8) result |= sq >> 7; // Down-right

        attacks[square] = result;
    }
    return attacks;
}

// Magic bitboard initialization functions

fn initRookMagics() [64]u64 {
    // These are pre-computed magic numbers for rook attacks
    // Found through trial and error or algorithms like Fancy Magic Bitboards
    return [64]u64{
        0xa8002c000108020,  0x6c00049b0002001,  0x100200010090040,  0x2480041000800801, 0x280028004000800,  0x900410008040022,  0x280020001001080,  0x2880002041000080,
        0xa000800080400034, 0x4808020004000,    0x2290802004801000, 0x411000d00100020,  0x402800800040080,  0xb000401004208,    0x2409000100040200, 0x1002100004082,
        0x22878001e24000,   0x1090810021004010, 0x801030040200012,  0x500808008001000,  0xa08018014000880,  0x8000808004000200, 0x201008080010200,  0x801020000441091,
        0x800080204005,     0x1040200040100048, 0x120200402082,     0xd14880480100080,  0x12040280080080,   0x100040080020080,  0x41000f00240200,   0x813241200148449,
        0x491604001800080,  0x100401000402001,  0x4820010021001040, 0x400402202000812,  0x209009005000802,  0x810800601800400,  0x4301083214000150, 0x204026458e001401,
        0x40204000808000,   0x8001008040010020, 0x8410820820420010, 0x1003001000090020, 0x804040008008080,  0x12000810020004,   0x1000100200040208, 0x430000a044020001,
        0x280009023410300,  0xe0100040002240,   0x200100401700,     0x2244100408008080, 0x8000400801980,    0x2000810040200,    0x8010100228810400, 0x2000009044210200,
        0x4080008040102101, 0x40002080411d01,   0x2005524060000901, 0x502001008400422,  0x489a000810200402, 0x1004400080a13,    0x4000011008020084, 0x26002114058042,
    };
}

fn initRookMasks() [64]u64 {
    @setEvalBranchQuota(20000);
    var masks: [64]u64 = undefined;
    for (0..64) |square| {
        const rank: u8 = @intCast(square / 8);
        const file: u8 = @intCast(square % 8);
        var mask: u64 = 0;

        // North
        var r: u8 = rank + 1;
        while (r < 7) : (r += 1) {
            mask |= @as(u64, 1) << @intCast(r * 8 + file);
        }

        // South
        r = rank;
        while (r > 0) {
            r -= 1;
            if (r == 0) break;
            mask |= @as(u64, 1) << @intCast(r * 8 + file);
        }

        // East
        var f: u8 = file + 1;
        while (f < 7) : (f += 1) {
            mask |= @as(u64, 1) << @intCast(rank * 8 + f);
        }

        // West
        f = file;
        while (f > 0) {
            f -= 1;
            if (f == 0) break;
            mask |= @as(u64, 1) << @intCast(rank * 8 + f);
        }

        masks[square] = mask;
    }
    return masks;
}

fn initRookShifts() [64]u8 {
    @setEvalBranchQuota(10000);
    var shifts: [64]u8 = undefined;
    for (0..64) |square| {
        const bits = @popCount(ROOK_MASKS[square]);
        shifts[square] = @intCast(64 - bits);
    }
    return shifts;
}

fn initBishopMagics() [64]u64 {
    // Pre-computed magic numbers for bishop attacks
    return [64]u64{
        0x40040844404084,   0x2004208a004208,   0x10190041080202,   0x108060845042010,  0x581104180800210,  0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
        0x4050404440404,    0x21001420088,      0x24d0080801082102, 0x1020a0a020400,    0x40308200402,      0x4011002100800,    0x401484104104005,  0x801010402020200,
        0x400210c3880100,   0x404022024108200,  0x810018200204102,  0x4002801a02003,    0x85040820080400,   0x810102c808880400, 0xe900410884800,    0x8002020480840102,
        0x220200865090201,  0x2010100a02021202, 0x152048408022401,  0x20080002081110,   0x4001001021004000, 0x800040400a011002, 0xe4004081011002,   0x1c004001012080,
        0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080, 0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
        0x209188240001000,  0x400408a884001800, 0x110400a6080400,   0x1840060a44020800, 0x90080104000041,   0x201011000808101,  0x1a2208080504f080, 0x8012020600211212,
        0x500861011240000,  0x180806108200800,  0x4000020e01040044, 0x300000261044000a, 0x802241102020002,  0x20906061210001,   0x5a84841004010310, 0x4010801011c04,
        0xa010109502200,    0x4a02012000,       0x500201010098b028, 0x8040002811040900, 0x28000010020204,   0x6000020202d0240,  0x8918844842082200, 0x4010011029020020,
    };
}

fn initBishopMasks() [64]u64 {
    @setEvalBranchQuota(20000);
    var masks: [64]u64 = undefined;
    for (0..64) |square| {
        const rank: i8 = @intCast(square / 8);
        const file: i8 = @intCast(square % 8);
        var mask: u64 = 0;

        // North-East
        var r: i8 = rank + 1;
        var f: i8 = file + 1;
        while (r < 7 and f < 7) : ({
            r += 1;
            f += 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // North-West
        r = rank + 1;
        f = file - 1;
        while (r < 7 and f > 0) : ({
            r += 1;
            f -= 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // South-East
        r = rank - 1;
        f = file + 1;
        while (r > 0 and f < 7) : ({
            r -= 1;
            f += 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        // South-West
        r = rank - 1;
        f = file - 1;
        while (r > 0 and f > 0) : ({
            r -= 1;
            f -= 1;
        }) {
            mask |= @as(u64, 1) << @intCast(r * 8 + f);
        }

        masks[square] = mask;
    }
    return masks;
}

fn initBishopShifts() [64]u8 {
    @setEvalBranchQuota(10000);
    var shifts: [64]u8 = undefined;
    for (0..64) |square| {
        const bits = @popCount(BISHOP_MASKS[square]);
        shifts[square] = @intCast(64 - bits);
    }
    return shifts;
}

// Helper function to compute rook attacks for a given square and occupancy
fn computeRookAttacks(square: u6, occupied: u64) u64 {
    var attacks: u64 = 0;
    const file: i8 = @intCast(square % 8);
    const rank: i8 = @intCast(square / 8);

    // North
    var r: i8 = rank + 1;
    while (r < 8) : (r += 1) {
        const sq: u6 = @intCast(r * 8 + file);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South
    r = rank - 1;
    while (r >= 0) : (r -= 1) {
        const sq: u6 = @intCast(r * 8 + file);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // East
    var f: i8 = file + 1;
    while (f < 8) : (f += 1) {
        const sq: u6 = @intCast(rank * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // West
    f = file - 1;
    while (f >= 0) : (f -= 1) {
        const sq: u6 = @intCast(rank * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    return attacks;
}

// Helper function to compute bishop attacks for a given square and occupancy
fn computeBishopAttacks(square: u6, occupied: u64) u64 {
    var attacks: u64 = 0;
    const file: i8 = @intCast(square % 8);
    const rank: i8 = @intCast(square / 8);

    // North-East
    var r: i8 = rank + 1;
    var f: i8 = file + 1;
    while (r < 8 and f < 8) : ({
        r += 1;
        f += 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // North-West
    r = rank + 1;
    f = file - 1;
    while (r < 8 and f >= 0) : ({
        r += 1;
        f -= 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South-East
    r = rank - 1;
    f = file + 1;
    while (r >= 0 and f < 8) : ({
        r -= 1;
        f += 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    // South-West
    r = rank - 1;
    f = file - 1;
    while (r >= 0 and f >= 0) : ({
        r -= 1;
        f -= 1;
    }) {
        const sq: u6 = @intCast(r * 8 + f);
        attacks |= @as(u64, 1) << sq;
        if ((occupied & (@as(u64, 1) << sq)) != 0) break;
    }

    return attacks;
}

fn initRookAttacks() [64][4096]u64 {
    @setEvalBranchQuota(1000000);
    var attacks: [64][4096]u64 = undefined;

    for (0..64) |square| {
        const sq: u6 = @intCast(square);
        const mask = ROOK_MASKS[square];
        const bits = @popCount(mask);
        const permutations: usize = @as(usize, 1) << @intCast(bits);

        for (0..permutations) |i| {
            const occupied = indexToOccupancy(i, bits, mask);
            const magic_index = (occupied *% ROOK_MAGICS[square]) >> @intCast(ROOK_SHIFTS[square]);
            attacks[square][magic_index] = computeRookAttacks(sq, occupied);
        }
    }

    return attacks;
}

fn initBishopAttacks() [64][512]u64 {
    @setEvalBranchQuota(1000000);
    var attacks: [64][512]u64 = undefined;

    for (0..64) |square| {
        const sq: u6 = @intCast(square);
        const mask = BISHOP_MASKS[square];
        const bits = @popCount(mask);
        const permutations: usize = @as(usize, 1) << @intCast(bits);

        for (0..permutations) |i| {
            const occupied = indexToOccupancy(i, bits, mask);
            const magic_index = (occupied *% BISHOP_MAGICS[square]) >> @intCast(BISHOP_SHIFTS[square]);
            attacks[square][magic_index] = computeBishopAttacks(sq, occupied);
        }
    }

    return attacks;
}

// Convert an index to an occupancy bitboard for magic bitboard initialization
fn indexToOccupancy(index: usize, bits: u8, mask: u64) u64 {
    @setEvalBranchQuota(2000000);
    var occupancy: u64 = 0;
    var temp_mask = mask;

    for (0..bits) |i| {
        const bit_index = @ctz(temp_mask);
        temp_mask &= temp_mask - 1; // Clear the least significant bit

        if ((index & (@as(usize, 1) << @intCast(i))) != 0) {
            occupancy |= @as(u64, 1) << @intCast(bit_index);
        }
    }

    return occupancy;
}

pub inline fn getKnightAttacks(square: u6) u64 {
    return KNIGHT_ATTACKS[square];
}

pub inline fn getKingAttacks(square: u6) u64 {
    return KING_ATTACKS[square];
}

pub inline fn getPawnAttacks(square: u6, pawn_color: pieceInfo.Color) u64 {
    return if (pawn_color == .white) WHITE_PAWN_ATTACKS[square] else BLACK_PAWN_ATTACKS[square];
}

pub inline fn getRookAttacks(square: u6, occupied: u64) u64 {
    const masked_occupied = occupied & ROOK_MASKS[square];
    const index = (masked_occupied *% ROOK_MAGICS[square]) >> @intCast(ROOK_SHIFTS[square]);
    return ROOK_ATTACKS[square][index];
}

pub inline fn getBishopAttacks(square: u6, occupied: u64) u64 {
    const masked_occupied = occupied & BISHOP_MASKS[square];
    const index = (masked_occupied *% BISHOP_MAGICS[square]) >> @intCast(BISHOP_SHIFTS[square]);
    return BISHOP_ATTACKS[square][index];
}

pub inline fn getQueenAttacks(square: u6, occupied: u64) u64 {
    return getRookAttacks(square, occupied) | getBishopAttacks(square, occupied);
}
