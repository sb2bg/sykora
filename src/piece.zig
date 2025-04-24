pub const Color = enum {
    white,
    black,
};

pub const Type = enum {
    pawn,
    knight,
    bishop,
    rook,
    queen,
    king,

    pub fn getName(self: Type) u8 {
        return switch (self) {
            .pawn => 'P',
            .knight => 'N',
            .bishop => 'B',
            .rook => 'R',
            .queen => 'Q',
            .king => 'K',
        };
    }

    pub fn fromChar(c: u8) ?Type {
        return switch (c) {
            'P', 'p' => .pawn,
            'N', 'n' => .knight,
            'B', 'b' => .bishop,
            'R', 'r' => .rook,
            'Q', 'q' => .queen,
            'K', 'k' => .king,
            else => null,
        };
    }
};
