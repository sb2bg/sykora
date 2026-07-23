use bullet_lib::game::{
    formats::bulletformat::ChessBoard,
    inputs::{ChessBucketsMirrored, SparseInputType},
};

pub const SCHEME_ID: u16 = 1;
pub const FEATURE_COUNT: usize = 60_720;
pub const MAX_ACTIVE_THREATS: usize = 240;
pub const PSQ_FACTOR_FEATURES: usize = 768;
pub const PSQ_RESIDUAL_FEATURES: usize = 7_680;
pub const THREAT_OFFSET: usize = PSQ_FACTOR_FEATURES + PSQ_RESIDUAL_FEATURES;
pub const TRAINING_INPUTS: usize = THREAT_OFFSET + FEATURE_COUNT;
pub const MAX_ACTIVE_INPUTS: usize = 64 + MAX_ACTIVE_THREATS;
pub const PACKING_HASH: [u8; 32] = [
    0x96, 0x45, 0x91, 0xed, 0xbe, 0x85, 0x6c, 0x9f, 0x90, 0x69, 0x4d, 0xcb, 0xfa, 0xbe, 0x42, 0xd5,
    0x8b, 0x01, 0x1a, 0x46, 0x9e, 0x32, 0x75, 0xa8, 0xaa, 0xa9, 0xe4, 0x24, 0x9b, 0x21, 0x98, 0x8a,
];

const OURS: usize = 0;
const THEIRS: usize = 1;
const PAWN: usize = 0;
const KNIGHT: usize = 1;
const BISHOP: usize = 2;
const ROOK: usize = 3;
const QUEEN: usize = 4;
const KING: usize = 5;

const GEOMETRY_COUNTS: [usize; 5] = [132, 336, 560, 896, 1_456];
const TARGET_SLOT_COUNTS: [usize; 5] = [6, 10, 8, 8, 10];
const TYPE_BASES: [usize; 5] = [0, 792, 4_152, 8_632, 15_800];
const FEATURES_PER_ATTACKER_COLOUR: usize = 30_360;

type GeometryIndices = [[[[i16; 64]; 64]; 5]; 2];

const fn abs_i8(value: i8) -> u8 {
    if value < 0 {
        (-value) as u8
    } else {
        value as u8
    }
}

const fn valid_geometry(colour: usize, attacker_type: usize, source: usize, target: usize) -> bool {
    if source == target {
        return false;
    }
    let source_rank = (source / 8) as i8;
    let source_file = (source % 8) as i8;
    let target_rank = (target / 8) as i8;
    let target_file = (target % 8) as i8;
    let dr = target_rank - source_rank;
    let df = target_file - source_file;
    let adr = abs_i8(dr);
    let adf = abs_i8(df);
    match attacker_type {
        PAWN => {
            if source_rank == 0 || source_rank == 7 {
                return false;
            }
            let forward = if colour == OURS { 1 } else { -1 };
            dr == forward && adf <= 1
        }
        KNIGHT => (adr == 1 && adf == 2) || (adr == 2 && adf == 1),
        BISHOP => adr == adf,
        ROOK => dr == 0 || df == 0,
        QUEEN => adr == adf || dr == 0 || df == 0,
        _ => false,
    }
}

const fn init_geometry_indices() -> GeometryIndices {
    let mut result = [[[[-1; 64]; 64]; 5]; 2];
    let mut colour = 0;
    while colour < 2 {
        let mut attacker_type = 0;
        while attacker_type < 5 {
            let mut geometry = 0i16;
            let mut source = 0;
            while source < 64 {
                let mut target = 0;
                while target < 64 {
                    if valid_geometry(colour, attacker_type, source, target) {
                        result[colour][attacker_type][source][target] = geometry;
                        geometry += 1;
                    }
                    target += 1;
                }
                source += 1;
            }
            assert!(geometry as usize == GEOMETRY_COUNTS[attacker_type]);
            attacker_type += 1;
        }
        colour += 1;
    }
    result
}

static GEOMETRY_INDICES: GeometryIndices = init_geometry_indices();

const fn target_type_rank(attacker_type: usize, target_type: usize) -> Option<usize> {
    match attacker_type {
        PAWN => match target_type {
            PAWN => Some(0),
            KNIGHT => Some(1),
            ROOK => Some(2),
            _ => None,
        },
        KNIGHT | QUEEN => match target_type {
            PAWN => Some(0),
            KNIGHT => Some(1),
            BISHOP => Some(2),
            ROOK => Some(3),
            QUEEN => Some(4),
            _ => None,
        },
        BISHOP | ROOK => match target_type {
            PAWN => Some(0),
            KNIGHT => Some(1),
            BISHOP => Some(2),
            ROOK => Some(3),
            _ => None,
        },
        _ => None,
    }
}

pub fn encode(
    attacker_colour: usize,
    attacker_type: usize,
    source: usize,
    target: usize,
    target_colour: usize,
    target_type: usize,
) -> Option<usize> {
    if attacker_colour > THEIRS
        || target_colour > THEIRS
        || attacker_type >= KING
        || target_type >= KING
        || source >= 64
        || target >= 64
    {
        return None;
    }
    let geometry = GEOMETRY_INDICES[attacker_colour][attacker_type][source][target];
    if geometry < 0 {
        return None;
    }
    let type_rank = target_type_rank(attacker_type, target_type)?;
    if target_type == PAWN && matches!(target / 8, 0 | 7) {
        return None;
    }
    if attacker_type == target_type {
        let enemy_pair = attacker_colour != target_colour;
        let friendly_symmetric = attacker_colour == target_colour && attacker_type != PAWN;
        if (enemy_pair || friendly_symmetric) && source >= target {
            return None;
        }
    }
    let types_per_colour = TARGET_SLOT_COUNTS[attacker_type] / 2;
    let slot = target_colour * types_per_colour + type_rank;
    let feature = attacker_colour * FEATURES_PER_ATTACKER_COLOUR
        + TYPE_BASES[attacker_type]
        + geometry as usize * TARGET_SLOT_COUNTS[attacker_type]
        + slot;
    debug_assert!(feature < FEATURE_COUNT);
    Some(feature)
}

#[derive(Clone, Copy, Default)]
struct Relation {
    attacker_colour: usize,
    attacker_type: usize,
    source: usize,
    target_colour: usize,
    target_type: usize,
    target: usize,
}

#[derive(Clone, Copy)]
struct Position {
    colours: [u64; 2],
    kinds: [u64; 6],
}

impl Position {
    fn from_bullet(pos: &ChessBoard) -> Self {
        let mut result = Self {
            colours: [0; 2],
            kinds: [0; 6],
        };
        for (piece, square) in *pos {
            let colour = usize::from(piece & 8 != 0);
            let kind = usize::from(piece & 7);
            let bit = 1u64 << square;
            result.colours[colour] |= bit;
            result.kinds[kind] |= bit;
        }
        result
    }

    fn occupied(self) -> u64 {
        self.colours[OURS] | self.colours[THEIRS]
    }

    fn piece_at(self, square: usize) -> Option<(usize, usize)> {
        let bit = 1u64 << square;
        let colour = if self.colours[OURS] & bit != 0 {
            OURS
        } else if self.colours[THEIRS] & bit != 0 {
            THEIRS
        } else {
            return None;
        };
        (0..6)
            .find(|&kind| self.kinds[kind] & bit != 0)
            .map(|kind| (colour, kind))
    }
}

fn jump_attacks(square: usize, deltas: &[(i8, i8)]) -> u64 {
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let mut attacks = 0u64;
    for &(dr, df) in deltas {
        let target_rank = rank + dr;
        let target_file = file + df;
        if (0..8).contains(&target_rank) && (0..8).contains(&target_file) {
            attacks |= 1u64 << (target_rank as usize * 8 + target_file as usize);
        }
    }
    attacks
}

fn slider_attacks(position: Position, square: usize, directions: &[(i8, i8)]) -> u64 {
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let occupied = position.occupied();
    let mut attacks = 0u64;
    for &(dr, df) in directions {
        let mut target_rank = rank + dr;
        let mut target_file = file + df;
        while (0..8).contains(&target_rank) && (0..8).contains(&target_file) {
            let bit = 1u64 << (target_rank as usize * 8 + target_file as usize);
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            target_rank += dr;
            target_file += df;
        }
    }
    attacks
}

fn occupied_targets(position: Position, colour: usize, kind: usize, square: usize) -> u64 {
    const KNIGHT_DELTAS: [(i8, i8); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    const ROOK_DIRECTIONS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ];
    let occupied = position.occupied();
    let attacks = match kind {
        PAWN => {
            let direction = if colour == OURS { 1 } else { -1 };
            let mut result = jump_attacks(square, &[(direction, -1), (direction, 1)]);
            let rank = (square / 8) as i8 + direction;
            if (0..8).contains(&rank) {
                let forward = rank as usize * 8 + square % 8;
                let bit = 1u64 << forward;
                if occupied & bit != 0 {
                    result |= bit;
                }
            }
            result
        }
        KNIGHT => jump_attacks(square, &KNIGHT_DELTAS),
        BISHOP => slider_attacks(position, square, &BISHOP_DIRECTIONS),
        ROOK => slider_attacks(position, square, &ROOK_DIRECTIONS),
        QUEEN => slider_attacks(position, square, &QUEEN_DIRECTIONS),
        _ => 0,
    };
    attacks & occupied
}

fn relations(position: Position) -> ([Relation; MAX_ACTIVE_THREATS], usize) {
    let mut result = [Relation::default(); MAX_ACTIVE_THREATS];
    let mut count = 0;
    for attacker_colour in [OURS, THEIRS] {
        for attacker_type in PAWN..=QUEEN {
            let mut attackers = position.colours[attacker_colour] & position.kinds[attacker_type];
            while attackers != 0 {
                let source = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let mut targets =
                    occupied_targets(position, attacker_colour, attacker_type, source);
                while targets != 0 {
                    let target = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    let (target_colour, target_type) =
                        position.piece_at(target).expect("occupied target");
                    if target_type == KING {
                        continue;
                    }
                    assert!(
                        count < result.len(),
                        "full_threats_v1 active bound exceeded"
                    );
                    result[count] = Relation {
                        attacker_colour,
                        attacker_type,
                        source,
                        target_colour,
                        target_type,
                        target,
                    };
                    count += 1;
                }
            }
        }
    }
    (result, count)
}

fn oriented_square(square: usize, stm_perspective: bool, physical_king: usize) -> usize {
    let mut oriented = if stm_perspective { square } else { square ^ 56 };
    let oriented_king = if stm_perspective {
        physical_king
    } else {
        physical_king ^ 56
    };
    if oriented_king % 8 > 3 {
        oriented ^= 7;
    }
    oriented
}

fn enumerate_threats(
    pos: &ChessBoard,
    stm_perspective: bool,
    output: &mut [usize; MAX_ACTIVE_THREATS],
) -> usize {
    let position = Position::from_bullet(pos);
    let physical_king = if stm_perspective {
        usize::from(pos.our_ksq())
    } else {
        usize::from(pos.opp_ksq() ^ 56)
    };
    let (relations, relation_count) = relations(position);
    let mut count = 0;
    for relation in &relations[..relation_count] {
        let attacker_colour = if stm_perspective {
            relation.attacker_colour
        } else {
            relation.attacker_colour ^ 1
        };
        let target_colour = if stm_perspective {
            relation.target_colour
        } else {
            relation.target_colour ^ 1
        };
        let Some(feature) = encode(
            attacker_colour,
            relation.attacker_type,
            oriented_square(relation.source, stm_perspective, physical_king),
            oriented_square(relation.target, stm_perspective, physical_king),
            target_colour,
            relation.target_type,
        ) else {
            continue;
        };
        output[count] = feature;
        count += 1;
    }
    output[..count].sort_unstable();
    if count == 0 {
        return 0;
    }
    let mut unique = 1;
    for index in 1..count {
        if output[index] != output[unique - 1] {
            output[unique] = output[index];
            unique += 1;
        }
    }
    unique
}

#[derive(Clone, Copy, Debug)]
pub struct FullThreatInputs {
    psq: ChessBucketsMirrored,
}

impl FullThreatInputs {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self {
            psq: ChessBucketsMirrored::new(buckets),
        }
    }
}

impl SparseInputType for FullThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TRAINING_INPUTS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_INPUTS
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.psq.map_features(pos, |stm, ntm| {
            f(stm % 768, ntm % 768);
            f(PSQ_FACTOR_FEATURES + stm, PSQ_FACTOR_FEATURES + ntm);
        });

        let mut stm = [0usize; MAX_ACTIVE_THREATS];
        let mut ntm = [0usize; MAX_ACTIVE_THREATS];
        let stm_count = enumerate_threats(pos, true, &mut stm);
        let ntm_count = enumerate_threats(pos, false, &mut ntm);
        assert_eq!(
            stm_count, ntm_count,
            "full_threats_v1 perspective feature-count mismatch"
        );
        for index in 0..stm_count {
            f(THREAT_OFFSET + stm[index], THREAT_OFFSET + ntm[index]);
        }
    }

    fn shorthand(&self) -> String {
        "virtual-factorised-psq10+full_threats_v1".to_string()
    }

    fn description(&self) -> String {
        "Mirrored PSQ with virtual factoriser and shared full-threat inputs".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn start_position_matches_golden_ids() {
        let board = ChessBoard::from_str(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5",
        )
        .unwrap();
        let mut stm = [0usize; MAX_ACTIVE_THREATS];
        let mut ntm = [0usize; MAX_ACTIVE_THREATS];
        let stm_count = enumerate_threats(&board, true, &mut stm);
        let ntm_count = enumerate_threats(&board, false, &mut ntm);
        assert_eq!(stm_count, 28);
        assert_eq!(stm_count, ntm_count);
        assert_eq!(&stm[..stm_count], &ntm[..ntm_count]);
        assert_eq!(&stm[..8], &[812, 1002, 4264, 4272, 4432, 4440, 8633, 8688]);
    }

    #[test]
    fn tactical_fixture_matches_runtime_golden_ids() {
        let board =
            ChessBoard::from_str("8/8/3q4/2b1r3/3N4/2P1P3/4R3/4K2k w - - 0 1 | 0 | 0.5").unwrap();
        let mut stm = [0usize; MAX_ACTIVE_THREATS];
        let mut ntm = [0usize; MAX_ACTIVE_THREATS];
        let stm_count = enumerate_threats(&board, true, &mut stm);
        let ntm_count = enumerate_threats(&board, false, &mut ntm);
        assert_eq!(stm_count, 10);
        assert_eq!(ntm_count, 10);
        assert_eq!(&stm[..4], &[193, 217, 2215, 9928]);
        assert_eq!(&ntm[..4], &[6221, 11714, 11748, 20353]);
    }
}
