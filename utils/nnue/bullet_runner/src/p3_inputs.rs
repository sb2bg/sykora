use bullet_lib::game::{
    formats::bulletformat::ChessBoard,
    inputs::{ChessBucketsMirrored, SparseInputType},
};

use crate::full_threats_v1::{self, FullThreatInputs};

/// The 64-square pawn identity is deliberate: Sykora accepts malformed FENs
/// with back-rank pawns, and reserving those rows keeps trainer/export/runtime
/// indexing total. Legal positions use 96 of the 128 allocated rows.
pub const RANK: usize = 32;
pub const PAWN_COUNT: usize = 2 * 64;
pub const CONTEXT_COUNT: usize = 2 * 5 * 64;

pub const PAWN_OFFSET: usize = full_threats_v1::TRAINING_INPUTS;
pub const CONTEXT_OFFSET: usize = PAWN_OFFSET + PAWN_COUNT;
pub const TRAINING_INPUTS: usize = CONTEXT_OFFSET + CONTEXT_COUNT;

const MAX_ACTIVE_P3_ENTITIES: usize = 32;
pub const MAX_ACTIVE_INPUTS: usize = full_threats_v1::MAX_ACTIVE_INPUTS + MAX_ACTIVE_P3_ENTITIES;

#[derive(Clone, Copy, Debug)]
pub struct P3Inputs {
    base: FullThreatInputs,
    psq: ChessBucketsMirrored,
}

impl P3Inputs {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self {
            base: FullThreatInputs::new(buckets),
            psq: ChessBucketsMirrored::new(buckets),
        }
    }
}

fn decode_factorised_psq(feature: usize) -> (usize, usize, usize) {
    debug_assert!(feature < 768);
    let relative_colour = feature / 384;
    let piece_and_square = feature % 384;
    let kind = piece_and_square / 64;
    let square = piece_and_square % 64;
    (relative_colour, kind, square)
}

fn context_index(relative_colour: usize, kind: usize, square: usize) -> usize {
    debug_assert!(relative_colour < 2);
    debug_assert!((1..=5).contains(&kind));
    debug_assert!(square < 64);
    (relative_colour * 5 + (kind - 1)) * 64 + square
}

fn pawn_identity(relative_colour: usize, square: usize) -> usize {
    debug_assert!(relative_colour < 2);
    debug_assert!(square < 64);
    relative_colour * 64 + square
}

impl SparseInputType for P3Inputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TRAINING_INPUTS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_INPUTS
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Decode the established mirrored PSQ orientation rather than
        // duplicating its perspective, colour, and horizontal-mirror rules.
        // P3 rows are emitted first so the custom operation sees a compact
        // entity prefix; sparse feature order is immaterial to the main FT.
        self.psq.map_features(pos, |stm, ntm| {
            let stm_factor = stm % 768;
            let ntm_factor = ntm % 768;
            let (stm_colour, stm_kind, stm_square) = decode_factorised_psq(stm_factor);
            let (ntm_colour, ntm_kind, ntm_square) = decode_factorised_psq(ntm_factor);
            debug_assert_eq!(stm_kind, ntm_kind);

            if stm_kind == 0 {
                f(
                    PAWN_OFFSET + pawn_identity(stm_colour, stm_square),
                    PAWN_OFFSET + pawn_identity(ntm_colour, ntm_square),
                );
            } else {
                f(
                    CONTEXT_OFFSET + context_index(stm_colour, stm_kind, stm_square),
                    CONTEXT_OFFSET + context_index(ntm_colour, ntm_kind, ntm_square),
                );
            }
        });
        self.base.map_features(pos, f);
    }

    fn shorthand(&self) -> String {
        "virtual-factorised-psq10+full_threats_v1+p3_anova_r32".to_string()
    }

    fn description(&self) -> String {
        "Mirrored PSQ/threat inputs with atomic pawn and non-pawn rows for rank-32 P3-ANOVA"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn p3_counts(fen: &str) -> (usize, usize) {
        let board = ChessBoard::from_str(&format!("{fen} | 0 | 0.5")).unwrap();
        let inputs = P3Inputs::new([0; 32]);
        let mut pawns = 0;
        let mut context = 0;
        inputs.map_features(&board, |stm, ntm| {
            assert!(stm < TRAINING_INPUTS && ntm < TRAINING_INPUTS);
            if (PAWN_OFFSET..CONTEXT_OFFSET).contains(&stm) {
                pawns += 1;
            } else if stm >= CONTEXT_OFFSET {
                context += 1;
            }
        });
        (pawns, context)
    }

    #[test]
    fn start_position_has_expected_p3_features() {
        assert_eq!(
            p3_counts("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            (16, 16)
        );
    }

    #[test]
    fn every_entity_gets_one_atomic_p3_row() {
        assert_eq!(p3_counts("4k3/8/2p5/2P5/3P4/8/8/4K3 w - - 0 1"), (3, 2));
    }

    #[test]
    fn atomic_ranges_are_dense_and_disjoint() {
        assert_eq!(PAWN_OFFSET + PAWN_COUNT, CONTEXT_OFFSET);
        assert_eq!(CONTEXT_OFFSET + CONTEXT_COUNT, TRAINING_INPUTS);
    }
}
