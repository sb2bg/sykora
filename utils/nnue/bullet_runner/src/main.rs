mod full_threats_v1;
mod p3_inputs;

use bullet_lib::{
    acyclib::graph::ir::operation::p3::P3Anova,
    game::{
        formats::{bulletformat::ChessBoard, sfbinpack::TrainingDataEntry},
        inputs::{SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr::LrScheduler, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use std::env;

use full_threats_v1::FullThreatInputs;
use p3_inputs::P3Inputs;

#[rustfmt::skip]
const BUCKET_LAYOUT_V3_10: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

const FT_QUANT: i16 = 255;
const THREAT_WEIGHT_MIN: f32 = i8::MIN as f32 / FT_QUANT as f32;
const THREAT_WEIGHT_MAX: f32 = i8::MAX as f32 / FT_QUANT as f32;
const POOL_QUANT: i32 = 128;
const DENSE_QUANT_I16: i16 = 64;
const DENSE_QUANT_I32: i32 = 64;

#[derive(Clone, Debug)]
struct CosineDecayFrom {
    initial_lr: f32,
    final_lr: f32,
    start_superbatch: usize,
    final_superbatch: usize,
}

impl LrScheduler for CosineDecayFrom {
    fn lr(&self, _batch: usize, superbatch: usize) -> f32 {
        if superbatch <= self.start_superbatch {
            return self.initial_lr;
        }
        if superbatch >= self.final_superbatch {
            return self.final_lr;
        }
        let elapsed = superbatch - self.start_superbatch;
        let duration = self.final_superbatch - self.start_superbatch;
        let progress = elapsed as f32 / duration as f32;
        let lambda = 0.5 * (1.0 - (std::f32::consts::PI * progress).cos());
        self.initial_lr + lambda * (self.final_lr - self.initial_lr)
    }

    fn colourful(&self) -> String {
        format!(
            "cosine {} at superbatch {} -> {} at superbatch {}",
            self.initial_lr, self.start_superbatch, self.final_lr, self.final_superbatch
        )
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn binpack_filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
}

#[derive(Clone, Copy, Debug)]
enum SykoraInputs {
    V8(FullThreatInputs),
    P3(P3Inputs),
}

impl SparseInputType for SykoraInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        match self {
            Self::V8(inputs) => inputs.num_inputs(),
            Self::P3(inputs) => inputs.num_inputs(),
        }
    }

    fn max_active(&self) -> usize {
        match self {
            Self::V8(inputs) => inputs.max_active(),
            Self::P3(inputs) => inputs.max_active(),
        }
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        match self {
            Self::V8(inputs) => inputs.map_features(pos, f),
            Self::P3(inputs) => inputs.map_features(pos, f),
        }
    }

    fn shorthand(&self) -> String {
        match self {
            Self::V8(inputs) => inputs.shorthand(),
            Self::P3(inputs) => inputs.shorthand(),
        }
    }

    fn description(&self) -> String {
        match self {
            Self::V8(inputs) => inputs.description(),
            Self::P3(inputs) => inputs.description(),
        }
    }
}

fn saved_format(with_p3: bool) -> Vec<SavedFormat> {
    let mut format = vec![
        // V8's l0w is a virtual-factorised vocabulary containing shared
        // PSQ, bucket residual, and threat rows. Raw checkpoints retain it
        // verbatim; checkpoint_raw_to_npz.py separates and merges the deployed
        // tensors. Mixed i16/i8 quantisation cannot be expressed in Bullet's
        // single SavedFormat item.
        SavedFormat::id("l0w").transform(|_, _| Vec::new()),
        SavedFormat::id("l0b").round().quantise::<i16>(FT_QUANT),
        SavedFormat::id("l1w")
            .transpose()
            .round()
            .quantise::<i8>(DENSE_QUANT_I16),
        SavedFormat::id("l1b")
            .round()
            .quantise::<i32>(POOL_QUANT * DENSE_QUANT_I32),
        SavedFormat::id("l2w")
            .transpose()
            .round()
            .quantise::<i8>(DENSE_QUANT_I16),
        SavedFormat::id("l2b")
            .round()
            .quantise::<i32>(DENSE_QUANT_I32 * DENSE_QUANT_I32),
        SavedFormat::id("l3w")
            .transpose()
            .round()
            .quantise::<i8>(DENSE_QUANT_I16),
        SavedFormat::id("l3b")
            .round()
            .quantise::<i32>(DENSE_QUANT_I32 * DENSE_QUANT_I32),
    ];
    if with_p3 {
        // Retain the factorised P3 tensors in full precision. Deployment
        // quantisation is calibrated after training.
        format.extend([
            SavedFormat::id("p3_pawnw").transform(|_, _| Vec::new()),
            SavedFormat::id("p3_contextw").transform(|_, _| Vec::new()),
            SavedFormat::id("p3_l1w").transform(|_, _| Vec::new()),
            SavedFormat::id("p3_l1b").transform(|_, _| Vec::new()),
        ]);
    }
    format
}

#[allow(clippy::too_many_arguments)]
fn run_network<const O: usize>(
    num_input_buckets: usize,
    dataset_paths: &[&str],
    data_format: &str,
    hl_size: usize,
    dense1_size: usize,
    dense2_size: usize,
    initial_lr: f32,
    final_lr: f32,
    lr_origin_superbatch: usize,
    lr_final_superbatch: usize,
    start_superbatch: usize,
    superbatches: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    wdl_proportion: f32,
    save_rate: usize,
    threads: usize,
    output_dir: &str,
    net_id: String,
    resume_from: Option<&str>,
    warm_start_weights: Option<&str>,
    with_p3: bool,
    freeze_p3_base: bool,
) {
    let save_format = saved_format(with_p3);
    let input_getter = if with_p3 {
        SykoraInputs::P3(P3Inputs::new(BUCKET_LAYOUT_V3_10))
    } else {
        SykoraInputs::V8(FullThreatInputs::new(BUCKET_LAYOUT_V3_10))
    };
    let training_inputs = input_getter.num_inputs();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(input_getter)
        .output_buckets(MaterialCount::<O>)
        .use_threads(threads)
        .save_format(&save_format)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(move |builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", training_inputs, hl_size);
            l0.init_with_effective_input_size(128);
            let l1 = builder.new_affine("l1", hl_size, O * dense1_size);
            let l2 = builder.new_affine("l2", 2 * dense1_size, O * dense2_size);
            let l3 = builder.new_affine("l3", dense2_size, O);

            let (stm_l0, ntm_l0) = if freeze_p3_base {
                let weights = l0.weights.copy_stop_grad();
                let bias = l0.bias.copy_stop_grad();
                (
                    weights.matmul(stm_inputs) + bias,
                    weights.matmul(ntm_inputs) + bias,
                )
            } else {
                (l0.forward(stm_inputs), l0.forward(ntm_inputs))
            };
            let stm_hidden = stm_l0.crelu().pairwise_mul();
            let ntm_hidden = ntm_l0.crelu().pairwise_mul();
            let pooled = stm_hidden.concat(ntm_hidden);

            let z1_base = if freeze_p3_base {
                (l1.weights.copy_stop_grad().matmul(pooled) + l1.bias.copy_stop_grad())
                    .select(output_buckets)
            } else {
                l1.forward(pooled).select(output_buckets)
            };
            let z1 = if with_p3 {
                let embedding_init = InitSettings::Normal {
                    mean: 0.0,
                    stdev: 0.05,
                };
                let p3_pawnw = builder.new_weights(
                    "p3_pawnw",
                    Shape::new(p3_inputs::RANK, p3_inputs::PAWN_COUNT),
                    embedding_init,
                );
                let p3_contextw = builder.new_weights(
                    "p3_contextw",
                    Shape::new(p3_inputs::RANK, p3_inputs::CONTEXT_COUNT),
                    embedding_init,
                );
                let p3_l1 = builder.new_affine("p3_l1", 4 * p3_inputs::RANK, O * dense1_size);

                let stm_anova = P3Anova {
                    pawn_weights: p3_pawnw.annotated_node(),
                    context_weights: p3_contextw.annotated_node(),
                    indices: stm_inputs.annotated_node(),
                    pawn_offset: p3_inputs::PAWN_OFFSET,
                    pawn_count: p3_inputs::PAWN_COUNT,
                    context_offset: p3_inputs::CONTEXT_OFFSET,
                    context_count: p3_inputs::CONTEXT_COUNT,
                    rank: p3_inputs::RANK,
                };
                let ntm_anova = P3Anova {
                    indices: ntm_inputs.annotated_node(),
                    ..stm_anova.clone()
                };
                // Signed and unclipped. The custom operation forms exact
                // distinct same-file and adjacent-file pawn moments, then
                // multiplies both channels by the non-pawn context sum.
                let stm_tri = builder.apply(stm_anova);
                let ntm_tri = builder.apply(ntm_anova);
                let z1_tri = p3_l1
                    .forward(stm_tri.concat(ntm_tri))
                    .select(output_buckets);
                z1_base + z1_tri
            } else {
                z1_base
            };
            let dual = z1.crelu().concat(z1.abs_pow(2.0).crelu());
            let hidden = l2.forward(dual).select(output_buckets).screlu();
            l3.forward(hidden).select(output_buckets)
        });

    let ft_clipping = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", ft_clipping);
    trainer.optimiser.add_clip_range_for_weight(
        "l0w",
        full_threats_v1::THREAT_OFFSET * hl_size,
        full_threats_v1::FEATURE_COUNT * hl_size,
        THREAT_WEIGHT_MIN,
        THREAT_WEIGHT_MAX,
    );
    if with_p3 {
        let p3_suffix = p3_inputs::TRAINING_INPUTS - full_threats_v1::TRAINING_INPUTS;
        trainer.optimiser.add_clip_range_for_weight(
            "l0w",
            full_threats_v1::TRAINING_INPUTS * hl_size,
            p3_suffix * hl_size,
            0.0,
            0.0,
        );

        trainer
            .optimiser
            .add_clip_range_for_weight("p3_l1b", 0, O * dense1_size, 0.0, 0.0);
    }

    let dense_clipping = AdamWParams {
        max_weight: 1.98,
        min_weight: -1.98,
        ..Default::default()
    };
    trainer
        .optimiser
        .set_params_for_weight("l1w", dense_clipping);
    trainer
        .optimiser
        .set_params_for_weight("l2w", dense_clipping);
    trainer
        .optimiser
        .set_params_for_weight("l3w", dense_clipping);
    if with_p3 {
        for id in ["p3_pawnw", "p3_contextw", "p3_l1w"] {
            trainer.optimiser.set_params_for_weight(id, dense_clipping);
        }
    }
    if freeze_p3_base {
        // Stopping the graph gradient leaves an allocated zero-gradient tensor;
        // AdamW would still decay those weights. Disable decay as well so the
        // mature FT and first material affine remain bit-identical in phase 1.
        trainer.optimiser.set_params_for_weight(
            "l0w",
            AdamWParams {
                decay: 0.0,
                ..ft_clipping
            },
        );
        for id in ["l0b", "l1w", "l1b"] {
            trainer.optimiser.set_params_for_weight(
                id,
                AdamWParams {
                    decay: 0.0,
                    ..dense_clipping
                },
            );
        }
    }

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL {
            value: wdl_proportion,
        },
        lr_scheduler: CosineDecayFrom {
            initial_lr,
            final_lr,
            start_superbatch: lr_origin_superbatch,
            final_superbatch: lr_final_superbatch,
        },
        save_rate,
    };

    // Bullet currently exposes this field but does not implement validation.
    // Validation is performed independently by validate_checkpoints.py.
    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: 32,
    };

    if let Some(path) = resume_from
        && !path.is_empty()
    {
        trainer.load_from_checkpoint(path);
    }
    if let Some(path) = warm_start_weights
        && !path.is_empty()
    {
        if resume_from.is_some_and(|value| !value.is_empty()) {
            panic!("SYK_RESUME and SYK_WARM_START_WEIGHTS are mutually exclusive");
        }
        trainer
            .optimiser
            .load_weights_from_file(path)
            .unwrap_or_else(|error| panic!("failed to load warm-start weights {path}: {error:?}"));
        println!("Warm-start weights: {path}");
    }

    trainer
        .optimiser
        .apply_weight_clip_ranges()
        .unwrap_or_else(|error| {
            panic!("failed to enforce threat-weight quantisation bounds: {error:?}")
        });
    println!(
        "Threat weights constrained to [{THREAT_WEIGHT_MIN}, {THREAT_WEIGHT_MAX}] to prevent i8 export saturation"
    );

    println!(
        "Architecture: {}",
        if with_p3 {
            "pairwise-mlp-p3"
        } else {
            "pairwise-mlp"
        }
    );
    println!(
        "Input layout: v3_10 ({} mirrored king buckets) + full_threats_v1",
        num_input_buckets
    );
    println!("FT width: {hl_size} per perspective (virtual-factorised)");
    println!(
        "Threat inputs: {}, scheme={}, packing hash={}",
        full_threats_v1::FEATURE_COUNT,
        full_threats_v1::SCHEME_ID,
        full_threats_v1::PACKING_HASH
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    println!("Output buckets: {O}");
    if with_p3 {
        println!(
            "P3-ANOVA Stage B: rank={}, atomic-pawns={}, context={}, moment-channels=same+adjacent, freeze_base={freeze_p3_base}",
            p3_inputs::RANK,
            p3_inputs::PAWN_COUNT,
            p3_inputs::CONTEXT_COUNT,
        );
    }
    println!(
        "Head: pairwise pool -> bucketed {hl_size} -> {dense1_size} -> dual {} -> {dense2_size} -> 1",
        2 * dense1_size
    );
    for path in dataset_paths {
        println!("  Dataset: {path}");
    }
    println!("Filter: ply>=16, not in check, |score|<=10000");

    match data_format {
        "binpack" => {
            use bullet_lib::value::loader::SfBinpackLoader;

            let binpack_buffer_mb = env_usize("SYK_BINPACK_BUFFER_MB", 1024);
            let binpack_threads = env_usize("SYK_BINPACK_THREADS", 4);
            println!(
                "Using SfBinpackLoader: buffer={}MB, threads={}",
                binpack_buffer_mb, binpack_threads
            );

            let dataloader = SfBinpackLoader::new_concat_multiple(
                dataset_paths,
                binpack_buffer_mb,
                binpack_threads,
                binpack_filter,
            );
            trainer.run(&schedule, &settings, &dataloader);
        }
        "bullet" => {
            println!("Using DirectSequentialDataLoader (bullet format)");
            let dataloader = DirectSequentialDataLoader::new(dataset_paths);
            trainer.run(&schedule, &settings, &dataloader);
        }
        other => panic!("unsupported SYK_DATA_FORMAT={other}"),
    }

    if let Ok(fen) = env::var("SYK_EVAL_FEN")
        && !fen.is_empty()
    {
        println!("SYK_EVAL_RAW={:.9}", trainer.eval(&fen));
    }
}

fn main() {
    let dataset_path = env_string("SYK_DATASET", "data/baseline.data");
    let architecture = env_string("SYK_ARCHITECTURE", "pairwise-mlp");
    let initial_lr = env_f32("SYK_LR_START", 0.001);
    let superbatches = env_usize("SYK_END_SUPERBATCH", 800);
    let start_superbatch = env_usize("SYK_START_SUPERBATCH", 1);
    let lr_origin_superbatch = env_usize("SYK_LR_ORIGIN_SUPERBATCH", 1);
    let lr_final_superbatch = env_usize("SYK_LR_FINAL_SUPERBATCH", superbatches);
    let final_lr = env_f32("SYK_LR_FINAL", initial_lr * 0.3f32.powi(5));
    let wdl_proportion = env_f32("SYK_WDL", 0.75);
    let save_rate = env_usize("SYK_SAVE_RATE", 10);
    let threads = env_usize("SYK_THREADS", 4);
    let batch_size = env_usize("SYK_BATCH_SIZE", 16_384);
    let batches_per_superbatch = env_usize("SYK_BATCHES_PER_SUPERBATCH", 6104);
    let output_dir = env_string("SYK_OUTPUT_DIR", "checkpoints");
    let net_id = env_string("SYK_NET_ID", "sykora_v8");
    let resume_from = env::var("SYK_RESUME").ok();
    let warm_start_weights = env::var("SYK_WARM_START_WEIGHTS").ok();
    let data_format = env_string("SYK_DATA_FORMAT", "bullet");
    let network_format = env_string("SYK_NETWORK_FORMAT", "syk8");
    let hl_size = env_usize("SYK_HIDDEN", 1024);
    let dense1_size = env_usize("SYK_DENSE1", 16);
    let dense2_size = env_usize("SYK_DENSE2", 32);
    let output_buckets = env_usize("SYK_OUTPUT_BUCKETS", 8);
    let bucket_layout_name = env_string("SYK_BUCKET_LAYOUT", "v3_10");
    let p3_rank = env_usize("SYK_P3_RANK", p3_inputs::RANK);
    let freeze_p3_base = env_usize("SYK_P3_FREEZE_BASE", 0) != 0;
    let with_p3 = architecture == "pairwise-mlp-p3";

    assert!(hl_size > 0, "SYK_HIDDEN must be > 0");
    assert!(dense1_size > 0, "SYK_DENSE1 must be > 0");
    assert!(dense2_size > 0, "SYK_DENSE2 must be > 0");
    assert!(batch_size > 0, "SYK_BATCH_SIZE must be > 0");
    assert!(
        lr_origin_superbatch <= start_superbatch,
        "SYK_LR_ORIGIN_SUPERBATCH must be <= SYK_START_SUPERBATCH"
    );
    assert!(
        lr_final_superbatch >= superbatches && lr_final_superbatch >= lr_origin_superbatch,
        "SYK_LR_FINAL_SUPERBATCH must cover the run and cosine origin"
    );
    assert!(
        batches_per_superbatch > 0,
        "SYK_BATCHES_PER_SUPERBATCH must be > 0"
    );
    assert!(
        hl_size.is_multiple_of(2),
        "pairwise architectures require an even FT width"
    );

    assert_eq!(
        bucket_layout_name, "v3_10",
        "only the proven v3_10 bucket layout is currently supported"
    );
    assert!(
        (network_format == "syk8" && architecture == "pairwise-mlp")
            || (network_format == "syk9" && with_p3),
        "SYKNNUE8 requires pairwise-mlp; SYKNNUE9 requires pairwise-mlp-p3"
    );
    assert_eq!(p3_rank, p3_inputs::RANK, "P3-ANOVA rank is fixed at 32");
    assert!(!freeze_p3_base || with_p3, "base freezing is P3-only");
    assert_eq!(
        output_buckets, 8,
        "SYKNNUE8 requires eight material output buckets"
    );
    assert!(
        hl_size == 1408 || hl_size == 1024 || hl_size == 768,
        "SYKNNUE8 registered widths are H=1408 (T1408), H=1024 (T1024), and H=768 (T768)"
    );

    let num_input_buckets = get_num_buckets(&BUCKET_LAYOUT_V3_10);
    let dataset_paths: Vec<&str> = dataset_path
        .split(';')
        .filter(|path| !path.is_empty())
        .collect();
    assert!(
        !dataset_paths.is_empty(),
        "SYK_DATASET must contain at least one path"
    );

    println!("Network format: {network_format}");
    println!(
        "Training samples per superbatch: {}",
        batch_size * batches_per_superbatch
    );

    run_network::<8>(
        num_input_buckets,
        &dataset_paths,
        &data_format,
        hl_size,
        dense1_size,
        dense2_size,
        initial_lr,
        final_lr,
        lr_origin_superbatch,
        lr_final_superbatch,
        start_superbatch,
        superbatches,
        batch_size,
        batches_per_superbatch,
        wdl_proportion,
        save_rate,
        threads,
        &output_dir,
        net_id,
        resume_from.as_deref(),
        warm_start_weights.as_deref(),
        with_p3,
        freeze_p3_base,
    );
}
