use bullet_lib::{
    game::{
        formats::sfbinpack::TrainingDataEntry,
        inputs::{ChessBucketsMirrored, get_num_buckets},
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
const POOL_QUANT: i32 = 128;
const DENSE_QUANT_I16: i16 = 64;
const DENSE_QUANT_I32: i32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Architecture {
    Linear,
    PairwiseLinear,
    PairwiseMlp,
}

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

impl Architecture {
    fn parse(value: &str) -> Self {
        match value {
            "linear" => Self::Linear,
            "pairwise-linear" => Self::PairwiseLinear,
            "pairwise-mlp" => Self::PairwiseMlp,
            other => panic!(
                "unsupported SYK_ARCHITECTURE={other}; expected linear, pairwise-linear, or pairwise-mlp"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::PairwiseLinear => "pairwise-linear",
            Self::PairwiseMlp => "pairwise-mlp",
        }
    }

    fn uses_pairwise(self) -> bool {
        self != Self::Linear
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

fn saved_format(architecture: Architecture, num_input_buckets: usize) -> Vec<SavedFormat> {
    let mut format = vec![
        // Keep the training-only factoriser in raw.bin. Bullet applies
        // transforms only to quantised.bin; without this explicit entry the
        // raw checkpoint contains l0w residuals but not the l0f values needed
        // to reconstruct effective FT weights. The empty transform excludes
        // this training-only tensor from quantised.bin while the raw writer,
        // which ignores transforms, still serialises it.
        SavedFormat::id("l0f").transform(|_, _| Vec::new()),
        SavedFormat::id("l0w")
            .transform(move |store, weights| {
                let factoriser = store.get("l0f").values.repeat(num_input_buckets);
                weights
                    .into_iter()
                    .zip(factoriser)
                    .map(|(bucket, shared)| bucket + shared)
                    .collect()
            })
            .round()
            .quantise::<i16>(FT_QUANT),
        SavedFormat::id("l0b").round().quantise::<i16>(FT_QUANT),
    ];

    match architecture {
        Architecture::Linear => {
            format.extend([
                SavedFormat::id("outw")
                    .round()
                    .quantise::<i16>(DENSE_QUANT_I16),
                SavedFormat::id("outb")
                    .round()
                    .quantise::<i32>(i32::from(FT_QUANT) * DENSE_QUANT_I32),
            ]);
        }
        Architecture::PairwiseLinear => {
            format.extend([
                SavedFormat::id("outw")
                    .transpose()
                    .round()
                    .quantise::<i16>(DENSE_QUANT_I16),
                SavedFormat::id("outb")
                    .round()
                    .quantise::<i32>(POOL_QUANT * DENSE_QUANT_I32),
            ]);
        }
        Architecture::PairwiseMlp => {
            format.extend([
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
            ]);
        }
    }

    format
}

#[allow(clippy::too_many_arguments)]
fn run_network<const O: usize>(
    num_input_buckets: usize,
    dataset_paths: &[&str],
    data_format: &str,
    architecture: Architecture,
    hl_size: usize,
    dense1_size: usize,
    dense2_size: usize,
    initial_lr: f32,
    final_lr: f32,
    lr_origin_superbatch: usize,
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
) {
    let save_format = saved_format(architecture, num_input_buckets);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT_V3_10))
        .output_buckets(MaterialCount::<O>)
        .use_threads(threads)
        .save_format(&save_format)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(move |builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            match architecture {
                Architecture::Linear => {
                    let out = builder.new_affine("out", 2 * hl_size, O);
                    let stm_hidden = l0.forward(stm_inputs).screlu();
                    let ntm_hidden = l0.forward(ntm_inputs).screlu();
                    out.forward(stm_hidden.concat(ntm_hidden))
                        .select(output_buckets)
                }
                Architecture::PairwiseLinear => {
                    let out = builder.new_affine("out", hl_size, O);
                    let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
                    let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
                    out.forward(stm_hidden.concat(ntm_hidden))
                        .select(output_buckets)
                }
                Architecture::PairwiseMlp => {
                    let l1 = builder.new_affine("l1", hl_size, O * dense1_size);
                    let l2 = builder.new_affine("l2", 2 * dense1_size, O * dense2_size);
                    let l3 = builder.new_affine("l3", dense2_size, O);

                    let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
                    let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
                    let pooled = stm_hidden.concat(ntm_hidden);

                    let z1 = l1.forward(pooled).select(output_buckets);
                    let dual = z1.crelu().concat(z1.abs_pow(2.0).crelu());
                    let hidden = l2.forward(dual).select(output_buckets).screlu();
                    l3.forward(hidden).select(output_buckets)
                }
            }
        });

    let ft_clipping = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", ft_clipping);
    trainer.optimiser.set_params_for_weight("l0f", ft_clipping);

    if architecture == Architecture::PairwiseMlp {
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
            final_superbatch: superbatches,
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

    println!("Architecture: {}", architecture.name());
    println!(
        "Input layout: v3_10 ({} mirrored king buckets)",
        num_input_buckets
    );
    println!("FT width: {hl_size} per perspective (factorised)");
    println!("Output buckets: {O}");
    match architecture {
        Architecture::Linear => println!("Head: bucketed linear {} -> 1", 2 * hl_size),
        Architecture::PairwiseLinear => {
            println!("Head: pairwise pool -> bucketed linear {hl_size} -> 1")
        }
        Architecture::PairwiseMlp => println!(
            "Head: pairwise pool -> bucketed {hl_size} -> {dense1_size} -> dual {} -> {dense2_size} -> 1",
            2 * dense1_size
        ),
    }
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
    let architecture = Architecture::parse(&env_string("SYK_ARCHITECTURE", "pairwise-mlp"));
    let initial_lr = env_f32("SYK_LR_START", 0.001);
    let superbatches = env_usize("SYK_END_SUPERBATCH", 800);
    let start_superbatch = env_usize("SYK_START_SUPERBATCH", 1);
    let lr_origin_superbatch = env_usize("SYK_LR_ORIGIN_SUPERBATCH", 1);
    let final_lr = env_f32("SYK_LR_FINAL", initial_lr * 0.3f32.powi(5));
    let wdl_proportion = env_f32("SYK_WDL", 0.75);
    let save_rate = env_usize("SYK_SAVE_RATE", 10);
    let threads = env_usize("SYK_THREADS", 4);
    let batch_size = env_usize("SYK_BATCH_SIZE", 16_384);
    let batches_per_superbatch = env_usize("SYK_BATCHES_PER_SUPERBATCH", 6104);
    let output_dir = env_string("SYK_OUTPUT_DIR", "checkpoints");
    let net_id = env_string("SYK_NET_ID", "sykora_v7");
    let resume_from = env::var("SYK_RESUME").ok();
    let data_format = env_string("SYK_DATA_FORMAT", "bullet");
    let network_format = env_string("SYK_NETWORK_FORMAT", "syk7");
    let hl_size = env_usize("SYK_HIDDEN", 1024);
    let dense1_size = env_usize("SYK_DENSE1", 16);
    let dense2_size = env_usize("SYK_DENSE2", 32);
    let output_buckets = env_usize("SYK_OUTPUT_BUCKETS", 8);
    let bucket_layout_name = env_string("SYK_BUCKET_LAYOUT", "v3_10");

    assert!(hl_size > 0, "SYK_HIDDEN must be > 0");
    assert!(dense1_size > 0, "SYK_DENSE1 must be > 0");
    assert!(dense2_size > 0, "SYK_DENSE2 must be > 0");
    assert!(batch_size > 0, "SYK_BATCH_SIZE must be > 0");
    assert!(
        lr_origin_superbatch <= start_superbatch,
        "SYK_LR_ORIGIN_SUPERBATCH must be <= SYK_START_SUPERBATCH"
    );
    assert!(
        batches_per_superbatch > 0,
        "SYK_BATCHES_PER_SUPERBATCH must be > 0"
    );
    if architecture.uses_pairwise() {
        assert!(
            hl_size.is_multiple_of(2),
            "pairwise architectures require an even FT width"
        );
    }

    assert_eq!(
        bucket_layout_name, "v3_10",
        "only the proven v3_10 bucket layout is currently supported"
    );
    match (network_format.as_str(), architecture) {
        ("syk6", Architecture::Linear) => {}
        ("syk7", Architecture::PairwiseLinear | Architecture::PairwiseMlp) => {}
        _ => panic!(
            "network format/architecture mismatch: format={network_format}, architecture={}",
            architecture.name()
        ),
    }

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

    macro_rules! dispatch {
        ($o:literal) => {
            run_network::<$o>(
                num_input_buckets,
                &dataset_paths,
                &data_format,
                architecture,
                hl_size,
                dense1_size,
                dense2_size,
                initial_lr,
                final_lr,
                lr_origin_superbatch,
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
            )
        };
    }

    match output_buckets {
        1 => dispatch!(1),
        8 => dispatch!(8),
        other => panic!("unsupported SYK_OUTPUT_BUCKETS={other}; expected 1 or 8"),
    }
}
