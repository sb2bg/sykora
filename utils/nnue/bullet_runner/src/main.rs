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
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use std::env;

// Proven v3 10-bucket layout (mirrored 32-entry half) — the SYKNNUE6 layout.
// This is the only supported layout for v6. See specs/syknnue6_spec.md §3.2.
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

/// Train a SYKNNUE6 net with `O` material-count output buckets and a
/// factorised sparse FT.
///
/// The factoriser (`l0f`) is a shared `768 → H` weight matrix trained across
/// all king buckets. At save time it is merged into the per-bucket weights
/// (`l0w`) via `.transform()`. The exported `raw.bin` contains `l0f`
/// followed by `l0w`; `checkpoint_raw_to_npz.py` merges them into the final
/// `ft_weights` tensor.
///
/// `O = 1` (with `MaterialCount::<1>`, which always selects bucket 0) is the
/// Stage-1 parity architecture. `O = 8` is the Stage-2/3 material-count head.
#[allow(clippy::too_many_arguments)]
fn run_syk6<const O: usize>(
    num_input_buckets: usize,
    dataset_paths: &[&str],
    data_format: &str,
    hl_size: usize,
    initial_lr: f32,
    final_lr: f32,
    start_superbatch: usize,
    superbatches: usize,
    wdl_proportion: f32,
    save_rate: usize,
    threads: usize,
    output_dir: &str,
    net_id: String,
    resume_from: Option<&str>,
) {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT_V3_10))
        .output_buckets(MaterialCount::<O>)
        .use_threads(threads)
        .save_format(&[
            // Factoriser merge: tile l0f across all buckets and add to l0w.
            // This produces the merged ft_weights that the runtime expects.
            SavedFormat::id("l0w")
                .transform(move |store, weights| {
                    let factoriser = store.get("l0f").values.repeat(num_input_buckets);
                    weights
                        .into_iter()
                        .zip(factoriser)
                        .map(|(a, b)| a + b)
                        .collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("outw").round().quantise::<i16>(64),
            SavedFormat::id("outb").round().quantise::<i32>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Factoriser: shared 768 → H weights, trained across all buckets.
            // This is the key component that v4/v5 dropped — without it,
            // each bucket's 768 weights are trained on only ~1/N of the data.
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            // Per-bucket FT weights + factoriser
            let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // Bucketed output head
            let out = builder.new_affine("out", 2 * hl_size, O);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);
            out.forward(hidden).select(output_buckets)
        });

    // Factoriser weights need tighter clipping (they're shared, so magnitudes
    // propagate across all buckets — same rationale as v3).
    let stricter_clipping = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer
        .optimiser
        .set_params_for_weight("l0w", stricter_clipping);
    trainer
        .optimiser
        .set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL {
            value: wdl_proportion,
        },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: superbatches,
        },
        save_rate,
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: 32,
    };

    if let Some(path) = resume_from {
        if !path.is_empty() {
            trainer.load_from_checkpoint(path);
        }
    }

    match data_format {
        "binpack" => {
            use bullet_lib::value::loader::SfBinpackLoader;

            let binpack_buffer_mb = env_usize("SYK_BINPACK_BUFFER_MB", 1024);
            let binpack_threads = env_usize("SYK_BINPACK_THREADS", 4);

            println!(
                "Using SfBinpackLoader: buffer={}MB, threads={}",
                binpack_buffer_mb, binpack_threads
            );
            println!(
                "Input layout: mirrored king buckets ({} buckets, v3_10), material output buckets ({})",
                num_input_buckets, O
            );
            println!("FT width: {} per perspective (factorised)", hl_size);
            println!("Head: bucketed linear {} -> 1", 2 * hl_size);
            for p in dataset_paths {
                println!("  Dataset: {}", p);
            }
            println!("Filter: ply>=16, not in check, |score|<=10000");

            let dataloader = SfBinpackLoader::new_concat_multiple(
                dataset_paths,
                binpack_buffer_mb,
                binpack_threads,
                binpack_filter,
            );

            trainer.run(&schedule, &settings, &dataloader);
        }
        _ => {
            println!("Using DirectSequentialDataLoader (bullet format)");
            println!(
                "Input layout: mirrored king buckets ({} buckets, v3_10), material output buckets ({})",
                num_input_buckets, O
            );
            println!("FT width: {} per perspective (factorised)", hl_size);
            println!("Head: bucketed linear {} -> 1", 2 * hl_size);
            for p in dataset_paths {
                println!("  Dataset: {}", p);
            }

            let dataloader = DirectSequentialDataLoader::new(dataset_paths);
            trainer.run(&schedule, &settings, &dataloader);
        }
    }
}

fn main() {
    let dataset_path = env_string("SYK_DATASET", "data/baseline.data");
    let initial_lr = env_f32("SYK_LR_START", 0.001);
    let superbatches = env_usize("SYK_END_SUPERBATCH", 640);
    let start_superbatch = env_usize("SYK_START_SUPERBATCH", 1);
    let final_lr = env_f32("SYK_LR_FINAL", initial_lr * 0.3f32.powi(5));
    let wdl_proportion = env_f32("SYK_WDL", 0.75);
    let save_rate = env_usize("SYK_SAVE_RATE", 1);
    let threads = env_usize("SYK_THREADS", 4);
    let output_dir = env_string("SYK_OUTPUT_DIR", "checkpoints");
    let net_id = env_string("SYK_NET_ID", "sykora_v6");
    let resume_from = env::var("SYK_RESUME").ok();
    let data_format = env_string("SYK_DATA_FORMAT", "bullet");
    let network_format = env_string("SYK_NETWORK_FORMAT", "syk6");
    let hl_size = env_usize("SYK_HIDDEN", 768);
    let output_buckets = env_usize("SYK_OUTPUT_BUCKETS", 8);
    let bucket_layout_name = env_string("SYK_BUCKET_LAYOUT", "v3_10");

    if bucket_layout_name != "v3_10" {
        panic!(
            "SYKNNUE6 only supports the v3_10 bucket layout (got {bucket_layout_name}); \
             see specs/syknnue6_spec.md §3.2"
        );
    }

    let num_input_buckets = get_num_buckets(&BUCKET_LAYOUT_V3_10);
    let dataset_paths: Vec<&str> = dataset_path.split(';').collect();

    println!("Network format: {}", network_format);
    println!("Bucket layout: v3_10 ({} buckets)", num_input_buckets);
    println!("Output buckets: {}", output_buckets);
    println!("FT width: {}", hl_size);

    if network_format != "syk6" {
        panic!("unsupported network format: {network_format}");
    }

    macro_rules! dispatch {
        ($o:literal) => {
            run_syk6::<$o>(
                num_input_buckets,
                &dataset_paths,
                &data_format,
                hl_size,
                initial_lr,
                final_lr,
                start_superbatch,
                superbatches,
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
        other => panic!("unsupported SYK_OUTPUT_BUCKETS={other}; SYKNNUE6 ladder uses 1 or 8"),
    }
}
