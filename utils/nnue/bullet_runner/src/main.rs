use bullet_lib::{
    game::{
        formats::sfbinpack::{
            chess::{piecetype::PieceType as SfPieceType, r#move::MoveType},
            TrainingDataEntry,
        },
        inputs::{get_num_buckets, ChessBucketsMirrored},
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use std::env;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

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

fn main() {
    let hl_size = env_usize("SYK_HIDDEN", 128);
    let dataset_path = env_string("SYK_DATASET", "data/baseline.data");
    let initial_lr = env_f32("SYK_LR_START", 0.001);
    let superbatches = env_usize("SYK_END_SUPERBATCH", 320);
    let start_superbatch = env_usize("SYK_START_SUPERBATCH", 1);
    let final_lr = env_f32("SYK_LR_FINAL", initial_lr * 0.3f32.powi(5));
    let wdl_proportion = env_f32("SYK_WDL", 0.75);
    let save_rate = env_usize("SYK_SAVE_RATE", 1);
    let threads = env_usize("SYK_THREADS", 4);
    let output_dir = env_string("SYK_OUTPUT_DIR", "checkpoints");
    let net_id = env_string("SYK_NET_ID", "sykora_bucketed");
    let resume_from = env::var("SYK_RESUME").ok();
    let data_format = env_string("SYK_DATA_FORMAT", "bullet");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .use_threads(threads)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                    weights
                        .into_iter()
                        .zip(factoriser)
                        .map(|(a, b)| a + b)
                        .collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", 2 * hl_size, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

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
        output_directory: &output_dir,
        batch_queue_size: 32,
    };

    if let Some(path) = resume_from.as_deref() {
        if !path.is_empty() {
            trainer.load_from_checkpoint(path);
        }
    }

    let dataset_paths: Vec<&str> = dataset_path.split(';').collect();

    match data_format.as_str() {
        "binpack" => {
            use bullet_lib::value::loader::SfBinpackLoader;

            let binpack_buffer_mb = env_usize("SYK_BINPACK_BUFFER_MB", 1024);
            let binpack_threads = env_usize("SYK_BINPACK_THREADS", 4);

            println!(
                "Using SfBinpackLoader: buffer={}MB, threads={}",
                binpack_buffer_mb, binpack_threads
            );
            println!(
                "Input layout: mirrored king buckets ({} buckets)",
                NUM_INPUT_BUCKETS
            );
            for p in &dataset_paths {
                println!("  Dataset: {}", p);
            }

            fn binpack_filter(entry: &TrainingDataEntry) -> bool {
                entry.ply >= 16
                    && !entry.pos.is_checked(entry.pos.side_to_move())
                    && entry.score.unsigned_abs() <= 10000
            }
            println!("Filter: ply>=16, not in check, |score|<=10000, normal moves, quiet only");

            let dataloader = SfBinpackLoader::new_concat_multiple(
                &dataset_paths,
                binpack_buffer_mb,
                binpack_threads,
                binpack_filter,
            );

            trainer.run(&schedule, &settings, &dataloader);
        }
        _ => {
            println!("Using DirectSequentialDataLoader (bullet format)");
            println!(
                "Input layout: mirrored king buckets ({} buckets)",
                NUM_INPUT_BUCKETS
            );
            for p in &dataset_paths {
                println!("  Dataset: {}", p);
            }

            let dataloader = DirectSequentialDataLoader::new(&dataset_paths);

            trainer.run(&schedule, &settings, &dataloader);
        }
    }
}
