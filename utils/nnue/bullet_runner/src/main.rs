use bullet_lib::{
    game::{
        formats::sfbinpack::{
            TrainingDataEntry,
        },
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
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

const OUTPUT_BUCKETS: usize = 8;

#[rustfmt::skip]
const BUCKET_LAYOUT_SYKORA10: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

#[rustfmt::skip]
const BUCKET_LAYOUT_SYKORA16: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 3, 3,
    4, 4, 5, 5,
    6, 6, 7, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    14, 14, 15, 15,
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

fn selected_bucket_layout(name: &str) -> [usize; 32] {
    match name {
        "sykora16" => BUCKET_LAYOUT_SYKORA16,
        _ => BUCKET_LAYOUT_SYKORA10,
    }
}

fn selected_bucket_layout_name(name: &str) -> &'static str {
    match name {
        "sykora16" => "sykora16",
        _ => "sykora10",
    }
}

fn binpack_filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
}

fn run_syk3(
    bucket_layout: [usize; 32],
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
        .inputs(ChessBucketsMirrored::new(bucket_layout))
        .use_threads(threads)
        .save_format(&[
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
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, hl_size);
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
                "Input layout: mirrored king buckets ({} buckets)",
                num_input_buckets
            );
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
                "Input layout: mirrored king buckets ({} buckets)",
                num_input_buckets
            );
            for p in dataset_paths {
                println!("  Dataset: {}", p);
            }

            let dataloader = DirectSequentialDataLoader::new(dataset_paths);
            trainer.run(&schedule, &settings, &dataloader);
        }
    }
}

fn run_syk4(
    bucket_layout: [usize; 32],
    num_input_buckets: usize,
    dataset_paths: &[&str],
    data_format: &str,
    hl_size: usize,
    dense_l1: usize,
    dense_l2: usize,
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
        .inputs(ChessBucketsMirrored::new(bucket_layout))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .use_threads(threads)
        .save_format(&[
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
            SavedFormat::id("l1w").transpose().round().quantise::<i16>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose().round().quantise::<i16>(64),
            SavedFormat::id("l2b"),
            SavedFormat::id("outw").transpose().round().quantise::<i16>(64),
            SavedFormat::id("outb"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, hl_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", hl_size, OUTPUT_BUCKETS * dense_l1);
            let l2 = builder.new_affine("l2", dense_l1, OUTPUT_BUCKETS * dense_l2);
            let out = builder.new_affine("out", dense_l2, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            let dense_1 = l1.forward(hidden_layer).select(output_buckets).screlu();
            let dense_2 = l2.forward(dense_1).select(output_buckets).screlu();
            out.forward(dense_2).select(output_buckets)
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
                "Input layout: mirrored king buckets ({} buckets), piece-count heads ({})",
                num_input_buckets, OUTPUT_BUCKETS
            );
            println!(
                "Dense head: {} -> {} -> 1",
                dense_l1, dense_l2
            );
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
                "Input layout: mirrored king buckets ({} buckets), piece-count heads ({})",
                num_input_buckets, OUTPUT_BUCKETS
            );
            println!(
                "Dense head: {} -> {} -> 1",
                dense_l1, dense_l2
            );
            for p in dataset_paths {
                println!("  Dataset: {}", p);
            }

            let dataloader = DirectSequentialDataLoader::new(dataset_paths);
            trainer.run(&schedule, &settings, &dataloader);
        }
    }
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
    let network_format = env_string("SYK_NETWORK_FORMAT", "syk3");
    let bucket_layout_name = env_string(
        "SYK_BUCKET_LAYOUT",
        if network_format == "syk4" {
            "sykora16"
        } else {
            "sykora10"
        },
    );
    let dense_l1 = env_usize("SYK_DENSE_L1", 16);
    let dense_l2 = env_usize("SYK_DENSE_L2", 32);

    let bucket_layout = selected_bucket_layout(&bucket_layout_name);
    let bucket_layout_name = selected_bucket_layout_name(&bucket_layout_name);
    let num_input_buckets = get_num_buckets(&bucket_layout);
    let dataset_paths: Vec<&str> = dataset_path.split(';').collect();

    println!("Network format: {}", network_format);
    println!("Bucket layout: {}", bucket_layout_name);

    match network_format.as_str() {
        "syk4" => run_syk4(
            bucket_layout,
            num_input_buckets,
            &dataset_paths,
            &data_format,
            hl_size,
            dense_l1,
            dense_l2,
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
        ),
        _ => run_syk3(
            bucket_layout,
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
        ),
    }
}
