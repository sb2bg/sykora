use bullet_lib::{
    game::formats::{
        bulletformat::{BulletFormat, ChessBoard},
        sfbinpack::TrainingDataEntry,
    },
    value::loader::{DataLoader, SfBinpackLoader},
};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

struct Args {
    inputs: Vec<String>,
    output: PathBuf,
    positions: usize,
    buffer_mb: usize,
    threads: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: sample_binpack --input FILE [--input FILE ...] --output FILE \
         [--positions N] [--buffer-mb N] [--threads N]"
    );
    std::process::exit(2);
}

fn parse_usize(name: &str, value: Option<String>) -> usize {
    value
        .unwrap_or_else(|| usage())
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
}

fn parse_args() -> Args {
    let mut inputs = Vec::new();
    let mut output = None;
    let mut positions = 262_144;
    let mut buffer_mb = 512;
    let mut threads = 4;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => inputs.push(args.next().unwrap_or_else(|| usage())),
            "--output" => output = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--positions" => positions = parse_usize("--positions", args.next()),
            "--buffer-mb" => buffer_mb = parse_usize("--buffer-mb", args.next()),
            "--threads" => threads = parse_usize("--threads", args.next()),
            "-h" | "--help" => usage(),
            other => panic!("unknown argument: {other}"),
        }
    }

    if inputs.is_empty() || output.is_none() || positions == 0 || buffer_mb == 0 || threads == 0 {
        usage();
    }

    Args {
        inputs,
        output: output.unwrap(),
        positions,
        buffer_mb,
        threads,
    }
}

fn binpack_filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
}

fn main() {
    let args = parse_args();
    for input in &args.inputs {
        assert!(
            std::path::Path::new(input).is_file(),
            "input not found: {input}"
        );
    }
    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).expect("failed to create validation output directory");
    }
    assert!(
        !args.output.exists(),
        "refusing to overwrite validation sample: {}",
        args.output.display()
    );

    let mut partial_name = args.output.as_os_str().to_os_string();
    partial_name.push(".partial");
    let partial_path = PathBuf::from(partial_name);

    let paths: Vec<&str> = args.inputs.iter().map(String::as_str).collect();
    let loader =
        SfBinpackLoader::new_concat_multiple(&paths, args.buffer_mb, args.threads, binpack_filter);
    let output = File::create(&partial_path).expect("failed to create partial validation sample");
    let mut writer = BufWriter::new(output);
    let mut written = 0usize;
    let mut results = [0usize; 3];
    let mut material_buckets = [0usize; 8];

    loader.map_batches(0, 16_384, |batch: &[ChessBoard]| {
        let take = usize::min(args.positions - written, batch.len());
        let selected = &batch[..take];
        for position in selected {
            results[usize::from(position.result)] += 1;
            let non_king_count = position.occ.count_ones().saturating_sub(2) as usize;
            material_buckets[usize::min(non_king_count / 4, 7)] += 1;
        }
        BulletFormat::write_to_bin(&mut writer, selected)
            .expect("failed to write validation positions");
        written += take;
        written >= args.positions
    });
    writer.flush().expect("failed to flush validation sample");
    drop(writer);

    assert_eq!(written, args.positions, "validation sampler ended early");
    std::fs::rename(&partial_path, &args.output)
        .expect("failed to atomically publish validation sample");
    println!("Wrote {} positions to {}", written, args.output.display());
    println!(
        "Results: losses={}, draws={}, wins={}",
        results[0], results[1], results[2]
    );
    println!("Material buckets: {material_buckets:?}");
}
