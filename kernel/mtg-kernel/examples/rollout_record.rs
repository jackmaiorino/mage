//! Deterministic RL-contract rollout recorder.
//!
//! Run:
//! cargo run -p mtg-kernel --example rollout_record --manifest-path kernel/Cargo.toml -- \
//!   --matchup burn_mirror --games 4 --seed 5151 --out local-training/kernel_rl/smoke_v1

use mtg_kernel::rl::{
    build_rollout_records, build_run_manifest, git_metadata, write_rollout_artifacts,
    BURN_MIRROR_MATCHUP, DEFAULT_MAX_DECISIONS,
};
use std::path::PathBuf;

struct Args {
    matchup: String,
    games: u64,
    seed: u64,
    out: PathBuf,
    raw: Vec<String>,
}

fn main() {
    let args = match parse_args(std::env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}");
            eprintln!(
                "usage: rollout_record --matchup burn_mirror --games <n> --seed <u64> --out <dir>"
            );
            std::process::exit(2);
        }
    };
    if args.matchup != BURN_MIRROR_MATCHUP {
        eprintln!(
            "unsupported matchup {:?}; v1 supports exactly {:?}",
            args.matchup, BURN_MIRROR_MATCHUP
        );
        std::process::exit(2);
    }

    let (records, summaries) =
        match build_rollout_records(args.games, args.seed, DEFAULT_MAX_DECISIONS) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("rollout failed: {err}");
                std::process::exit(1);
            }
        };
    let manifest = build_run_manifest(
        args.games,
        args.seed,
        DEFAULT_MAX_DECISIONS,
        args.raw.clone(),
        &args.out,
        &summaries,
        git_metadata(),
    );
    if let Err(err) = write_rollout_artifacts(&args.out, &records, &manifest) {
        eprintln!("artifact write failed: {err}");
        std::process::exit(1);
    }

    println!(
        "wrote {} records for {} games to {} (p0_wins={} p1_wins={} draws={} halted={} decisions={})",
        records.len(),
        args.games,
        args.out.display(),
        manifest.aggregate.p0_wins,
        manifest.aggregate.p1_wins,
        manifest.aggregate.draws,
        manifest.aggregate.halted,
        manifest.aggregate.total_decisions
    );
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    let mut matchup = None;
    let mut games = None;
    let mut seed = None;
    let mut out = None;
    let mut i = 0;
    while i < raw.len() {
        let flag = raw[i].as_str();
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?
            .clone();
        match flag {
            "--matchup" => matchup = Some(value),
            "--games" => {
                games = Some(
                    value
                        .parse::<u64>()
                        .map_err(|e| format!("invalid --games value {value:?}: {e}"))?,
                )
            }
            "--seed" => {
                seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|e| format!("invalid --seed value {value:?}: {e}"))?,
                )
            }
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument {other:?}")),
        }
        i += 2;
    }

    let games = games.ok_or("missing --games")?;
    if games == 0 {
        return Err("--games must be positive".to_string());
    }
    Ok(Args {
        matchup: matchup.ok_or("missing --matchup")?,
        games,
        seed: seed.ok_or("missing --seed")?,
        out: out.ok_or("missing --out")?,
        raw,
    })
}
