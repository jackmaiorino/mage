//! First end-to-end oracle check: parse the entire golden-trace corpus and
//! report what the kernel will be asked to reproduce.
//! Run: cargo run --release --example corpus_stats -- <game_logs dir>

use mtg_kernel::trace;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let root = std::env::args().nth(1).map(PathBuf::from).expect("usage: corpus_stats <game_logs dir>");
    let (traces, errors) = trace::load_corpus(&root);
    println!("traces parsed: {}   parse errors: {}", traces.len(), errors.len());
    for e in errors.iter().take(5) {
        println!("  ERR {e}");
    }
    let mut by_matchup: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut action_types: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_decisions = 0usize;
    let mut with_winner = 0usize;
    for t in &traces {
        let e = by_matchup.entry(t.header.opp_deck.clone()).or_default();
        e.0 += 1;
        e.1 += t.decisions.len();
        total_decisions += t.decisions.len();
        if t.winner.is_some() {
            with_winner += 1;
        }
        for d in &t.decisions {
            *action_types.entry(d.action_type.clone()).or_default() += 1;
        }
        assert!(t.opening_library().is_some(), "trace without opening library: {}", t.source_path);
    }
    println!("total decisions: {total_decisions}   traces with recorded winner: {with_winner}");
    println!("\nper matchup (games, decisions):");
    for (m, (g, d)) in &by_matchup {
        println!("  {m:<28} {g:>3} games  {d:>6} decisions");
    }
    println!("\ndecision types:");
    for (a, n) in &action_types {
        println!("  {a:<28} {n}");
    }
}
