# Exact XMage Rally vs CP7 Anchor

## Purpose

Measure the promoted Rally checkpoint against deterministic XMage CP7 skill 7 on the same Mono Red Rally deck. The harness swaps the candidate seat within each environment-seed pair and fails closed on action mapping, terminal winner, Rust step count, or physical-decision count mismatch.

## Frozen inputs

- XMage integration commit: `a6e9d2d223f` (`RL: Add exact CP7 Rally anchor`)
- Promoted run SHA-256: `2c9b7423004428c0e2bb138afafc15ec65957f6bd98c4587bea704fbf9549aae`
- Generation-384 checkpoint SHA-256: `4bd38cf3a9af3fb03fb04428fbc4286d4635007e848c7b9f0740122e430cbba8`
- Shadow scorer SHA-256: `b7b375142ffc9fc346fbfd0ffc87a09a48ec931ca2e45262d653d2f54e3b9435`
- Rally deck SHA-256: `4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834`
- Opponent: XMage `ComputerPlayer7`, skill 7
- Java: Oracle JDK 23.0.1 on Windows 11 amd64
- Maven: 3.9.9
- Scorer Rust toolchain cache: rustc 1.94.1 (`e408947bf`), LLVM 21.1.8, `x86_64-pc-windows-msvc`
- Linker: MSVC `link.exe`; the exact selected installed version was not captured. The evaluated executable is locked by its SHA-256 above.
- GPU ordinal: none. XMage and the shadow scorer ran CPU-only.
- Determinism flags: `AI_DETERMINISTIC_TIEBREAKS=true`, `AI_DETERMINISTIC_SEARCH=true`, `AI_DETERMINISTIC_MAX_NODES=5000`, `AI_MAX_THREADS_FOR_SIMULATIONS=1`

## Results

| Base seed | Status | Games | Candidate | On play | On draw | Candidate sweeps | Opponent sweeps | Split pairs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 930001 | exploratory | 16 | 5-11 (31.25%) | 0-8 | 5-3 | 0 | 3 | 5 |
| 940001 | exploratory | 32 | 15-17 (46.88%) | 11-5 | 4-12 | 2 | 3 | 11 |
| 950001 | fixed formal anchor | 64 | 26-38 (40.63%) | 11-21 | 15-17 | 3 | 9 | 20 |
| Combined | descriptive only | 112 | 46-66 (41.07%) | 22-34 | 24-32 | 5 | 15 | 36 |

All 112 games completed with exact bridge validation. The formal run covered 7,860 Rust policy steps and 6,476 physical decisions. Across all runs, the harness covered 112 natural terminals and 13,132 Rust policy steps.

The formal gate was frozen before base seed 950001:

- 40 or more wins: clearly ahead of CP7
- 24 or fewer wins: clearly behind CP7
- 25 through 39 wins: inconclusive or roughly competitive

The checkpoint landed in the inconclusive band at 26 wins. Its 95% Wilson interval is 29.5% to 52.9%. The combined paired outcome distribution is 15 opponent sweeps, 36 splits, and 5 candidate sweeps; a pair bootstrap gives a descriptive 95% interval of 33.9% to 48.2%. The combined inference is post hoc because it includes exploratory samples.

## Conclusion

The promoted checkpoint is playable and approximately CP7-competitive, but the best estimate is about nine percentage points below CP7. It has not demonstrated CP7 superiority and provides no pro-level claim. The early extreme play/draw splits reversed across fresh seeds and averaged to 39.3% on play versus 42.9% on draw, so there is no stable seat-bias finding.

The next rapid unit is policy improvement, not more evaluation of this checkpoint. Reuse the exact CP7 mapper to export Rust observation/action teacher examples, fit a small held-out supervised update from the promoted checkpoint, and compare it on the fixed 950001 pairs plus a fresh held-out seed block.
