# Witness Protection reference comparisons

Sixteen strict XMage reference cases in `FdnWitnessProtectionTest` cover the
kernel batch's derived characteristics, card-type removal, preserved Legendary,
older/later Armor and Flying grants, retained counters, removed lord/static
abilities, Clockwork Percussionist death LKI with restoration control, Aura/host removal,
removed mana/draw abilities and removed ward.

Source authority: `Mage.Sets/src/mage/cards/w/WitnessProtection.java`.
The kernel counterpart is `mtg-kernel/tests/fdn_witness_protection_v1.rs` in
PR138. This changes tests, focused CI and documentation only.

Execution is pending. The focused `FDN fixture reference tests` workflow builds
the reference from source on an Ubuntu24.04 hosted runner using Java23.0.2,
Maven3.9.9, processors2 and T1. It runs all sixteen Witness cases and the
existing `LondonMulliganTest`, and retains source hashes, logs and XML reports.
This lets correctness verification proceed while Jack and Haley remain reserved.
The workflow refuses missing reports, failing cases and skipped cases.
No claim of parity is made before observing passing results.
