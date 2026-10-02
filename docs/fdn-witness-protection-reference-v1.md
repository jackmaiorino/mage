# Witness Protection reference comparisons

Sixteen strict XMage reference cases in `FdnWitnessProtectionTest` cover the
kernel batch's derived characteristics, card-type removal, preserved Legendary,
older/later Armor and Flying grants, retained counters, removed lord/static
abilities, Clockwork Percussionist death LKI with restoration control, Aura/host removal,
removed mana/draw abilities and removed ward.

Source authority: `Mage.Sets/src/mage/cards/w/WitnessProtection.java`.
The kernel counterpart is `mtg-kernel/tests/fdn_witness_protection_v1.rs` in
PR138. This changes tests and documentation only.

Execution is pending. Preserve the active Haley sequential comparison window
before running the small targeted Maven suite. The prior owned reference tree's
verified Java23.0.2/Maven3.9.9 main classes may be reused only after verifying
unchanged main-source hashes; use `-Dmaven.main.skip=true`, processors2 and T1.
Keep the exact test result, command, toolchain and source hashes in the completed
reference report. No claim of parity is made before observing passing results.
