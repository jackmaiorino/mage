# Celestial Armor reference cases

Ten strict XMage cases cover entry attachment and temporary protection,
flash on the opponent's turn, empty-board casting and exact three-land
payment, cleanup, re-equip, source removal before and after entry resolution,
target loss, lethal damage, destruction and zero toughness.

Reference sources: `Mage.Sets/src/mage/cards/c/CelestialArmor.java` and
`Mage/src/main/java/mage/abilities/common/EntersBattlefieldAttachToTarget.java`.
No main source changes. Reuse the verified main classes and compile this
new test with Java 23.0.2, Maven 3.9.9, one Maven thread and two JVM
processors. Guarded prefix `C:/Users/hostuser/fdn-mage-armor-001` exited zero: ten tests,
zero failures/errors/skips, Maven48.772seconds. Source commit
`90b592325b7b01ef0080c8ed20d862adff83eb16`. Projected1GiB, cap2GiB,
reserve60GiB. No GPU or research run.

SHA-256 pins:
- Test source: `ef41112f95268a52106ef256f045c824906d3fcb1d4916c1f7c98435f119e1e2`
- XML: `ff88565aa6291450e2154728fdf33aee578d6c3e4a749b8f66511d4657cace86`
- Output: `2045f24a4649856451410dedd3175081c0e9237016af4a5ca7cb5cf48bf2f318`

XML, log and manifest are copied to the maintainer's and verified against the remote
manifest. Reference source hashes match between both machines. Neither
main source changed. The test compiled against the verified reused main
classes. These are bounded rules comparisons for mtg-kernel issue110.
