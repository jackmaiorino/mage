# Celestial Armor reference cases

Ten strict XMage cases cover entry attachment and temporary protection,
flash on the opponent's turn, empty-board casting and exact three-land
payment, cleanup, re-equip, source removal before and after entry resolution,
target loss, lethal damage, destruction and zero toughness.

Reference sources: `Mage.Sets/src/mage/cards/c/CelestialArmor.java` and
`Mage/src/main/java/mage/abilities/common/EntersBattlefieldAttachToTarget.java`.
No main source changes. Reuse the verified main classes and compile this
new test with Java 23.0.2, Maven 3.9.9, one Maven thread and two JVM
processors. Guarded verification is pending. No GPU or research run.
