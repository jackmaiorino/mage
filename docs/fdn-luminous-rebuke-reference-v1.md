# FDN Luminous Rebuke reference checks

Eleven strict-choice XMage cases pass for the kernel's Luminous Rebuke
slice. Tapped creatures cost two mana, untapped creatures cost five, and
having five available does not remove the discount. The white pip remains
required. Own creatures and legendary creatures are legal targets.

An actual Quirion Ranger response returns its Forest and untaps the target;
Rebuke still destroys that creature after the paid two-mana cast. A
Momentary Blink response preserves the returned creature's new incarnation.
Ward two is paid separately after the discounted cast, and declining ward
counters the spell. A Rebuke death enables a friendly Prowler's end-step
counter.

The kernel's fifteen focused checks additionally cover offer/menu
affordability, rejected actions without state mutation, protected targets,
target departure/reentry, and pending-target and ward save/restore. Kernel
save/restore is checked directly; the XMage reference cases compare the
shared rules behavior rather than whole-engine traces.

Source is `904d26386d3d64589b319c9a5464c3b2ec333239`,
`Mage.Tests/src/test/java/org/mage/test/cards/single/fdn/FdnLuminousRebukeTest.java`.
The upstream card implementation is pinned at
`a5c90fe180021e70e2a644ade00eeab07f857a40`,
`Mage.Sets/src/mage/cards/l/LuminousRebuke.java`.
Test-source SHA-256:
`b6ef8bb7a1f92b31e39b5db3d4dce8b69ea48ea21de387c792c8fbc0eae1f57f`.

`fdn-mage-rebuke-001` exited zero on HaleysPC. Surefire reports eleven tests,
zero failures, errors or skips, in 29.859 seconds; the reactor completed in
38.143 seconds. Surefire XML SHA-256:
`72224af6359caf54980b4e7eecd863f4a1ee51fcc0c3a0ff6828568ef5b29341`.
Log SHA-256:
`a747ee5e8dfc646871b59c71af9a4350c37e05d5ba8138580a7c30d4c6d934f7`.

Java 23.0.2, Maven 3.9.9, one reactor worker and two active JVM processors;
no GPU. Main source is unchanged from the verified Prowler build, so
`-Dmaven.main.skip=true` reuses those classes while recompiling this test.
The manifest declares a 128 MiB incremental allocation cap and 60 GiB free
reserve, monitored against the owned process tree. No storage stop occurred.
Logs and manifests remain outside Git. No playing-strength claim.
