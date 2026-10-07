# FDN Kiora reference cases

Nine strict-choice cases passed against XMage source pin
`a5c90fe180021e70e2a644ade00eeab07f857a40` in
`fdn-mage-kiora-003` on the compute host. The corresponding mtg-kernel batch
implements Kiora, the Rising Tide and Scion of the Deep in opt-in FDN v43.

| Case | Verified behavior |
| --- | --- |
| Entry loot | Three-mana casting, draw two before selected discard two, exact hand and graveyard results. |
| Empty hand | Both newly drawn cards are discarded without an optional refusal. |
| Six own cards | Seven opposing graveyard cards cannot enable Kiora's threshold. |
| Seven own cards | Accepted trigger creates exactly one legendary blue 8/8 Octopus with zero mana value. |
| Optional refusal | Declining creates no Scion. |
| Lost threshold | Exiling the graveyard in response suppresses the optional choice. |
| Source departure | Returning Kiora to hand leaves its queued token trigger usable. |
| Duplicate Scion | A second accepted trigger invokes the legend rule, leaving one Scion. |
| Second draw | Kiora's loot also queues the Mystic and Horde token triggers after discard. |

The kernel's lost-threshold case moves one graveyard card to hand; the
reference exiles all seven with Tormod's Crypt. The kernel's source-departure
case commits the zone change directly; the reference casts Unsummon. These
compare the intervening condition and queued-trigger contracts, rather than
identical whole-game traces. Save/restore is separately covered in the kernel.

The first attempt retained eight passing cases and one selector failure.
The Crypt response used the printed card name where the harness exposes a
source placeholder. The corrected selector identifies the only tap-cost
activation on player B's battlefield. The first corrected rerun was stopped
by the 60 GiB free-space guard before tests and is retained. No rules
assertion or strict-choice requirement was weakened.

Successful source commit: `2146ab93da227547d5c756fe9b6b6c42d0297004`.
Source SHA-256:
`696d8b75ea4999d69c8d4c94c8967fa2017d45a2d09bb4b18528b7993cf9648a`.
Surefire XML SHA-256:
`337c9d8e608e5f3fc010ec61dd6ceb746fd3f1ba470ef3f445a2d4d63ccd3e88`.
Output log SHA-256:
`e8cf6309377a636d68575a002fda063f658b66882393c77ad25ce5c6d10ecfb6`.
Java 23.0.2, Maven 3.9.9, two active processors, one Maven reactor worker,
no GPU. Nine tests, zero failures/errors/skips; Surefire 16.569 seconds,
reactor `BUILD SUCCESS` in 23.521 seconds.

The successful retry recompiles the changed test class and reuses the
unchanged main classes already exercised by the first attempt. Its
[`maven.main.skip`](https://maven.apache.org/plugins-archives/maven-compiler-plugin-3.8.1/compile-mojo.html#skipMain)
setting avoids a redundant main-source compilation. Owned cold Mage classes
were compressed before retry; the complete sorted class-file content hash
was identical before and after. The guarded launcher records source/log/XML
hashes and checks the 60 GiB reserve every three seconds, stopping only its
identified child tree. Raw manifests, logs and XML remain outside Git under
`C:/Users/hostuser/fdn-mage-kiora-003*`.

These bounded rules comparisons establish no full-set Limited support,
original-fixture natural-terminal result or playing-strength claim.
