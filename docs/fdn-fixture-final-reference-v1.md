# FDN fixture reference completion checks

## Canonical reference integration, October 5

The FDN reference integration carries all 30 test/workflow/report paths from
qualified source `d98525a0` on a clean branch from `master`. All executable
reference inputs and the workflow are byte-identical to that source, which
passed 146 cases in 16 classes with zero failures, errors or skips. Java
production sources and Maven inputs match the default branch. The new head
still requires its hosted reference check and exact-head review before merge.

Former parent `codex/kernel-independent-trainer` contains unrelated embedded
kernel, manifests and general verification changes. They remain exclusively
in the unchanged parent branch and PR #1. Its shared commit hook requires full
local RL/kernel validation when those paths are staged; no hook was disabled.
The attempted local hook build was stopped, then the owned uncommitted cleanup
was restored. This clean FDN branch changes none of those gated paths.

FDN prefix PRs #3 through #15 remain preserved in branch
`codex/fdn-fixture-final-parity-v1` at `d98525a0`; their exact final tests,
workflow and reports are retained here, with this delivery note added. No
unrelated trainer change is merged or claimed superseded. Original branches,
worktrees and evidence remain retained. Kernel's canonical integration is #140.

The fixture milestone needs executed combat references. The earlier kernel
combat report inspected `DamageDistributionTest` and `FirstStrikeTest` but
did not execute them. Extended the existing focused hosted workflow to run those
classes, all twelve earlier FDN card comparison classes, London mulligans and
four new strict Foundations allocation positions. Main Java sources are unchanged.

| XMage scenario | Action and assertion | Kernel counterpart |
| --- | --- | --- |
| nontramplerCanAssignLessThanLethalToFirstBlocker | Six-power attacker assigns2/4 against two4/4 blockers; only the second dies and the player takes no damage. | arbitrary_gang_block_allocation_can_skip_lethal_on_the_first_blocker |
| tramplerCanOverassignSingleBlockerAndDealNoPlayerDamage | Assign all six to a4/4 blocker; blocker dies and player damage is zero. | trample_can_assign_excess_to_player_or_overassign_the_blocker |
| tramplerCanSkipFirstBlockerWhenAllDamageStaysOnBlockers | Six-power trampler assigns0/6 against two4/4 blockers; no damage goes to the player. | trample_requires_lethal_to_every_blocker_but_allows_player_zero |
| trampleDeathtouchAssignsOneToEachBlockerBeforeFourToPlayer | Six-power trampler with deathtouch assigns1/1 against two4/4 blockers; both die and the player takes four. | trample_and_deathtouch_require_one_for_each_blocker_before_excess |
| DamageDistributionTest | Blocked/unblocked double strike, trample, marked damage with indestructibility, deathtouch and simultaneous assignment. | blocked-creature, marked-lethal, double-strike and simultaneous combat cases |
| FirstStrikeTest | First-strike kills, priority between waves and gaining/losing strike abilities between waves. | first_strike_kill_removes_the_blocker_before_normal_damage; first_strike_has_a_real_priority_window_before_normal_damage; gaining_or_losing_first_strike_between_waves_does_not_change_normal_eligibility |

The printed reference creatures give the same power and blocker toughness as
the scripted kernel positions. These are targeted rules comparisons, not a
complete automatic differential harness. The Foundations allocation contract
comes from [Wizards' release notes](https://magic.wizards.com/en/news/feature/foundations-release-notes):
arbitrary blocker allocation, lethal to every blocker before trample reaches a
player, and one damage per blocker with deathtouch.

Run 37048866770 at 2d71bf29ecc executed all 146 cases across the 16 classes:
145 passed, one failed, zero errors/skips. The failure was the new nontrample
fixture: Vorstclaw is printed 7/7, so assigning 2/4 fails the requirement to
assign all seven damage. Replace it with the source-verified vanilla 6/6
Kindercatch. The other three new allocation cases, 24 existing combat cases,
seven London cases and all earlier FDN card classes passed. Retain the failed
run's XML and log; verify the corrected source before claiming the aggregate.

Corrected run 37049882555 at `5cc4decd8ffe16c317e1c2540d18e064c19648a4`
passed all 146 cases in all 16 selected classes, with zero failures, errors or
skips. This includes all four new allocation positions, 24 existing combat
cases, seven London cases and 111 earlier FDN card cases. Compiled merge commit:
`07cf417941e148f18b3310e2e178465193f0b657`. Independently checked all 23 source
hashes against Git blobs, every output digest, and each XML report's counts.

The inherited Temurin23.0.2 / Maven3.9.9 hosted workflow used one Maven thread
and two JVM processors. Actual build/dependency footprint was 691,086,878 bytes,
below the 8GiB cap; free storage was 90,055,323,648 bytes, above the 60GiB reserve.
Both local PCs' reservations were preserved.

Passing evidence is sealed at `E:/mtg-fdn-fixtures/fdn-fixture-final-reference-002`,
with an independently verified mirror at
`C:/Users/Jack/fdn-fixture-final-reference-002-sealed`. The failed first attempt
is retained under the corresponding `001` paths. Scratch was deleted only
after verification, with committed `docs/reports/fdn_fixture_final_reference_001_prune.json`
and `fdn_fixture_final_reference_002_prune.json` receipts. These comparisons
verify the targeted scenarios; complete kernel regression/CI remains separate.
