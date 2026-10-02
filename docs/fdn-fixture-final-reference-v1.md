# FDN fixture reference completion checks

The fixture milestone needs executed combat references. The earlier kernel
combat report inspected `DamageDistributionTest` and `FirstStrikeTest` but
did not execute them. Extend the existing focused hosted workflow to run those
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

Execution is pending. Use the inherited Java23.0.2 / Maven3.9.9 hosted workflow,
one Maven thread, two JVM processors, the8GiB storage cap and60GiB reserve.
Preserve both local PCs' actual reservations. Require every selected class's XML
report with nonzero tests and zero failures/errors/skips; retain source/output
hashes and the exact completed counts before claiming success.
