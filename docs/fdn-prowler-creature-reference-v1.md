# FDN Prowler reference comparisons

All twelve strict-choice XMage cases pass for Cackling Prowler at source pin
`a5c90fe180021e70e2a644ade00eeab07f857a40`. The reference creature is a green
4/3 Hyena Rogue for `{3}{G}` with ward `{2}` and a controller-end-step
intervening morbid trigger that adds one +1/+1 counter.

| Cases | Observed result |
| --- | --- |
| Exact four-mana cast and printed characteristics | Green 4/3, mana value four, Hyena and Rogue. |
| No death | No counter. |
| Own and opposing creature death | One counter, including an opposing death before Prowler enters. |
| Multiple deaths | One counter at end step. |
| Creature token death | Counts after the token disappears. |
| Noncreature death | Does not count. |
| Controller/end-step and next-turn boundary | No trigger on the opponent's end step; old death history does not carry over. |
| Death after end step begins | Cannot create a missed beginning-of-step trigger. |
| Blink in response | The new battlefield incarnation receives no old counter. |
| Ward decline/payment | Decline counters Unsummon; paying two permits its resolution. |

Run `fdn-mage-prowler-001` exited zero with reactor `BUILD SUCCESS` in
22.519 seconds. Surefire recorded twelve tests, zero failures, errors or
skips, in 16.233 seconds. Test source commit `a34d3e40b70`; source SHA-256
`9d27d2da25fea3afd229b22f31fc7b494d50c80e99b2ba4d85e692ada4e73d1e`.
Surefire XML SHA-256
`0ee0b1ee3eb1f9aa5446f6ed292d626e42aa45875b62259a8a20d84b8b5b7804`;
combined log SHA-256
`2ad6c7b569007f833e67639260ba21fb15a2a7bb3620b065579418039e366da3`.

The compute host used Java 23.0.2, Maven 3.9.9, one Maven thread, two active JVM
processors, no GPU, a 128 MiB build allowance and a 60 GiB storage reserve.
The guard monitors and can stop only its own identified child tree.
Unchanged main classes from the successful Kiora reference run were reused;
the new Prowler test was compiled. Logs, XML and the small manifest stay
outside Git under `C:/Users/hostuser/fdn-mage-prowler-001`.

The kernel checks the same rules contracts. Its ward tests use Snap, paying
that spell's `{1}{U}` cost before ward; XMage uses Unsummon for `{U}`. Its
incarnation test leaves for hand and returns directly, while XMage uses
Momentary Blink. Token species differ across the death checks. These
comparisons establish the shared rule behavior, not identical whole-game
traces. Original-fixture completion and playing strength remain unclaimed.
