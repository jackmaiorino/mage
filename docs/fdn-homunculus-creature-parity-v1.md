# FDN Homunculus Horde parity

Five executed XMage cases match the kernel's Homunculus Horde scenarios.
Reference engine pin: `a5c90fe180021e70e2a644ade00eeab07f857a40`.
Test source commit: `4a5b56e0121`. No XMage engine behavior was changed.

| Scenario | XMage result |
| --- | --- |
| Tidings draws four | One original and one copy; four cards in hand. The new copy does not trigger retroactively. |
| Two original Hordes, Divination draws two | Four Hordes; two cards in hand. Strict choice mode orders both triggers. |
| Original and copy draw twice on opponent's turn | Four Hordes; four cards in hand. The token copy retains the trigger. |
| Counter on original before copying | Felling Blow makes the original 3/3; the copy is 2/2. Both have the printed name, blue color, Homunculus subtype and mana value four. |
| Opponent draws twice | The controller still has one Horde; no copy is created. |

`fdn-mage-homunculus-004` exited zero: five tests, zero failures, errors
or skips, reactor `BUILD SUCCESS`, 38.573 seconds. Earlier logs preserve
the corrected test setup: identify tokens through `PermanentToken`, use
the reference engine's literal trigger-order label, supply both Felling
Blow targets, and expect its single +1/+1 counter.

Small manifest: Java 23.0.2, Maven 3.9.9; CPU only, one reactor thread,
JVM active processor count two, GPU ordinal none. Owned reference copy:
`C:/Users/haley/mage-fdn-counter-parity-codex`. Logs and XML remain outside
Git. Test source SHA-256:
`bb40a4fe4134732d8ad16bf36d454fe566245c7fb21bdc42d9e2f2a23e7af096`.
Surefire XML SHA-256:
`5e4a722cb1566369111d07bfbce7d6ce1e2ad48be817479829f0f2f6a40e35c6`.

Command:

```text
mvn -B -T 1 -pl Mage.Tests -am -Dtest=FdnHomunculusCreaturesTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

These are rules comparisons for the named cases. They do not establish
complete FDN coverage, original-deck terminal gameplay or playing strength.
