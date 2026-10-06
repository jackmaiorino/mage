# FDN Koma reference cases

Nine executed XMage cases establish expected behavior for the next
original-UG fixture creature. Kernel counterparts remain pending.
Reference engine pin: `a5c90fe180021e70e2a644ade00eeab07f857a40`.
Test source commit: `e32e548e4d3`. No XMage engine behavior was changed.

| Scenario | XMage result |
| --- | --- |
| Unblocked combat | Koma is 8/12, the opponent falls to 12 life, and four 3/3 blue Serpent Coils with mana value zero enter. |
| Fully blocked combat | Tolarian Terror with four +1/+1 counters is 9/9 and takes all eight damage; no player damage or Coils. Koma survives. This uses a blocker already supported by the kernel. |
| Trample and simultaneous source death | Explicitly assign four to Treetop Snarespinner and four to its controller. Both creatures die, the opponent falls to 16 life, and Koma's queued trigger still creates four Coils. |
| Noncombat damage | Felling Blow makes Koma 9/13 and kills the opposing Snarespinner without creating Coils. |
| Counterspell targeting | Counterspell legally targets Koma and goes to the graveyard; Koma resolves onto the battlefield. |
| Counter unless payment | With one mana still available, Koma's caster is offered and declines Force Spike payment. Koma still resolves. |
| Opponent declines ward | Unsummon is countered and Koma remains on the battlefield. |
| Opponent pays ward | Unsummon resolves and puts Koma into its owner's hand. |
| Own-controller targeting | Unsummon resolves with only its ordinary one-mana cost and no ward choice. |

`fdn-mage-koma-004` exited zero: nine tests, zero failures, errors or
skips, reactor `BUILD SUCCESS`, 33.028 seconds (Surefire 20.186 seconds). Strict choice mode includes
the explicit trample allocation and payable Force Spike refusal. The prior
seven-case run passed. The intermediate nine-case attempt's missing
damage-allocation choice was corrected; its failed log and XML remain
preserved outside Git. The preceding nine-case run also passed; its XML
is preserved separately at `C:/Users/haley/fdn-mage-koma-003.xml`.

Small manifest: Java 23.0.2, Maven 3.9.9; CPU only, one reactor thread,
JVM active processor count two, GPU ordinal none. Owned reference copy:
`C:/Users/haley/mage-fdn-counter-parity-codex`. Logs and XML remain outside
Git. Test source SHA-256:
`797af8c555d7f6473f695b121ee9d51c7605a1dbc5cfd65c28f1e171bb0c6a8a`.
Surefire XML SHA-256:
`b8e605476dac54de297b1ea219621d98174bc488564a84aa0408c6e508f3a2b4`.

Command:

```text
mvn -B -T 1 -pl Mage.Tests -am -Dtest=FdnKomaCreaturesTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

These reference positions prepare the fixture implementation. Complete
kernel parity, original-deck terminals and full FDN coverage remain
outstanding.
