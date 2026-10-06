# FDN draw creature parity

Five executed XMage cases match the kernel's Strix Lookout and Mischievous
Mystic scenarios. Reference engine pin:
`a5c90fe180021e70e2a644ade00eeab07f857a40`.
Test source commit: `3bf7ea717e701b1059e5540ed943f5702d9ef63a`.
No XMage engine behavior was changed.

| Scenario | XMage result |
| --- | --- |
| Four draws from Tidings | One blue 1/1 flying Faerie; four cards in hand. |
| Two Mystics, two draws from Divination | Two Faeries; two cards in hand. Strict choice mode explicitly orders the simultaneous triggers. |
| Draws on both players' turns | Divination on the controller's turn and two Think Twice spells on the opponent's turn create two Faeries in total; four cards in hand. |
| Lookout attacks, then loots in second main | Opponent reaches 19 life; Lookout stays available through the attack, then taps to loot; discarded Forest is in the graveyard and one card remains in hand. |
| Lookout draws the second card, then discards | One Faerie; Lookout tapped; discarded Forest in the graveyard; two cards in hand. |

`fdn-mage-draw-003` exited zero: five tests, zero failures, errors or
skips, reactor `BUILD SUCCESS`, 36.628 seconds. The first attempt found
an incorrect tapped-assertion signature; the second required an explicit
ordering choice for the two simultaneous Mystic triggers. Both test setup
errors were corrected, and their logs remain preserved.

Small manifest: Java 23.0.2, Maven 3.9.9; CPU only, one Maven reactor
thread, JVM active processor count two; GPU ordinal none. Owned reference
copy: `C:/Users/haley/mage-fdn-counter-parity-codex`. Logs and XML remain
outside Git. Test source SHA-256:
`447980bc1d5249cfc225245bf57a36cca1f23050988bc981329428a097f15cef`.
Surefire XML SHA-256:
`f1f69d8accad5d91c51b3ac6d29438ce539fdd94740c6d1320036febe62ecaee`.

Command:

```text
mvn -B -T 1 -pl Mage.Tests -am -Dtest=FdnDrawCreaturesTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

These comparisons cover the named rules scenarios. They do not establish
complete FDN support, original-deck terminal gameplay or playing strength.
