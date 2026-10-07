# FDN Uncharted Voyage reference cases

Eleven strict XMage gameplay cases pass on the compute host using Java 23.0.2,
Maven 3.9.9, one Maven task and two JVM processors. No GPU is used. The
reactor completed in 36.522 seconds; the eleven tests took 27.333 seconds,
with zero failures, errors or skips.

Source commit: `f103b1b58fe46c109cb7e1a6b8f51ee114885224`.
Test source SHA-256:
`e9beba07f0e439a7fb6b45c7c4871335987042d53e161e0aec4fb26f4319abdc`.

The cases cover opposing-owner top and bottom placement, keep and
graveyard surveil, own top placement followed by either surveil answer,
a stolen creature returning to its owner's library, a blink response
invalidating the only target, own and opposing token targets, an empty
caster library, and payable ward accepted or declined. Exact four-mana
payment and the additional two-mana ward payment are asserted.

The own-token case proves that placing a token on top does not prevent
surveilling the actual top card. The kernel retains the departed token
until its normal state-based cleanup and excludes it from the card
selection, consistent with comprehensive rules 111.6 through 111.8.

```powershell
mvn -B -T 1 -pl Mage.Tests -am -Dmaven.main.skip=true -Dtest=FdnUnchartedVoyageTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

The verification checkout reused the unchanged main classes validated by
the preceding FDN reference batches and recompiled the new tests. A cold
checkout must compile its main classes instead of using the skip flag.

Retained prefix: `C:/Users/hostuser/fdn-mage-voyage-003`.
Surefire XML SHA-256:
`cf029b76feceeb0f14d1470ba9d1e757cc3166cf0d0322a744150618d0ee4fcf`.
Log SHA-256:
`8bd3cb2694e32ceb70f83f0575e88d2542502eda071e0c3fa33e194cee3751b8`.

Preparation 001 passed nine cases; two setup errors were corrected by
creating opposing tokens in upkeep and clearing the caster's library
without subsequently adding cards. Preparation 002 stopped before tests
when its 512 MiB volume-allocation allowance was reached during a
concurrent Rust build. Preparation 003 used a declared 2 GiB allowance
and retained the 60 GiB reserve. All failed preparation logs remain kept.

These are bounded rules comparisons. The two original FDN decks still
require remaining cards and London mulligans before their full gameplay
milestone can be claimed.
