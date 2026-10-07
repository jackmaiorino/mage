# FDN life-gain creature parity

Six executed XMage cases match the kernel's life-gain creature scenarios.
Reference engine pin: `a5c90fe180021e70e2a644ade00eeab07f857a40`.
Test source commit: `b273481ec8a`. No XMage engine behavior was changed.

| Scenario | XMage result |
| --- | --- |
| Two separate gains in one turn | Exemplar gets two counters, draws one card and its controller reaches 22 life. |
| Felling Blow | Exemplar gets one counter, becomes 4/4, kills the 1/4 Spider and draws one card. |
| Opponent's turn | A second gain gets another counter and another draw; the limit resets each turn. |
| Unkicked Healer | 3/1 with lifelink; the creature card stays in the graveyard. |
| Kicked Healer returning an artifact | Ichor Wellspring returns and draws a card; a land and mana-value-four Exemplar stay in the graveyard. |
| Returning an Aura | Bind the Monster attaches to the opposing hexproof Bogle, taps it and deals one damage to its controller. The opposing protection creature is also present. |

`fdn-mage-lifegain-003` exited zero: six tests, zero failures, errors or
skips, reactor `BUILD SUCCESS`, 31.090 seconds. The first two attempts
exposed test setup errors in the host choice and the attachment assertion's
controller argument; both were corrected without changing engine behavior.

Small manifest: Java 23.0.2, Maven 3.9.9; CPU only, one Maven reactor thread,
JVM active processor count two; GPU ordinal none. Owned reference copy:
`C:/Users/hostuser/mage-fdn-counter-parity-codex`. Logs and XML remain outside
Git. Test source SHA-256:
`89a6a8f6c52645f783493d9f780c3fdda31bb2ce3f0912fabc558777835bb739`.
Surefire XML SHA-256:
`73d474c78750e1c83dd1d27d2516f51b122e081890cbdeed0574c1ab1374a8fa`.

Command:

```text
mvn -B -T 1 -pl Mage.Tests -am -Dtest=FdnLifegainCreaturesTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

These are rules comparisons for the named cases. They do not establish
complete FDN support, original-deck terminal gameplay or playing strength.
