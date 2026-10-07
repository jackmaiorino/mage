# Foundations counter-creature comparisons

Five deterministic XMage scenarios pass at source `2a0b7edfb59`, based on
`a5c90fe180021e70e2a644ade00eeab07f857a40`. They compare required outcomes
with mtg-kernel's `fdn_counter_creatures_v1` tests at `109bd020`.

| Scenario | Shared expected outcome |
| --- | --- |
| Colony without kicker | Zero counters, 2/2, no trample. |
| Colony with kicker | Two counters, 4/4, trample. |
| Continuous Colony grant | Own Elf with counters gains trample; own counterless Vanguard and opposing Elf with counters do not. |
| Hydra entry | Resolves alive with one counter and 1/1 stats before state-based actions. |
| Controlled landfall | Two own land entries take counters from one to four; an opposing entry contributes no doubling. |

The compute host command, from the owned source copy:

```text
mvn -B -T 1 -pl Mage.Tests -am -Dtest=FdnCounterCreaturesTest -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

Result: five tests, zero failures/errors/skips; reactor BUILD SUCCESS,
36.072 seconds on the cached corrected run. The first run passed four
cases but scheduled a land play while Hydra was still on the stack.
Adding the harness's explicit stack-resolution wait corrected that test.
The original failure remains in `C:/Users/hostuser/fdn-mage-counter-001.log`.
Passing log: `fdn-mage-counter-002.log/.exit`; Surefire XML remains in the
owned `mage-fdn-counter-parity-codex/Mage.Tests/target/surefire-reports/`.

Small manifest: Java 23.0.2, Maven 3.9.9, one reactor worker, two active
processors, Maven heap four GiB/test heap three GiB, CPU only, GPU ordinal
none. Test choices are explicit and strict; no random operation is used
by these scenarios. Source SHA-256 is
`92539ba6453e76eef7d454e10afe862b78cfc23adab01de2a463f6840834fed3`;
passing XML SHA-256 is
`fe164ed54f56dd55c6359634f5633964f572b1e0ca083d995312698211c588e9`.

These are bounded rules comparisons for two cards. They do not prove all
Foundations mechanics, original-deck games, save/restore equivalence across
engines, or playing strength.
