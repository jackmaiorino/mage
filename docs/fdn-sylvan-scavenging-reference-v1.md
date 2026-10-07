# Sylvan Scavenging reference cases

All twelve strict XMage cases pass. Modes are chosen at trigger placement,
before targets. The token mode remains selectable on an empty battlefield,
and checks the controller's current creature power only on resolution.

Coverage: counter placement, exact 3/3 token at four power, three-power and
empty-board no-ops, opponent-only large creatures, controller-only end steps,
power gained/lost in response, source removal, target loss, token entry
triggers and exact three-mana casting cost.

Functional source: `b4cb878d691b8421082780b0b79afbbe8535c7eb`.
Reference sources are `Mage.Sets/src/mage/cards/s/SylvanScavenging.java`
and `Mage/src/main/java/mage/game/permanent/token/RaccoonToken.java`.
Neither main source changed. The newly compiled test uses the previously
verified main classes.

Guarded run prefix: `C:/Users/hostuser/fdn-mage-scavenging-003`.
Java 23.0.2, Maven 3.9.9, one Maven thread, two JVM processors,
4 GiB Maven heap and 3 GiB test heap. No GPU or training.
Projected allocation 1 GiB, cap 2 GiB, reserve 60 GiB.

```text
mvn -B -T 1 -pl Mage.Tests -am -Dmaven.main.skip=true \
  -Dtest=FdnSylvanScavengingTest -Dsurefire.failIfNoSpecifiedTests=false \
  -DfailIfNoTests=false -Dxmage.dataCollectors.printGameLogs=false \
  "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

Observed exit zero: 12 tests, zero failures, errors or skips.
SHA-256s:

- Test source: `5241da54dd5e584e61b27cfc443ddb0648c331cd24754cd3d700b71c0a99b581`
- XML: `89bc43bdba798b95a90f64139e011cb6d09144f7b8993ad128134aaca7fef5d1`
- Output log: `bdef21629d9a64a342c6e00e6e4c6fb43085acefefb7f2ee7cab008857355cf8`

Runs 001/002 are retained. Initial strict setup omitted Snap's optional
untap prompt and the Unicorn/Angel trigger-order choice. The target-loss case
now uses Unsummon. The order choice uses the rule prefix because
`TestPlayer.chooseTriggeredAbility` matches with `startsWith`.

These are bounded rules comparisons for issue #110, with no full-set or
playing-strength claim.
