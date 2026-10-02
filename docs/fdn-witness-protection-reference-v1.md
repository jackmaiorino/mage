# Witness Protection reference comparisons

Sixteen strict XMage reference cases in `FdnWitnessProtectionTest` cover the
kernel batch's derived characteristics, card-type removal, preserved Legendary,
older/later Armor and Flying grants, retained counters, removed lord/static
abilities, Clockwork Percussionist death LKI with restoration control, Aura/host removal,
removed mana/draw abilities and removed ward.

Source authority: `Mage.Sets/src/mage/cards/w/WitnessProtection.java`.
The kernel counterpart is `mtg-kernel/tests/fdn_witness_protection_v1.rs` in
PR138. This changes tests, focused CI and documentation only.

All16 Witness cases and all7 existing `LondonMulliganTest` cases passed, with
zero failures, errors or skips. Hosted run37044173670 built the reference from
source on Ubuntu24.04 using Temurin23.0.2+7, Maven3.9.9, processors2 and T1.
Maven exited zero after04:01minutes. Both local PCs kept their reservations.

The public source is `d9536815d5826ac2a48446d2e1c21f6aba77f0ce`; the compiled
PR merge commit is `f337692e257c53c433ee6723e3dc6d7c5c2b1055`.
Downloaded input hashes match the public source's exact Git blobs. XML and
output hashes match the uploaded metadata. Witness test-source SHA-256:
`5ada47841b03880a7f7295fc54923a65f2f6f0013830fe9418818a03e1468eb3`.

The command was:

```text
mvn -B -ntp -T 1 -pl Mage.Tests -am \
  -Dtest=FdnWitnessProtectionTest,LondonMulliganTest \
  -Dsurefire.failIfNoSpecifiedTests=false -DfailIfNoTests=false \
  -Dxmage.dataCollectors.printGameLogs=false \
  "-DargLine=-Xmx3g -XX:ActiveProcessorCount=2 -Dfile.encoding=UTF-8" test
```

Sealed evidence: `E:/mtg-fdn-fixtures/fdn-mage-witness-hosted-001`.
Independent verified mirror: `C:/Users/Jack/fdn-mage-witness-hosted-001-sealed`.
`seal.json` and `closure.json` identify each retained manifest, log and XML.
The committed `docs/reports/fdn_mage_witness_hosted_001_prune.json` records
removal of duplicate scratch after both retained copies were verified.

Witness XML SHA-256:
`5d56400d690b60b5886ce8242d6c701d56350c457d38f54ed10392c2b5a57d89`.
London XML SHA-256:
`e938083f1730a3e7a85c5d4aae4c9be0a27a4cfa59d4db92af35c113092748e7`.
Maven log SHA-256:
`d0bd74e81d9f55159f2534e242dc85e3882fb464d49c96fdd7ce9febf512666d`.

The690,678,376-byte hosted build/dependency footprint remained below the8GiB
cap with90,056,036,352bytes free, above the60GiB reserve. Local sealed evidence
is275,753bytes before the seal index. These are focused rules comparisons,
with no full-set or playing-strength claim.
