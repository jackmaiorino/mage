# Thesis-Clean Generic Prefix Policy-Miss Distillation

Date: 2026-05-12

## Question

Can generic prefix-search distillation work if it trains only on states where the current policy actually misses the winning target set?

The prior generic prefix-search family found terminal-winning lines, but direct distillation over-steered because the accepted policy already matched most individual tensors. This branch adds a generic hard-example filter:

```text
--policy-miss-only=true
```

The filter scores each serialized training tensor with the current model and keeps only examples where the policy top-1 action is outside the positive search target set. It does not inspect card names or Spy-specific game facts.

## Code Change

`ActionCounterfactualTrainer` now supports:

```text
--policy-miss-only=true
```

Wrappers:

```powershell
scripts/run_spy_line_search.ps1 -PolicyMissOnly
scripts/run_action_counterfactual.ps1 -PolicyMissOnly
```

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
PowerShell parser checks for both wrappers
```

## Source Data Probe

Source corpus:

```text
local-training/local_pbt/generic_prefix_data/20260511_balanced_cp1_cp7_hardmatchups
```

Accepted-checkpoint score probe:

```text
20260512_thesis_clean_prefix_policy_miss_score_accepted
```

Filtered examples:

```text
8/124 policy misses
top1=0/8
target_set_top1=0/8
avg_target_prob=0.277048
avg_target_rank=2.5000
```

## Clone

Fresh clone:

```text
Pauper-Spy-Combo-Value-GenericPrefixMiss-20260512
```

Source:

```text
accepted Pauper-Spy-Combo-Value checkpoint
```

The branch remains thesis-clean:

```text
RL_HEURISTIC_STEP_REWARDS=0
RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE=0
no Spy terminal mode
no action-text regex
no Spy hand pool
no selective MCTS
```

## Training

Run:

```text
20260512_thesis_clean_generic_prefix_policy_miss_train8_e8p4
```

Settings:

```text
import_training_data_path=local-training/local_pbt/generic_prefix_data/20260511_balanced_cp1_cp7_hardmatchups
policy_miss_only=true
train_epochs=8
candidate_permutations=4
MCTS_KL_LOSS_COEF=1.0
MCTS_TARGET_POLICY_MIX=1.0
LOAD_OPTIMIZER_STATE=0
RESET_TRAINING_STATE_ON_LOAD=1
```

Result:

```text
training_examples=10
actual_train_pass_samples=228
reported_train_pass_samples=320
train_steps=9
```

Clone hashes after training:

```text
model=ABAD97AEE1CB
model_latest=ABAD97AEE1CB
```

Accepted active profile hashes stayed unchanged:

```text
Pauper-Spy-Combo-Value       model=14E1A1DA4F3E latest=72857AA2975A
Pauper-Wildfire-Value        model=217707FE2EF0 latest=6F72507293CC
Pauper-Rally-Anchor-Value    model=C4ADFD672072 latest=5A04AFF24179
Pauper-Affinity-Anchor-Value model=A692C64954F3 latest=B7FD1930779E
```

## Reduced CP7 Screen

Run:

```text
20260512_generic_prefix_policy_miss_train8_spy_cp7_unique_eval16
22/64 = 34.38%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 11 | 16 | 68.75% |
| Jund Wildfire | 6 | 16 | 37.50% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 2 | 16 | 12.50% |

## Verdict

Rejected. Do not run CP1 or HPC.

The hard-example filter confirms the direct prefix corpus is mostly already solved at the single-state level. Training only the missed tensors still damages hard-pressure play, so the remaining failure is not simply "too many solved prefix labels." Prefix search needs to enter the loop as online policy improvement or trajectory-level credit, not as another small offline KL patch.
