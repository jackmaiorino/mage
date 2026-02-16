# ML Architecture Overview

## Pipeline: Game State → Prediction

```
Game State → State Encoder (Java) → Transformer (Python) → Action Selection (Java)
```

## 1. State Encoding (Java)

`StateSequenceBuilder` converts the full game state into a fixed-format sequence:

| Token Type | Content |
|---|---|
| CLS | Placeholder for aggregate representation |
| Phase | Current game phase |
| Player Stats | Life, hand size, library size, lands played (normalized) |
| Opponent Stats | Same as above |
| Cards | Hand, battlefield, graveyard, library, opponent permanents, stack |

Each token is a **128-dim float vector**. Sequence is padded to **256 tokens max**.
Each token also carries a **hashed card ID** (vocab 65536) for learned identity embeddings.

## 2. Candidate Features (Java)

For each decision, `ComputerPlayerRL` builds up to **64 candidates**, each a **32-dim feature vector**:

- **f[0]**: action type
- **f[1]**: is_pass
- **f[2-9]**: source permanent/ability properties (creature, land, tapped, power, toughness, targets, uses_stack, mana_cost)
- **f[10-17]**: target properties (is_player, is_you, life, is_permanent, creature, tapped, power, toughness)
- **f[18-23]**: game context (opponent life, own life, creature counts, hand size, untapped lands)
- **f[24-25]**: is_spell_from_hand, is_opponent_controlled

Each candidate also gets a **hashed action ID** from its ability/target name.

## 3. Main Model (Python — `MTGTransformerModel`)

```
State Sequence [B, 256, 128]
        │
        ▼
  Input Projection (128 → 512) + Token ID Embedding (65536 → 512)
        │
        ▼
  Prepend CLS Token
        │
        ▼
  6-Layer Transformer Encoder (d=512, 8 heads, FFN=2048)
        │
        ▼
  CLS Token Output [B, 512]     ◄── encodes full game context
        │
   ┌────┴─────────────────┐
   ▼                      ▼
Value Head            Candidate Scoring
critic MLP → [-10,10]    │
   scalar value           ▼
                    Candidate Projection (32 → 512) + Action ID Embedding
                          │
                          ▼
                    Concat(CLS, Candidate) → [B, N, 1024]
                          │
                          ▼
                    Per-Head MLP Scorer → scalar per candidate
                    (LayerNorm → Linear(1024,256) → ReLU → Linear(256,1))
                          │
                          ▼
                    Softmax over valid candidates → action probabilities
```

Three separate MLP scorer heads: **action**, **target**, **card_select**.

The MLP scoring (vs dot-product) allows learning non-linear interactions between game context and candidate features — e.g., "damage ability + target is self → low score."

## 4. Mulligan Model (Python — `MulliganNet`)

Separate Q-learning model for mulligan decisions.

```
Input: [mulligan_num, land_count, creature_count, avg_cmc, 7 hand IDs, 60 deck IDs]
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  Hand IDs       Deck IDs      Scalars
  embed(32)      embed(32)     (mulligan_num +
  self-attn      mean pool      3 explicit)
  mean pool      Linear(32)
  [B, 32]        [B, 32]        [B, 4]
     │              │              │
     └──────────────┴──────────────┘
                    │
                    ▼
              Concat → [B, 68]
              FC(64) → FC(32) → Output(2)
                    │
                    ▼
           [Q_keep, Q_mull] → keep if Q_keep ≥ Q_mull
```

## Key Dimensions

| Component | Shape |
|---|---|
| State sequence | `[B, ≤256, 128]` |
| Token IDs | `[B, ≤256]` |
| CLS embedding | `[B, 512]` |
| Candidate features | `[B, ≤64, 32]` |
| Candidate action IDs | `[B, ≤64]` |
| Policy output | `[B, 64]` probabilities |
| Value output | `[B, 1]` scalar |
| Mulligan input | `[B, 68]` |
| Mulligan output | `[B, 2]` Q-values |
