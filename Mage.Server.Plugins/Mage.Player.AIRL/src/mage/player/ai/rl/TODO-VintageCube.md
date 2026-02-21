# Vintage Cube - Post-Initial-Implementation TODO

## Training Improvements

- [ ] Self-play drafting: all 8 drafters are RL agents (population-based or snapshot opponents)
- [ ] Card embedding transfer: initialize draft model's token_id_emb from game model's trained embeddings
- [ ] RL-controlled land count (16/17/18) via auxiliary model output
- [ ] RL-controlled non-basic land selection from the drafted pool (duals, fetches, utility lands)
- [ ] Sideboarding between games in a match based on opponent's deck
- [ ] Hate-drafting: model learns to deny powerful cards from opponents even when off-color
- [ ] Snapshot opponents for game self-play (prevent co-training collapse)

## Activation Failure Fixes (Vintage Cards)

- [ ] Planeswalker loyalty ability activation
- [ ] Alternate costs (Force of Will exile-a-blue, Snuff Out pay life, Fireblast sacrifice lands)
- [ ] Complex modal spells (Cryptic Command, Mystic Confluence, Kolaghan's Command)
- [ ] Combo-enabling cards (Doomsday, Yawgmoth's Will, Recurring Nightmare, Sneak Attack)
- [ ] X-cost spells (Walking Ballista, Everflowing Chalice)
- [ ] Transform/MDFC cards (Fable of the Mirror-Breaker, Jace Vryn's Prodigy)
- [ ] Sacrifice-as-cost abilities (Lion's Eye Diamond, Zuran Orb)
- [ ] Flashback/retrace/escape costs
- [ ] Cascade/discover triggers

## Evaluation & Monitoring

- [ ] Draft evaluation league (periodic benchmark drafts with greedy mode)
- [ ] Track archetype emergence (is the model learning to draft Storm, Reanimator, Aggro, etc?)
- [ ] Per-archetype winrate tracking
- [ ] Compare draft model picks against known human cube pick orders
- [ ] Monitor co-training stability (detect if one model stagnates while the other improves)

## Model Architecture Experiments

- [ ] Tune draft model size (d_model, num_layers) based on initial training results
- [ ] Experiment with separate value heads for draft picks vs construction decisions
- [ ] Consider attention masking between pool and seen tokens (pool-to-seen but not seen-to-seen)
- [ ] Evaluate whether 315 seen tokens is too large; consider capping or summarizing old seen cards
