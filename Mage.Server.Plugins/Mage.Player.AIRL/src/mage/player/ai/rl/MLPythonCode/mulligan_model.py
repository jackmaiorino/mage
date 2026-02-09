import torch
import torch.nn.functional as F


class MulliganNet(torch.nn.Module):
    """
    Q-learning neural network for mulligan decisions.
    Uses self-attention over hand cards + explicit hand features.

    Input: [mulligan_num, land_count, creature_count, avg_cmc, hand_card_ids[7], deck_card_ids[60]]
    - mulligan_num: scalar (0-7)
    - land_count: scalar (0-7) - number of lands in hand
    - creature_count: scalar (0-7) - number of creatures in hand
    - avg_cmc: scalar (0-~8) - average mana value of non-land cards
    - hand_card_ids: 7 card IDs (token vocab)
    - deck_card_ids: 60 card IDs (token vocab)

    Output: [Q_keep, Q_mull] - Q-values for each action
    Decision: Choose KEEP if Q_keep >= Q_mull, else MULLIGAN

    Architecture:
    1. Embed each card ID
    2. Self-attention over hand embeddings (learns card interactions)
    3. Mean-pool deck embeddings (deck signature)
    4. Combine with mulligan number and explicit features
    5. MLP to output two Q-values
    """

    def __init__(self, vocab_size=65536, embed_dim=32, max_hand=7, max_deck=60, num_explicit=3):
        super(MulliganNet, self).__init__()

        self.max_hand = max_hand
        self.max_deck = max_deck
        self.num_explicit = num_explicit

        # Card embedding (shared for hand and deck)
        self.card_embed = torch.nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)

        # Self-attention for hand cards (learns card interactions)
        self.hand_attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True)
        self.hand_norm = torch.nn.LayerNorm(embed_dim)

        # Deck processing (mean pool is fine for deck signature)
        self.deck_fc = torch.nn.Linear(embed_dim, 32)

        # Combine: [mulligan_num(1) + explicit(num_explicit) + hand_attn_pool(embed_dim) + deck_pool(32)]
        combined_dim = 1 + num_explicit + embed_dim + 32
        self.fc1 = torch.nn.Linear(combined_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        # Q-values: output 2 values (Q_keep, Q_mull)
        self.output = torch.nn.Linear(32, 2)

    def forward(self, x):
        # x shape: [batch_size, 1 + num_explicit + max_hand + max_deck]
        offset = 1 + self.num_explicit

        # Extract components
        mulligan_num = x[:, 0:1]  # [batch, 1]
        explicit_feats = x[:, 1:offset]  # [batch, num_explicit]
        hand_ids = x[:, offset:offset + self.max_hand].long()  # [batch, 7]
        deck_ids = x[:, offset + self.max_hand:offset +
                     self.max_hand + self.max_deck].long()  # [batch, 60]

        # Embed hand cards
        hand_embeds = self.card_embed(hand_ids)  # [batch, 7, embed_dim]
        hand_pad_mask = (hand_ids == 0)  # [batch, 7], True = padding

        # Self-attention over hand (learns card interactions and importance)
        attn_out, _ = self.hand_attn(
            hand_embeds, hand_embeds, hand_embeds,
            key_padding_mask=hand_pad_mask)
        attn_out = self.hand_norm(attn_out + hand_embeds)  # residual

        # Pool attended hand: mean over non-padded positions
        pool_mask = (~hand_pad_mask).float().unsqueeze(-1)  # [batch, 7, 1]
        hand_pool = (attn_out * pool_mask).sum(dim=1) / \
            (pool_mask.sum(dim=1) + 1e-8)  # [batch, embed_dim]

        # Embed and pool deck cards (mean pool for deck signature)
        deck_embeds = self.card_embed(deck_ids)  # [batch, 60, embed_dim]
        deck_mask = (deck_ids != 0).float().unsqueeze(-1)  # [batch, 60, 1]
        deck_pool = (deck_embeds * deck_mask).sum(dim=1) / \
            (deck_mask.sum(dim=1) + 1e-8)  # [batch, embed_dim]
        deck_feat = F.relu(self.deck_fc(deck_pool))  # [batch, 32]

        # Combine all features
        combined = torch.cat(
            [mulligan_num, explicit_feats, hand_pool, deck_feat], dim=1)

        # MLP
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # Output Q-values: [batch, 2] where [:, 0] = Q_keep, [:, 1] = Q_mull
        return self.output(x)
