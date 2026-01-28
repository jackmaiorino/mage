import torch
import torch.nn.functional as F


class MulliganNet(torch.nn.Module):
    """
    Q-learning neural network for mulligan decisions.
    No heuristics - learns entirely from game outcomes.

    Input: [mulligan_num, hand_card_ids[7], deck_card_ids[60]]
    - mulligan_num: scalar (0-7)
    - hand_card_ids: 7 card IDs (token vocab)
    - deck_card_ids: 60 card IDs (token vocab)

    Output: [Q_keep, Q_mull] - Q-values for each action
    Decision: Choose KEEP if Q_keep >= Q_mull, else MULLIGAN

    Architecture:
    1. Embed each card ID
    2. Pool hand and deck embeddings separately
    3. Combine with mulligan number
    4. MLP to output two Q-values
    """

    def __init__(self, vocab_size=65536, embed_dim=32, max_hand=7, max_deck=60):
        super(MulliganNet, self).__init__()

        self.max_hand = max_hand
        self.max_deck = max_deck

        # Card embedding (shared for hand and deck)
        self.card_embed = torch.nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)

        # Separate processing for hand and deck
        self.hand_fc = torch.nn.Linear(embed_dim, 32)
        self.deck_fc = torch.nn.Linear(embed_dim, 32)

        # Combine everything: [mulligan_num(1), hand_pool(32), deck_pool(32)] = 65
        self.fc1 = torch.nn.Linear(1 + 32 + 32, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        # Q-values: output 2 values (Q_keep, Q_mull)
        self.output = torch.nn.Linear(32, 2)

    def forward(self, x):
        # x shape: [batch_size, 68] = [mulligan_num(1), hand_ids(7), deck_ids(60)]

        # Extract components
        mulligan_num = x[:, 0:1]  # [batch, 1]
        hand_ids = x[:, 1:1+self.max_hand].long()  # [batch, 7]
        deck_ids = x[:, 1+self.max_hand:1+self.max_hand +
                     self.max_deck].long()  # [batch, 60]

        # Embed cards
        hand_embeds = self.card_embed(hand_ids)  # [batch, 7, embed_dim]
        deck_embeds = self.card_embed(deck_ids)  # [batch, 60, embed_dim]

        # Pool: mean over cards (ignoring padding=0)
        hand_mask = (hand_ids != 0).float().unsqueeze(-1)  # [batch, 7, 1]
        deck_mask = (deck_ids != 0).float().unsqueeze(-1)  # [batch, 60, 1]

        hand_pool = (hand_embeds * hand_mask).sum(dim=1) / \
            (hand_mask.sum(dim=1) + 1e-8)  # [batch, embed_dim]
        deck_pool = (deck_embeds * deck_mask).sum(dim=1) / \
            (deck_mask.sum(dim=1) + 1e-8)  # [batch, embed_dim]

        # Process hand and deck
        hand_feat = F.relu(self.hand_fc(hand_pool))  # [batch, 32]
        deck_feat = F.relu(self.deck_fc(deck_pool))  # [batch, 32]

        # Combine all features
        combined = torch.cat(
            [mulligan_num, hand_feat, deck_feat], dim=1)  # [batch, 65]

        # MLP
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # Output Q-values: [batch, 2] where [:, 0] = Q_keep, [:, 1] = Q_mull
        return self.output(x)
