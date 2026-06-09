"""Diff current model weights vs the frozen baseline, grouped by submodule.

Confirms (without the unlogged candidate_q loss) that the Q-head is learning
and that CANDIDATE_Q_ONLY froze the policy/encoder (safety).
"""
import sys
import torch

MD = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Spy-Combo-Value/models"
BAK = "local-training/backups/spy_value_baseline_20260531"


def load(p):
    c = torch.load(p, map_location="cpu")
    return c["state_dict"] if isinstance(c, dict) and "state_dict" in c else c


def main():
    live = load(MD + "/model_latest.pt")
    base = load(BAK + "/model_latest.pt")
    groups = ["candidate_q_scorer", "policy_scorer", "transformer_layers",
              "critic_proj", "critic_transformer", "value_scale", "token_id_emb",
              "belief_head", "cls_token", "input_proj"]
    acc = {g: [0.0, 0] for g in groups}
    other = [0.0, 0]
    for k in base:
        if k not in live:
            continue
        d = (live[k].float() - base[k].float()).abs().mean().item()
        for g in groups:
            if g in k:
                acc[g][0] += d
                acc[g][1] += 1
                break
        else:
            other[0] += d
            other[1] += 1
    print(f'{"param group":<22}{"#tensors":>9}{"mean|delta|":>16}')
    for g in groups:
        s, n = acc[g]
        print(f'{g:<22}{n:>9}{(s/n if n else 0):>16.6e}')
    print(f'{"(other)":<22}{other[1]:>9}{(other[0]/other[1] if other[1] else 0):>16.6e}')
    print("\n=> candidate_q_scorer delta>0 = Q-head IS learning; "
          "policy_scorer/transformer ~0 = frozen (safe).")


if __name__ == "__main__":
    main()
