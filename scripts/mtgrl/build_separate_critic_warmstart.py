#!/usr/bin/env python3
"""Build a warm-started separate-critic model + verify WM gradient routing.

Codex-endorsed conversion experiment: isolate the value path in its own critic
encoder so world-model de-myopia shapes VALUE (not the policy encoder), and let
the improved value reach the actor only through terminal-reward PPO advantages.

Steps:
  1. Build MTGTransformerModel with VALUE_USE_SEPARATE_CRITIC_ENCODER=1,
     WORLD_MODEL_DIM=18, CRITIC_NUM_LAYERS=num_layers (critic depth == policy).
  2. Load the teacher (strict=False). Teacher's critic ENCODER weights are random
     (separate-critic was off when it trained), so warm-start the critic encoder
     by COPYING the trained policy encoder into it. Keep the teacher-trained value
     head (critic_norm/proj1/proj2/value_scale).
  3. VERIFY gradient routing: a world-model loss computed from the CRITIC CLS must
     deposit gradient on the critic encoder and NONE on the policy encoder. Same
     for the value loss. This is the invariant the whole experiment depends on.
  4. Save the warm-started checkpoint (same format as teacher) for continue-from.

Usage:
  python build_separate_critic_warmstart.py --teacher PATH --out PATH [--save]
"""
import argparse
import os
import sys

# Force the architecture flags BEFORE importing the model module.
os.environ["VALUE_USE_SEPARATE_CRITIC_ENCODER"] = "1"
os.environ["WORLD_MODEL_DIM"] = os.environ.get("WORLD_MODEL_DIM", "18")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MLDIR = os.path.join(HERE, "..", "..", "Mage.Server.Plugins", "Mage.Player.AIRL",
                     "src", "mage", "player", "ai", "rl", "MLPythonCode")
sys.path.insert(0, os.path.abspath(MLDIR))
from mtg_transformer import MTGTransformerModel  # noqa: E402


def build(cfg):
    os.environ["CRITIC_NUM_LAYERS"] = os.environ.get("CRITIC_NUM_LAYERS", str(cfg["num_layers"]))
    m = MTGTransformerModel(
        input_dim=cfg.get("input_dim", 128),
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"], cand_feat_dim=48,
    )
    return m


def warmstart_critic_from_policy(m):
    """Copy the trained policy encoder into the (random) critic encoder."""
    sd = m.state_dict()
    copied = []
    # input proj / norm / cls token
    pairs = [("input_proj", "critic_input_proj"),
             ("input_norm", "critic_input_norm")]
    with torch.no_grad():
        for src, dst in pairs:
            for suf in ("weight", "bias"):
                sk, dk = f"{src}.{suf}", f"{dst}.{suf}"
                if sk in sd and dk in sd and sd[sk].shape == sd[dk].shape:
                    sd[dk].copy_(sd[sk]); copied.append(dk)
        if "cls_token" in sd and "critic_cls_token" in sd and sd["cls_token"].shape == sd["critic_cls_token"].shape:
            sd["critic_cls_token"].copy_(sd["cls_token"]); copied.append("critic_cls_token")
        # transformer layers -> critic transformer layers (match by index)
        n_pol = len(m.transformer_layers)
        n_cri = len(m.critic_transformer)
        for i in range(min(n_pol, n_cri)):
            for k in list(sd.keys()):
                pref = f"transformer_layers.{i}."
                if k.startswith(pref):
                    dk = f"critic_transformer.{i}." + k[len(pref):]
                    if dk in sd and sd[k].shape == sd[dk].shape:
                        sd[dk].copy_(sd[k]); copied.append(dk)
    m.load_state_dict(sd)
    return copied, n_pol, n_cri


def grad_routing_test(m):
    m.train()
    B, S, D = 4, 12, m.input_dim
    seq = torch.randn(B, S, D)
    mask = torch.zeros(B, S)            # 0 = real token (Java mask: 1=pad)
    tok = torch.zeros(B, S, dtype=torch.long)

    def encoder_grad_norm(prefix):
        g = 0.0
        for n, p in m.named_parameters():
            if n.startswith(prefix) and p.grad is not None:
                g += float(p.grad.abs().sum().item())
        return g

    results = {}
    for name, fn in [("world_model", lambda c: m.world_model_logits_from_cls(c)),
                     ("value", lambda c: m._process_value_from_cls(c))]:
        m.zero_grad(set_to_none=True)
        cls = m.encode_state_critic(seq, mask, tok)
        out = fn(cls)
        out.sum().backward()
        g_critic = encoder_grad_norm("critic_transformer") + encoder_grad_norm("critic_input_proj")
        g_policy = encoder_grad_norm("transformer_layers") + encoder_grad_norm("input_proj")
        results[name] = (g_critic, g_policy)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.teacher, map_location="cpu")
    cfg = dict(ck["config"])
    print("teacher config:", cfg)
    m = build(cfg)

    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    enc_missing = [k for k in missing if k.startswith("critic_transformer") or k.startswith("critic_input")
                   or k == "critic_cls_token"]
    print(f"load strict=False: missing={len(missing)} (critic-encoder among them={len(enc_missing)}) "
          f"unexpected={len(unexpected)}")
    wm_keys = [k for k in m.state_dict() if "world_model" in k]
    print(f"world_model_head keys present: {len(wm_keys)} (expect >0 since WORLD_MODEL_DIM=18)")

    copied, n_pol, n_cri = warmstart_critic_from_policy(m)
    print(f"warm-start: critic depth={n_cri} (policy={n_pol}); copied {len(copied)} critic-encoder tensors from policy")

    res = grad_routing_test(m)
    ok = True
    for name, (gc, gp) in res.items():
        verdict = "OK" if (gc > 1e-8 and gp < 1e-10) else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(f"  [{name}] critic-encoder grad={gc:.4f}  policy-encoder grad={gp:.6f}  -> {verdict}")

    if not ok:
        print("GRAD ROUTING FAILED -- not saving. The WM/value loss must NOT touch the policy encoder.")
        sys.exit(2)
    print("GRAD ROUTING OK: WM + value de-myopia is isolated to the critic encoder; policy encoder untouched.")

    if args.save:
        out = {
            "state_dict": m.state_dict(),
            "config": {**cfg, "world_model_dim": 18},
            "train_step_counter": int(ck.get("train_step_counter", 0)),
            "main_train_sample_counter": int(ck.get("main_train_sample_counter", 0)),
            "gae_enabled_step": ck.get("gae_enabled_step", None),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(out, args.out)
        print(f"saved warm-started separate-critic model -> {args.out}")


if __name__ == "__main__":
    main()
