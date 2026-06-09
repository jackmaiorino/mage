"""MC-dropout epistemic-uncertainty probe on the dominated Spy-cast niche.

Loads the baseline model, replays captured Spy-cast model-inputs with DROPOUT ON
(model.train(); transformer LayerNorm is train/eval-invariant so only dropout
changes), runs N stochastic forward passes, and measures the prediction VARIANCE
(value + cast-Spy prob) per state. Question: is the model's own prediction more
UNCERTAIN (higher dropout variance) on the dominated (creatures<2) casts than on
the finishable (creatures>=2) ones? High variance on the niche => a thesis-clean
learnable "search now" trigger exists. Low/equal variance => confidently blind.

Usage: python mc_dropout_probe.py <capture_file> [N]
Capture line: cre|spy_idx|seq_len|dim|max_cand|cfd|b64(seq)|b64(mask)|b64(tok)|b64(candFeat)|b64(candIds)|b64(candMask)
"""
import base64
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MLDIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "Mage.Server.Plugins", "Mage.Player.AIRL",
    "src", "mage", "player", "ai", "rl", "MLPythonCode"))
MODEL = os.path.normpath(os.path.join(
    HERE, "..", "..", "Mage.Server.Plugins", "Mage.Player.AIRL", "src", "mage",
    "player", "ai", "rl", "profiles", "Pauper-Spy-Combo-Value", "models", "model_latest.pt"))
sys.path.insert(0, MLDIR)
os.environ.setdefault("MODEL_D_MODEL", "128")
os.environ.setdefault("MODEL_NUM_LAYERS", "2")
os.environ.setdefault("MODEL_NHEAD", "4")
os.environ.setdefault("MODEL_DIM_FEEDFORWARD", "512")

from mtg_transformer import MTGTransformerModel  # noqa: E402

INIT_KEYS = ["input_dim", "d_model", "nhead", "num_layers", "dim_feedforward",
             "dropout", "num_actions", "token_vocab", "action_vocab", "cand_feat_dim"]


def load_model():
    ck = torch.load(MODEL, map_location="cpu")
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    kw = {k: cfg[k] for k in INIT_KEYS if k in cfg}
    kw.setdefault("d_model", 128); kw.setdefault("nhead", 4)
    kw.setdefault("num_layers", 2); kw.setdefault("dim_feedforward", 512)
    kw.setdefault("cand_feat_dim", 48)
    m = MTGTransformerModel(**kw)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    res = m.load_state_dict(sd, strict=False)
    print(f"loaded model ({sum(p.numel() for p in m.parameters())/1e6:.1f}M); "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}; dropout={getattr(m,'dropout',cfg.get('dropout','?'))}")
    return m


def main():
    cap = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    m = load_model()
    rows = []
    for line in open(cap, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("|")
        if len(parts) < 12:
            continue
        cre, spy_idx, seq_len, dim, max_cand, cfd = (int(x) for x in parts[:6])
        b = [base64.b64decode(p) for p in parts[6:12]]
        seq = np.frombuffer(b[0], dtype="<f4").reshape(1, seq_len, dim).copy()
        mask = np.frombuffer(b[1], dtype="<i4").reshape(1, seq_len).astype(np.bool_)
        tok = np.frombuffer(b[2], dtype="<i4").reshape(1, seq_len).astype(np.int64)
        cf = np.frombuffer(b[3], dtype="<f4").reshape(1, max_cand, cfd).copy()
        cid = np.frombuffer(b[4], dtype="<i4").reshape(1, max_cand).astype(np.int64)
        cm = np.frombuffer(b[5], dtype="<i4").reshape(1, max_cand).astype(np.bool_)
        seqt = torch.from_numpy(seq); maskt = torch.from_numpy(mask)
        tokt = torch.from_numpy(tok); cft = torch.from_numpy(cf)
        cidt = torch.from_numpy(cid); cmt = torch.from_numpy(cm)
        # deterministic (dropout off) baseline
        m.eval()
        with torch.no_grad():
            p0, v0 = m.score_candidates(seqt, maskt, tokt, cft, cidt, cmt, head_id="action")[:2]
        v_det = float(v0.reshape(-1)[0]); sp_det = float(p0.reshape(p0.shape[0], -1)[0, spy_idx]) if spy_idx >= 0 else float("nan")
        # MC-dropout: N stochastic passes
        m.train()
        vals, sps = [], []
        with torch.no_grad():
            for _ in range(N):
                p, v = m.score_candidates(seqt, maskt, tokt, cft, cidt, cmt, head_id="action")[:2]
                vals.append(float(v.reshape(-1)[0]))
                if spy_idx >= 0:
                    sps.append(float(p.reshape(p.shape[0], -1)[0, spy_idx]))
        rows.append((cre, v_det, float(np.std(vals)), sp_det,
                     float(np.std(sps)) if sps else float("nan")))
    print(f"\ncaptured states: {len(rows)}  (N={N} dropout passes each)")
    print(f"{'creatures':>9} {'value_det':>9} {'value_STD':>9} {'castp_det':>9} {'castp_STD':>9}")
    for r in sorted(rows):
        print(f"{r[0]:>9} {r[1]:>9.3f} {r[2]:>9.4f} {r[3]:>9.3f} {r[4]:>9.4f}")
    dom = [r for r in rows if r[0] < 2]
    good = [r for r in rows if r[0] >= 2]
    def mean(xs, i):
        xs = [x[i] for x in xs if not np.isnan(x[i])]
        return float(np.mean(xs)) if xs else float("nan")
    print(f"\nDOMINATED (creatures<2, n={len(dom)}): value_STD mean={mean(dom,2):.4f} | castp_STD mean={mean(dom,4):.4f}")
    print(f"FINISHABLE (creatures>=2, n={len(good)}): value_STD mean={mean(good,2):.4f} | castp_STD mean={mean(good,4):.4f}")
    print("=> if DOMINATED value_STD/castp_STD >> FINISHABLE -> epistemic signal on the niche (learnable trigger). If similar -> confidently blind.")


if __name__ == "__main__":
    main()
