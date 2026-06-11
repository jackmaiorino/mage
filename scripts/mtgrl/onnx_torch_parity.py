"""ONNX-vs-torch inference parity on real states from candq dumps.

The hybrid training loop samples actions (and records behavior log-probs) from
the ONNX export while the learner computes gradient log-probs from the torch
checkpoint. Any systematic divergence biases every PPO importance ratio.

Usage: py -3.12 scripts/mtgrl/onnx_torch_parity.py [n_rows]
"""
import sys
import glob
import os

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
BAK = 'local-training/backups/spy_value_baseline_20260531'

ckpt = torch.load(f'{BAK}/model_latest.pt', map_location='cpu', weights_only=False)
model = MTGTransformerModel(d_model=128, nhead=4, num_layers=2,
                            dim_feedforward=512, cand_feat_dim=48)
missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print(f'torch load: missing={len(missing)} unexpected={len(unexpected)}')
model.eval()

active = open(f'{BAK}/onnx/.active_dir').read().strip()
onnx_path = f'{BAK}/onnx/{active}/model_action.onnx'
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
print('onnx:', onnx_path)
print('onnx inputs:', [(i.name, i.type, i.shape) for i in sess.get_inputs()])
print('onnx outputs:', [o.name for o in sess.get_outputs()])

files = sorted(glob.glob('local-training/candq_dumps/*.npz'))
rows = []
for f in files:
    d = np.load(f)
    if int(d['head_ids'][0]) != 0:
        continue
    rows.append({k: d[k] for k in d.files})
    if len(rows) >= N:
        break
print(f'action-head rows: {len(rows)}')


def onnx_dtype(name):
    for i in sess.get_inputs():
        if i.name == name:
            return i.type
    return None


stats = []
for r in rows:
    seq = r['seq'].astype(np.float32)
    mask_b = r['mask'].astype(np.bool_)
    tok = r['tok_ids'].astype(np.int64 if 'int64' in str(onnx_dtype('token_ids')) else np.int32)
    cf = r['cand_feat'].astype(np.float32)
    cid = r['cand_ids'].astype(np.int64 if 'int64' in str(onnx_dtype('cand_ids')) else np.int32)
    cm_b = r['cand_mask'].astype(np.bool_)

    out = sess.run(None, {
        'sequences': seq, 'masks': mask_b, 'token_ids': tok,
        'cand_features': cf, 'cand_ids': cid, 'cand_mask': cm_b,
    })
    p_onnx = out[0][0]

    with torch.inference_mode():
        res = model.score_candidates(
            torch.tensor(seq), torch.tensor(mask_b), torch.tensor(r['tok_ids'].astype(np.int64)),
            torch.tensor(cf), torch.tensor(r['cand_ids'].astype(np.int64)),
            torch.tensor(cm_b), 'action', 0, 0, 0)
        p_torch = res[0][0].numpy() if isinstance(res, tuple) else res[0].numpy()

    valid = cm_b[0]
    po = np.clip(p_onnx[valid], 1e-9, 1.0)
    pt = np.clip(p_torch[valid], 1e-9, 1.0)
    po = po / po.sum()
    pt = pt / pt.sum()
    kl = float(np.sum(pt * np.log(pt / po)))
    amax_t = int(np.argmax(pt))
    amax_o = int(np.argmax(po))
    # the PPO ratio bias if the ONNX-sampled (argmax) action is trained with torch logp
    ratio = float(pt[amax_o] / po[amax_o])
    stats.append((float(np.abs(pt - po).max()), kl, amax_t == amax_o, ratio,
                  float(pt.max()), float(po.max())))

a = np.array([s[:2] + (1.0 if s[2] else 0.0,) + s[3:] for s in stats], dtype=np.float64)
print(f"\nrows={len(a)}")
print(f"max|dp|      mean={a[:,0].mean():.4f}  p50={np.percentile(a[:,0],50):.4f}  p90={np.percentile(a[:,0],90):.4f}  max={a[:,0].max():.4f}")
print(f"KL(t||o)     mean={a[:,1].mean():.5f}  p90={np.percentile(a[:,1],90):.5f}  max={a[:,1].max():.5f}")
print(f"argmax agree {100*a[:,2].mean():.1f}%")
print(f"IS-ratio at onnx-argmax: mean={a[:,3].mean():.3f}  p10={np.percentile(a[:,3],10):.3f}  p90={np.percentile(a[:,3],90):.3f}  min={a[:,3].min():.3f}  max={a[:,3].max():.3f}")
print(f"peak prob: torch mean={a[:,4].mean():.3f}  onnx mean={a[:,5].mean():.3f}")
