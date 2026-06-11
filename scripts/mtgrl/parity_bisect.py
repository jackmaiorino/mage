"""Bisect ONNX-vs-torch divergence: score_candidates vs SingleHeadScorer(torch) vs ONNX."""
import sys
import glob

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode')
from mtg_transformer import MTGTransformerModel, SingleHeadScorer  # noqa: E402
from onnx_export import _replace_mha  # noqa: E402

N = 50
BAK = 'local-training/backups/spy_value_baseline_20260531'

ckpt = torch.load(f'{BAK}/model_latest.pt', map_location='cpu', weights_only=False)
model = MTGTransformerModel(d_model=128, nhead=4, num_layers=2,
                            dim_feedforward=512, cand_feat_dim=48)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

import copy
model_for_export = copy.deepcopy(model)
wrapper = SingleHeadScorer(model_for_export, 'action')
wrapper.eval()
_replace_mha(wrapper)

active = open(f'{BAK}/onnx/.active_dir').read().strip()
sess = ort.InferenceSession(f'{BAK}/onnx/{active}/model_action.onnx',
                            providers=['CPUExecutionProvider'])

rows = []
for f in sorted(glob.glob('local-training/candq_dumps/*.npz')):
    d = np.load(f)
    if int(d['head_ids'][0]) != 0:
        continue
    rows.append({k: d[k] for k in d.files})
    if len(rows) >= N:
        break

deltas = {'sc_vs_wrap': [], 'wrap_vs_onnx': [], 'sc_vs_onnx': []}
peaks = {'sc': [], 'wrap': [], 'onnx': []}
for r in rows:
    seq_t = torch.tensor(r['seq'].astype(np.float32))
    mask_t = torch.tensor(r['mask'].astype(np.bool_))
    tok_t = torch.tensor(r['tok_ids'].astype(np.int64))
    cf_t = torch.tensor(r['cand_feat'].astype(np.float32))
    cid_t = torch.tensor(r['cand_ids'].astype(np.int64))
    cm_t = torch.tensor(r['cand_mask'].astype(np.bool_))

    with torch.inference_mode():
        p_sc = model.score_candidates(seq_t, mask_t, tok_t, cf_t, cid_t, cm_t,
                                      'action', 0, 0, 0)[0][0].numpy()
        p_wrap = wrapper(seq_t, mask_t, tok_t, cf_t, cid_t, cm_t)[0][0].numpy()

    p_onnx = sess.run(None, {
        'sequences': r['seq'].astype(np.float32), 'masks': r['mask'].astype(np.bool_),
        'token_ids': r['tok_ids'].astype(np.int64), 'cand_features': r['cand_feat'].astype(np.float32),
        'cand_ids': r['cand_ids'].astype(np.int64), 'cand_mask': r['cand_mask'].astype(np.bool_),
    })[0][0]

    valid = r['cand_mask'][0].astype(bool)
    a, b, c = p_sc[valid], p_wrap[valid], p_onnx[valid]
    deltas['sc_vs_wrap'].append(float(np.abs(a - b).max()))
    deltas['wrap_vs_onnx'].append(float(np.abs(b - c).max()))
    deltas['sc_vs_onnx'].append(float(np.abs(a - c).max()))
    peaks['sc'].append(float(a.max()))
    peaks['wrap'].append(float(b.max()))
    peaks['onnx'].append(float(c.max()))

print(f'rows={len(rows)}')
for k, v in deltas.items():
    v = np.array(v)
    print(f'max|dp| {k:14s} mean={v.mean():.4f} p90={np.percentile(v, 90):.4f} max={v.max():.4f}')
for k, v in peaks.items():
    print(f'peak prob {k:5s} mean={np.mean(v):.3f}')
