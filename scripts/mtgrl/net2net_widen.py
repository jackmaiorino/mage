"""Net2Net function-preserving widen: 7k 128d/2L -> 256d/2L (Codex #48).
Channel DUPLICATION (256=[x,x]) preserves LayerNorm; nhead 4->8 keeps head_dim=32 so MHA is
preserved; dim_feedforward stays 512; num_layers stays 2. The model couples MLP-head hidden dims
to d_model (hidden = d_model//2: 64->128) so those heads also get Net2Net hidden-widening.
Validate separately: the 256/2 must reproduce the 7k policy on the same inputs.
"""
import sys, os, torch
SRC = sys.argv[1] if len(sys.argv) > 1 else r'E:/mage-training/backups/affinity_ar_league_7k_snap/model.pt'
DST = sys.argv[2] if len(sys.argv) > 2 else r'E:/mage-training/backups/affinity_256_transplant/model.pt'
D = 128

def dup_vec(v):  return torch.cat([v, v], dim=0)                       # [n]->[2n]
def dup_rows(W): return torch.cat([W, W], dim=0)                       # [o,i]->[2o,i]
def halfdup_cols(W): return torch.cat([W/2, W/2], dim=1)              # [o,i]->[o,2i] (dup-halve, preserves W@[x,x])
def widen_dd(W):                                                       # d_model->d_model [128,128]->[256,256]
    return dup_rows(halfdup_cols(W))
def scorer_cols(W):                                                    # over concat([cls128,att128]) cols -> [c,c,a,a]
    Wc, Wa = W[:, :D], W[:, D:]
    return torch.cat([Wc/2, Wc/2, Wa/2, Wa/2], dim=1)
def scorer_ln(v):
    c, a = v[:D], v[D:]
    return torch.cat([c, c, a, a], dim=0)

# explicit role sets
OUTPUT_WIDEN_PROJ = ('input_proj.weight', 'critic_input_proj.weight')  # input_dim(128) IN, d_model OUT
SCORERS = ('policy_scorer', 'candidate_q_scorer')
MLP_HEADS_INTO_HID = ('actor_proj1', 'critic_proj1', 'belief_head.1')   # d_model IN, hidden(64->128) OUT
MLP_HEADS_OUT_HID  = ('actor_proj2', 'critic_proj2', 'belief_head.3')   # hidden(64->128) IN, small OUT
HID_VECS = ('actor_norm1', 'critic_norm1')                              # hidden(64->128) norm/bias

d = torch.load(SRC, map_location='cpu')
sd = d['state_dict']; out = {}
for k, v in sd.items():
    if not torch.is_tensor(v) or v.dim() == 0 or v.numel() == 1:
        out[k] = v.clone(); continue
    leaf = k.split('.')[-1]
    # --- scorers: input=concat(cls,att)=256, hidden 64->128 ---
    if any(s in k for s in SCORERS):
        if k.endswith('.0.weight') or k.endswith('.0.bias'):  out[k] = scorer_ln(v)            # LN over concat
        elif k.endswith('.1.weight'):                         out[k] = dup_rows(scorer_cols(v)) # in concat-widen + out hidden-dup
        elif k.endswith('.1.bias'):                           out[k] = dup_vec(v)               # hidden bias
        elif k.endswith('.3.weight'):                         out[k] = halfdup_cols(v)          # out-of-hidden: in halve+dup
        else:                                                 out[k] = v.clone()                # .3.bias [1]
        continue
    # --- embeddings ---
    if 'token_id_emb' in k or 'action_id_emb' in k:
        out[k] = torch.cat([v, v], dim=1); continue
    # --- input_dim->d_model projections (input stays 128, output widens) ---
    if k in OUTPUT_WIDEN_PROJ:
        out[k] = dup_rows(v); continue
    # --- attention in_proj (stacked Q,K,V each d_model->d_model) ---
    if leaf == 'in_proj_weight':
        q, kk, vv = v[:D], v[D:2*D], v[2*D:]
        out[k] = torch.cat([widen_dd(q), widen_dd(kk), widen_dd(vv)], dim=0); continue
    if leaf == 'in_proj_bias':
        q, kk, vv = v[:D], v[D:2*D], v[2*D:]
        out[k] = torch.cat([dup_vec(q), dup_vec(kk), dup_vec(vv)], dim=0); continue
    # --- FFN (hidden dim_feedforward=512 unchanged) ---
    if 'linear1' in k:
        out[k] = halfdup_cols(v) if leaf == 'weight' else v.clone(); continue   # in d_model-widen, out=512 fixed
    if 'linear2' in k:
        out[k] = dup_rows(v) if leaf == 'weight' else dup_vec(v); continue       # in=512 fixed, out d_model-widen
    # --- cand_feat_proj.0 (48 IN, d_model OUT) ---
    if 'cand_feat_proj.0' in k:
        out[k] = dup_rows(v) if leaf == 'weight' else dup_vec(v); continue
    # --- MLP head hidden (d_model//2: 64->128) ---
    if any(h in k for h in MLP_HEADS_INTO_HID):                                   # d_model IN, hidden OUT
        out[k] = dup_rows(halfdup_cols(v)) if leaf == 'weight' else dup_vec(v); continue
    if any(h in k for h in MLP_HEADS_OUT_HID):                                    # hidden IN, small OUT
        out[k] = halfdup_cols(v) if leaf == 'weight' else v.clone(); continue
    if any(h in k for h in HID_VECS):                                             # hidden norm/bias
        out[k] = dup_vec(v); continue
    # --- generic d_model -> d_model Linear weights ---
    if leaf == 'weight' and v.dim() == 2:
        oN, iN = v.shape
        if oN == D and iN == D:   out[k] = widen_dd(v)
        elif oN == D:             out[k] = dup_rows(v)       # ?->d_model
        elif iN == D:             out[k] = halfdup_cols(v)   # d_model->small
        else:                     out[k] = v.clone()
        continue
    # --- 1-D: LN/bias over d_model (128) duplicate, else copy ---
    if v.dim() == 1:
        out[k] = dup_vec(v) if v.shape[0] == D else v.clone(); continue
    if v.dim() == 3 and v.shape[-1] == D:    # cls_token [1,1,128]
        out[k] = torch.cat([v, v], dim=-1); continue
    out[k] = v.clone(); print(f"[WARN] copied unhandled: {k} {tuple(v.shape)}", flush=True)

cfg = dict(d.get('config', {}))
cfg.update(d_model=256, nhead=8, num_layers=2, dim_feedforward=512)
newd = dict(d); newd['state_dict'] = out; newd['config'] = cfg
newd.pop('optimizer_state_dict', None)
os.makedirs(os.path.dirname(DST), exist_ok=True); torch.save(newd, DST)
print(f"widened -> {DST}  tensors={len(out)}")
