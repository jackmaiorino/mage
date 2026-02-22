"""
Draft Model Py4J Entry Point

Serves the draft model (pick_head + construction_head + value_head) over Py4J.
Starts a separate process from the game model on DRAFT_PY4J_PORT (default 25335).

Model hyperparameters (smaller than game model):
  input_dim=32, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024
  cand_feat_dim=32, token_vocab=65536
"""

import argparse
import os
import sys
import time
import struct
import threading
import logging
import traceback
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from py4j.clientserver import ClientServer, JavaParameters, PythonParameters

# Add script directory to path so mtg_transformer can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from mtg_transformer import MTGTransformerModel

logging.basicConfig(
    level=getattr(logging, os.getenv("MTG_AI_LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="%(asctime)s %(levelname)s [draft_model] %(message)s",
)
logger = logging.getLogger("draft_model")
# Suppress Py4J protocol chatter like "Received command c on object id t"
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DRAFT_INPUT_DIM   = 32
DRAFT_D_MODEL     = 256
DRAFT_NHEAD       = 4
DRAFT_NUM_LAYERS  = 4
DRAFT_DIM_FF      = 1024
DRAFT_CAND_DIM    = 32
DRAFT_TOKEN_VOCAB = 65536
DRAFT_MAX_LEN     = 420

PPO_EPSILON       = float(os.getenv("DRAFT_PPO_EPSILON", "0.2"))
VALUE_LOSS_COEF   = float(os.getenv("DRAFT_VALUE_LOSS_COEF", "5.0"))
ENTROPY_COEF      = float(os.getenv("DRAFT_ENTROPY_COEF", "0.02"))
LEARNING_RATE     = float(os.getenv("DRAFT_LR", "3e-4"))
MAX_GRAD_NORM     = float(os.getenv("DRAFT_MAX_GRAD_NORM", "1.0"))
GAMMA             = float(os.getenv("DRAFT_GAMMA", "0.99"))
GAE_LAMBDA        = float(os.getenv("DRAFT_GAE_LAMBDA", "0.95"))

# ---------------------------------------------------------------------------
# Draft Model Entry Point
# ---------------------------------------------------------------------------

class DraftModelEntryPoint:

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "draft_model.pt")
        self.device = self._select_device()
        self.model = None
        self.optimizer = None
        self._lock = threading.Lock()

        # Stats
        self._train_steps = 0
        self._train_samples = 0
        self._last_policy_loss = 0.0
        self._last_value_loss = 0.0
        self._last_entropy = 0.0
        self._last_clip_frac = 0.0

        # Running advantage normalization
        self._adv_running_mean = 0.0
        self._adv_running_var = 1.0
        self._adv_ema_alpha = float(os.getenv("DRAFT_ADV_EMA_ALPHA", "0.01"))

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # -----------------------------------------------------------------------
    # Py4J interface: initialization
    # -----------------------------------------------------------------------

    def initializeDraftModel(self):
        """Create or load the draft model."""
        with self._lock:
            print(f"[DRAFT_PY] initializeDraftModel: device={self.device}, models_dir={self.models_dir}", flush=True)
            os.makedirs(self.models_dir, exist_ok=True)
            print(f"[DRAFT_PY] Building model (d_model={DRAFT_D_MODEL}, layers={DRAFT_NUM_LAYERS}, nhead={DRAFT_NHEAD})...", flush=True)
            self.model = MTGTransformerModel(
                input_dim=DRAFT_INPUT_DIM,
                d_model=DRAFT_D_MODEL,
                nhead=DRAFT_NHEAD,
                num_layers=DRAFT_NUM_LAYERS,
                dim_feedforward=DRAFT_DIM_FF,
                cand_feat_dim=DRAFT_CAND_DIM,
                token_vocab=DRAFT_TOKEN_VOCAB,
            ).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            if os.path.exists(self.model_path):
                print(f"[DRAFT_PY] Loading checkpoint from {self.model_path}...", flush=True)
                try:
                    ckpt = torch.load(self.model_path, map_location=self.device)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                        if "optimizer_state_dict" in ckpt:
                            try:
                                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                            except Exception:
                                pass
                        self._train_steps = ckpt.get("train_steps", 0)
                    else:
                        self.model.load_state_dict(ckpt, strict=False)
                    print(f"[DRAFT_PY] Loaded draft model (train_steps={self._train_steps})", flush=True)
                except Exception as e:
                    print(f"[DRAFT_PY] Failed to load draft model, starting fresh: {e}", flush=True)
            else:
                print(f"[DRAFT_PY] No checkpoint found at {self.model_path}, starting fresh", flush=True)

            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[DRAFT_PY] Draft model ready: device={self.device}, params={n_params:,}", flush=True)

    def saveDraftModel(self, path: str):
        with self._lock:
            if self.model is None:
                return
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_steps": self._train_steps,
                }, path)
                logger.info("Saved draft model to %s", path)
            except Exception as e:
                logger.error("Failed to save draft model: %s", e)

    def loadDraftModel(self, path: str):
        with self._lock:
            if self.model is None:
                return
            try:
                ckpt = torch.load(path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    self.model.load_state_dict(ckpt, strict=False)
                logger.info("Loaded draft model from %s", path)
            except Exception as e:
                logger.error("Failed to load draft model: %s", e)

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def _parse_infer_input(self, state_bytes, mask_bytes, token_id_bytes,
                           cand_feat_bytes, cand_id_bytes, num_cands, seq_len, dim):
        """Parse byte arrays into tensors for inference."""
        state = np.frombuffer(state_bytes, dtype="<f4").reshape(1, seq_len, dim)
        mask  = np.frombuffer(mask_bytes,  dtype="<i4").reshape(1, seq_len)
        toks  = np.frombuffer(token_id_bytes, dtype="<i4").reshape(1, seq_len)
        cf    = np.frombuffer(cand_feat_bytes, dtype="<f4").reshape(1, num_cands, dim)
        cids  = np.frombuffer(cand_id_bytes,   dtype="<i4").reshape(1, num_cands)

        dev = self.device
        seq_t   = torch.tensor(state, dtype=torch.float32, device=dev)
        mask_t  = torch.tensor(mask,  dtype=torch.bool,    device=dev)
        toks_t  = torch.tensor(toks,  dtype=torch.long,    device=dev)
        cf_t    = torch.tensor(cf,    dtype=torch.float32, device=dev)
        cids_t  = torch.tensor(cids,  dtype=torch.long,    device=dev)

        # candidate mask: all real (no padding in single-request inference)
        cmask_t = torch.ones(1, num_cands, dtype=torch.bool, device=dev)

        return seq_t, mask_t, toks_t, cf_t, cids_t, cmask_t

    def _score(self, head_id: str, state_bytes, mask_bytes, token_id_bytes,
               cand_feat_bytes, cand_id_bytes, num_cands, seq_len, dim) -> bytes:
        """Run model inference and return float32[numCands + 1] as bytes."""
        if self.model is None:
            # Return uniform + 0 value
            arr = np.zeros(num_cands + 1, dtype="<f4")
            arr[:num_cands] = 1.0 / max(num_cands, 1)
            return arr.tobytes()

        try:
            with self._lock:
                seq_t, mask_t, toks_t, cf_t, cids_t, cmask_t = self._parse_infer_input(
                    state_bytes, mask_bytes, token_id_bytes,
                    cand_feat_bytes, cand_id_bytes, num_cands, seq_len, dim)

                self.model.eval()
                with torch.inference_mode():
                    probs, value = self.model.score_candidates(
                        seq_t, mask_t, toks_t, cf_t, cids_t, cmask_t,
                        head_id=head_id)

                probs_np = probs[0, :num_cands].cpu().float().numpy()
                value_np = value[0].cpu().float().item()

                out = np.zeros(num_cands + 1, dtype="<f4")
                out[:num_cands] = probs_np
                out[num_cands]  = value_np
                return out.tobytes()

        except Exception as e:
            logger.error("Score error (%s): %s", head_id, e)
            arr = np.zeros(num_cands + 1, dtype="<f4")
            arr[:num_cands] = 1.0 / max(num_cands, 1)
            return arr.tobytes()

    def scoreDraftPick(self, state_bytes, mask_bytes, token_id_bytes,
                       cand_feat_bytes, cand_id_bytes,
                       num_cands, seq_len, dim_per_token) -> bytes:
        """Score pack cards via pick_head (default scorer)."""
        return self._score("pick", state_bytes, mask_bytes, token_id_bytes,
                           cand_feat_bytes, cand_id_bytes, num_cands, seq_len, dim_per_token)

    def scoreDraftConstruction(self, state_bytes, mask_bytes, token_id_bytes,
                               cand_feat_bytes, cand_id_bytes,
                               num_cands, seq_len, dim_per_token) -> bytes:
        """Score pool cards via construction_head (card_select scorer)."""
        return self._score("card_select", state_bytes, mask_bytes, token_id_bytes,
                           cand_feat_bytes, cand_id_bytes, num_cands, seq_len, dim_per_token)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def trainDraftBatch(self,
                        state_bytes, mask_bytes, token_id_bytes,
                        cand_feat_bytes, cand_id_bytes, cand_mask_bytes,
                        chosen_idx_bytes, old_logp_bytes, old_value_bytes,
                        reward_bytes, done_bytes, head_idx_bytes,
                        num_steps, seq_len, dim_per_token, max_cands):
        """PPO update for a batch of draft decisions (picks + construction)."""
        if self.model is None:
            return

        try:
            with self._lock:
                self._ppo_update(
                    state_bytes, mask_bytes, token_id_bytes,
                    cand_feat_bytes, cand_id_bytes, cand_mask_bytes,
                    chosen_idx_bytes, old_logp_bytes, old_value_bytes,
                    reward_bytes, done_bytes, head_idx_bytes,
                    num_steps, seq_len, dim_per_token, max_cands)
        except Exception as e:
            logger.error("trainDraftBatch error: %s\n%s", e, traceback.format_exc())

    def _ppo_update(self,
                    state_bytes, mask_bytes, token_id_bytes,
                    cand_feat_bytes, cand_id_bytes, cand_mask_bytes,
                    chosen_idx_bytes, old_logp_bytes, old_value_bytes,
                    reward_bytes, done_bytes, head_idx_bytes,
                    num_steps, seq_len, dim, max_cands):
        """Compute PPO loss and update model."""
        dev = self.device

        states   = torch.tensor(np.frombuffer(state_bytes,   dtype="<f4").reshape(num_steps, seq_len, dim),         dtype=torch.float32, device=dev)
        masks    = torch.tensor(np.frombuffer(mask_bytes,    dtype="<i4").reshape(num_steps, seq_len),              dtype=torch.bool,    device=dev)
        tok_ids  = torch.tensor(np.frombuffer(token_id_bytes, dtype="<i4").reshape(num_steps, seq_len),             dtype=torch.long,    device=dev)
        cf       = torch.tensor(np.frombuffer(cand_feat_bytes, dtype="<f4").reshape(num_steps, max_cands, dim),     dtype=torch.float32, device=dev)
        cids     = torch.tensor(np.frombuffer(cand_id_bytes,   dtype="<i4").reshape(num_steps, max_cands),          dtype=torch.long,    device=dev)
        cmask    = torch.tensor(np.frombuffer(cand_mask_bytes, dtype="<i4").reshape(num_steps, max_cands),          dtype=torch.bool,    device=dev)
        chosen   = torch.tensor(np.frombuffer(chosen_idx_bytes, dtype="<i4").reshape(num_steps),                   dtype=torch.long,    device=dev)
        old_logp = torch.tensor(np.frombuffer(old_logp_bytes,  dtype="<f4").reshape(num_steps),                   dtype=torch.float32, device=dev)
        old_val  = torch.tensor(np.frombuffer(old_value_bytes, dtype="<f4").reshape(num_steps),                   dtype=torch.float32, device=dev)
        rewards  = torch.tensor(np.frombuffer(reward_bytes,    dtype="<f4").reshape(num_steps),                   dtype=torch.float32, device=dev)
        dones    = torch.tensor(np.frombuffer(done_bytes,      dtype="<i4").reshape(num_steps),                   dtype=torch.float32, device=dev)
        head_ids = np.frombuffer(head_idx_bytes, dtype="<i4").reshape(num_steps)

        # Compute GAE returns
        returns = self._compute_gae(rewards.cpu().numpy(), old_val.cpu().numpy(),
                                    dones.cpu().numpy())
        returns_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        advantages = returns_t - old_val
        advantages = self._normalize_advantages(advantages)

        self.model.train()
        self.optimizer.zero_grad()

        total_policy_loss = torch.tensor(0.0, device=dev)
        total_value_loss  = torch.tensor(0.0, device=dev)
        total_entropy     = torch.tensor(0.0, device=dev)
        total_clip_frac   = 0.0

        # Process in two head groups: pick (0) and construction (1)
        for head_idx_val, head_name in [(0, "pick"), (1, "card_select")]:
            idx_mask = torch.tensor(head_ids == head_idx_val, dtype=torch.bool, device=dev)
            if not idx_mask.any():
                continue

            s_s = states[idx_mask]
            m_s = masks[idx_mask]
            t_s = tok_ids[idx_mask]
            c_s = cf[idx_mask]
            ci_s = cids[idx_mask]
            cm_s = cmask[idx_mask]
            ch_s = chosen[idx_mask]
            ol_s = old_logp[idx_mask]
            adv_s = advantages[idx_mask]
            ret_s = returns_t[idx_mask]
            ov_s  = old_val[idx_mask]

            probs, values = self.model.score_candidates(
                s_s, m_s, t_s, c_s, ci_s, cm_s, head_id=head_name)

            # Gather chosen probabilities
            chosen_probs = probs.gather(1, ch_s.unsqueeze(1)).squeeze(1)
            new_logp = torch.log(chosen_probs.clamp(min=1e-8))
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()

            # PPO clip ratio
            ratio = torch.exp(new_logp - ol_s)
            surr1 = ratio * adv_s
            surr2 = ratio.clamp(1 - PPO_EPSILON, 1 + PPO_EPSILON) * adv_s
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values_flat = values.squeeze(-1)
            clipped_val = ov_s + (values_flat - ov_s).clamp(-PPO_EPSILON, PPO_EPSILON)
            val_loss1 = (values_flat - ret_s).pow(2)
            val_loss2 = (clipped_val - ret_s).pow(2)
            value_loss = 0.5 * torch.max(val_loss1, val_loss2).mean()

            clip_frac = ((ratio - 1.0).abs() > PPO_EPSILON).float().mean().item()

            n = idx_mask.sum().item()
            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss  = total_value_loss  + value_loss
            total_entropy     = total_entropy     + entropy
            total_clip_frac   += clip_frac

        loss = (total_policy_loss
                + VALUE_LOSS_COEF * total_value_loss
                - ENTROPY_COEF * total_entropy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        self._train_steps += 1
        self._train_samples += num_steps
        self._last_policy_loss = total_policy_loss.item()
        self._last_value_loss  = total_value_loss.item()
        self._last_entropy     = total_entropy.item()
        self._last_clip_frac   = total_clip_frac

        logger.debug("Draft PPO step %d: policy=%.4f value=%.4f entropy=%.4f clip=%.3f",
                     self._train_steps, self._last_policy_loss, self._last_value_loss,
                     self._last_entropy, self._last_clip_frac)

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE returns."""
        n = len(rewards)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if dones[t] == 1 else (values[t + 1] if t + 1 < n else 0.0)
            delta = rewards[t] + GAMMA * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
            returns[t] = gae + values[t]
        return returns

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        alpha = self._adv_ema_alpha
        batch_mean = advantages.mean().item()
        batch_var  = advantages.var().item() if advantages.numel() > 1 else 1.0
        self._adv_running_mean = (1 - alpha) * self._adv_running_mean + alpha * batch_mean
        self._adv_running_var  = (1 - alpha) * self._adv_running_var  + alpha * batch_var
        std = math.sqrt(max(self._adv_running_var, 1e-8))
        return (advantages - self._adv_running_mean) / std

    # -----------------------------------------------------------------------
    # Stats / metrics
    # -----------------------------------------------------------------------

    def getDraftTrainingStats(self):
        return {"train_steps": self._train_steps, "train_samples": self._train_samples}

    def getDraftLossMetrics(self):
        return {
            "policy_loss": float(self._last_policy_loss),
            "value_loss":  float(self._last_value_loss),
            "entropy":     float(self._last_entropy),
            "clip_frac":   float(self._last_clip_frac),
            "train_steps": int(self._train_steps),
        }

    class Java:
        implements = ["mage.player.ai.rl.DraftPythonEntryPoint"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("[DRAFT_PY] draft_py4j_entry_point.py starting...", flush=True)
    parser = argparse.ArgumentParser(description="Draft Model Py4J server")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("DRAFT_PY4J_PORT", "25360")))
    args = parser.parse_args()

    print(f"[DRAFT_PY] importing torch...", flush=True)
    import torch as _torch_check
    print(f"[DRAFT_PY] torch {_torch_check.__version__}, CUDA={_torch_check.cuda.is_available()}", flush=True)

    models_dir = os.getenv("DRAFT_MODELS_DIR", os.path.join(script_dir, "draft_models"))
    print(f"[DRAFT_PY] models_dir={models_dir}", flush=True)
    os.makedirs(models_dir, exist_ok=True)

    entry = DraftModelEntryPoint(models_dir)

    # Python server listens on args.port; Java connects to it.
    # This mirrors the pattern of py4j_entry_point.py.
    max_retries = 5
    gateway = None
    for attempt in range(max_retries):
        try:
            gateway = ClientServer(
                java_parameters=JavaParameters(),
                python_parameters=PythonParameters(port=args.port),
                python_server_entry_point=entry,
            )
            logger.info("Draft model Py4J server started on port %d", args.port)
            print(f"[draft_model] Server ready on port {args.port}", flush=True)
            break
        except Exception as e:
            logger.error("Failed to start Py4J gateway (attempt %d/%d): %s",
                         attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if gateway:
            gateway.shutdown()


if __name__ == "__main__":
    main()
