import os
import torch
from collections import OrderedDict
from mtg_transformer import MTGTransformerModel
from logging_utils import logger, LogCategory
from profile_paths import profile_models_dir


class SnapshotManager:
    """Manages snapshot save/load and LRU caching for opponent models."""

    def __init__(self, device):
        self.device = device
        self.snapshot_dir = os.getenv(
            'SNAPSHOT_DIR',
            f'{profile_models_dir()}/snapshots'
        )
        self.snapshot_save_every_steps = int(os.getenv('SNAPSHOT_SAVE_EVERY_STEPS', '1000'))
        self.snapshot_max_files = int(os.getenv('SNAPSHOT_MAX_FILES', '20'))
        self.snapshot_cache_size = int(os.getenv('SNAPSHOT_CACHE_SIZE', '2'))
        self.model_kwargs = {
            "d_model": int(os.getenv("MODEL_D_MODEL", "128")),
            "nhead": int(os.getenv("MODEL_NHEAD", "4")),
            "num_layers": int(os.getenv("MODEL_NUM_LAYERS", "2")),
            "dim_feedforward": int(os.getenv("MODEL_DIM_FEEDFORWARD", "512")),
            "cand_feat_dim": 48,
        }
        self.snapshot_models = OrderedDict()  # LRU: key -> model

        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        except Exception:
            self.snapshot_dir = ""

    def resolve_snapshot_path(self, snap_id: str) -> str:
        if not snap_id:
            return ""
        snap_id = str(snap_id)
        if snap_id.startswith("snap:"):
            snap_id = snap_id[len("snap:"):]
        if os.path.isabs(snap_id):
            return snap_id
        if not self.snapshot_dir:
            return ""
        return os.path.join(self.snapshot_dir, snap_id)

    def get_snapshot_model(self, snap_id: str):
        """Return an eval-only MTGTransformerModel for a snapshot id (LRU-cached)."""
        path = self.resolve_snapshot_path(snap_id)
        if not path or not os.path.exists(path):
            return None

        key = os.path.abspath(path)
        # LRU hit
        if key in self.snapshot_models:
            m = self.snapshot_models.pop(key)
            self.snapshot_models[key] = m
            return m

        # Load new snapshot (with retry for transient failures).
        # Auto-detect the snapshot's OWN architecture from its state_dict so a
        # main net of a different size (e.g. d_model=256) can still serve a
        # smaller opponent snapshot (e.g. 128) -- heterogeneous-dim league play.
        # Without this, MTGTransformerModel(**self.model_kwargs) uses the global
        # MODEL_D_MODEL and shape-mismatches on m.load(path).
        kwargs = dict(self.model_kwargs)
        try:
            _sd = torch.load(path, map_location='cpu', weights_only=False)
            _sd = _sd.get('state_dict', _sd) if isinstance(_sd, dict) else _sd
            if 'cls_token' in _sd:
                kwargs['d_model'] = int(_sd['cls_token'].shape[-1])
            _nl = len({k.split('.')[1] for k in _sd if k.startswith('transformer_layers.')})
            if _nl > 0:
                kwargs['num_layers'] = _nl
            if 'transformer_layers.0.linear1.weight' in _sd:
                kwargs['dim_feedforward'] = int(_sd['transformer_layers.0.linear1.weight'].shape[0])
            # nhead must divide d_model; fall back to 8 for 256, else 4
            if kwargs['d_model'] % kwargs.get('nhead', 4) != 0:
                kwargs['nhead'] = 8 if kwargs['d_model'] % 8 == 0 else 4
        except Exception:
            kwargs = dict(self.model_kwargs)
        import time
        for attempt in range(2):
            try:
                m = MTGTransformerModel(**kwargs).to('cpu')
                m.load(path)
                m = m.to(self.device)
                m.eval()
                break  # Success
            except Exception as e:
                if attempt == 0:
                    # First failure: wait and retry (file might be mid-write)
                    time.sleep(0.1)
                    continue
                else:
                    # Second failure: give up
                    logger.warning(LogCategory.MODEL_LOAD,
                                   "Failed to load snapshot model %s after retry: %s", path, str(e))
                    return None

        self.snapshot_models[key] = m
        # Evict LRU
        while len(self.snapshot_models) > max(0, self.snapshot_cache_size):
            old_key, old_model = self.snapshot_models.popitem(last=False)
            try:
                # Move to CPU before deletion to free VRAM immediately
                if torch.cuda.is_available():
                    old_model.cpu()
                del old_model
            except Exception:
                pass
        # Trigger garbage collection and clear CUDA cache
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        return m

    def get_policy_model(self, policy_key: str, main_model):
        """Get policy model by key - returns main model or snapshot."""
        if not policy_key or str(policy_key).strip() == "" or str(policy_key).strip() == "train":
            return main_model
        if str(policy_key).startswith("snap:") or self.snapshot_dir:
            snap = self.get_snapshot_model(str(policy_key))
            if snap is not None:
                return snap
        return main_model

    def maybe_save_snapshot(self, train_step_counter: int, save_fn):
        """Save snapshot if at checkpoint interval. save_fn(path) should save the model."""
        if not self.snapshot_dir or self.snapshot_save_every_steps <= 0:
            return
        if train_step_counter <= 0:
            return
        if train_step_counter % self.snapshot_save_every_steps != 0:
            return
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            fname = f"snapshot_step_{train_step_counter}.pt"
            path = os.path.join(self.snapshot_dir, fname)
            
            # Atomic write: save to temp file, then rename
            temp_path = path + ".tmp"
            save_fn(temp_path)
            os.replace(temp_path, path)  # Atomic on most filesystems
            
            self.prune_snapshots()
            logger.info(LogCategory.MODEL_SAVE,
                        "Snapshot saved at step %d -> %s",
                        train_step_counter, path)
        except Exception as e:
            logger.warning(LogCategory.MODEL_SAVE,
                           "Failed to save snapshot at step %d: %s",
                           train_step_counter, str(e))

    def prune_snapshots(self):
        """Remove oldest snapshots to stay under max_files limit."""
        if not self.snapshot_dir or self.snapshot_max_files <= 0:
            return
        try:
            files = []
            for fn in os.listdir(self.snapshot_dir):
                # Clean up temp files from interrupted saves
                if fn.endswith(".pt.tmp"):
                    try:
                        os.remove(os.path.join(self.snapshot_dir, fn))
                    except Exception:
                        pass
                    continue
                if fn.endswith(".pt") and fn.startswith("snapshot_step_"):
                    files.append(os.path.join(self.snapshot_dir, fn))
            files.sort(key=lambda p: os.path.getmtime(p))
            while len(files) > self.snapshot_max_files:
                p = files.pop(0)
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass
