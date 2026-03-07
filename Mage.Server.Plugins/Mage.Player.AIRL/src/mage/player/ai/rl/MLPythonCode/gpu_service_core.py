import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional


_ENV_LOCK = threading.Lock()


def _header_env(headers: Dict[str, str]) -> Dict[str, str]:
    env = {}
    for key, value in headers.items():
        if key.startswith("env."):
            env[key[4:]] = value
    return env


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    with _ENV_LOCK:
        previous = {}
        removed = []
        for key, value in overrides.items():
            if key in os.environ:
                previous[key] = os.environ[key]
            else:
                removed.append(key)
            os.environ[key] = value
        try:
            yield
        finally:
            for key, value in previous.items():
                os.environ[key] = value
            for key in removed:
                os.environ.pop(key, None)


class ProfileContext:
    def __init__(self, profile_id: str, headers: Dict[str, str], role: str = "inference"):
        from py4j_entry_point import PythonEntryPoint

        self.profile_id = profile_id
        self.role = (role or "inference").strip().lower()
        self.env = _header_env(headers)
        self.env["MODEL_PROFILE"] = profile_id
        self.env["PY_BACKEND_MODE"] = "single"
        self.env["INFER_WORKERS"] = "0"
        self.env["MODEL_RELOAD_EVERY_MS"] = "0"
        self.env["MODEL_SYNC_EVERY_MS"] = "0"
        self.env["MODEL_SYNC_EVERY_TRAIN_STEPS"] = "0"
        self.env.setdefault("MULLIGAN_DEVICE", "cpu")
        # PythonEntryPoint expects the canonical role names ("inference" / "learner").
        self.env["PY_ROLE"] = self.role
        self.lock = threading.RLock()
        with _temporary_env(self.env):
            self.entry = PythonEntryPoint()
        if self.role == "learner":
            self._ensure_model_initialized()

    def _ensure_model_initialized(self) -> None:
        with self.lock:
            model = getattr(self.entry, "model", None)
            optimizer = getattr(self.entry, "optimizer", None)
            if model is None or optimizer is None:
                self.entry.initializeModel()

    def score_batch(
        self,
        seq_bytes: bytes,
        mask_bytes: bytes,
        token_bytes: bytes,
        cand_feat_bytes: bytes,
        cand_ids_bytes: bytes,
        cand_mask_bytes: bytes,
        policy_key: str,
        head_id: str,
        pick_index: int,
        min_targets: int,
        max_targets: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        max_candidates: int,
        cand_feat_dim: int,
    ) -> bytes:
        with self.lock:
            return self.entry.scoreCandidatesPolicyFlat(
                seq_bytes,
                mask_bytes,
                token_bytes,
                cand_feat_bytes,
                cand_ids_bytes,
                cand_mask_bytes,
                policy_key,
                head_id,
                pick_index,
                min_targets,
                max_targets,
                batch_size,
                seq_len,
                d_model,
                max_candidates,
                cand_feat_dim,
            )

    def train_batch(
        self,
        sequences: bytes,
        masks: bytes,
        token_ids: bytes,
        cand_feat: bytes,
        cand_ids: bytes,
        cand_mask: bytes,
        rewards: bytes,
        chosen_indices: bytes,
        chosen_count: bytes,
        old_logp_total: bytes,
        old_value: bytes,
        sample_weights: bytes,
        dones: bytes,
        head_idx: bytes,
        batch_size: int,
        seq_len: int,
        d_model: int,
        max_candidates: int,
        cand_feat_dim: int,
    ) -> bool:
        self._ensure_model_initialized()
        with self.lock:
            return bool(
                self.entry.trainCandidatesMultiFlat(
                    sequences,
                    masks,
                    token_ids,
                    cand_feat,
                    cand_ids,
                    cand_mask,
                    rewards,
                    chosen_indices,
                    chosen_count,
                    old_logp_total,
                    old_value,
                    sample_weights,
                    dones,
                    head_idx,
                    batch_size,
                    seq_len,
                    d_model,
                    max_candidates,
                    cand_feat_dim,
                )
            )

    def predict_mulligan(self, features) -> float:
        with self.lock:
            return float(self.entry.predictMulligan(features))

    def predict_mulligan_scores(self, features) -> bytes:
        with self.lock:
            return self.entry.predictMulliganScores(features)

    def train_mulligan(
        self,
        features_bytes: bytes,
        decisions_bytes: bytes,
        outcomes_bytes: bytes,
        game_lengths_bytes: bytes,
        early_land_scores_bytes: bytes,
        overrides_bytes: bytes,
        batch_size: int,
    ) -> None:
        with self.lock:
            self.entry.trainMulligan(
                features_bytes,
                decisions_bytes,
                outcomes_bytes,
                game_lengths_bytes,
                early_land_scores_bytes,
                overrides_bytes,
                batch_size,
            )

    def save_mulligan_model(self) -> None:
        with self.lock:
            self.entry.saveMulliganModel()

    def save_model(self, path: str) -> None:
        if self.role == "learner":
            self._ensure_model_initialized()
        with self.lock:
            self.entry.saveModel(path)

    def save_latest_model_atomic(self, path: Optional[str] = None) -> bool:
        if self.role == "learner":
            self._ensure_model_initialized()
        with self.lock:
            return bool(self.entry.saveLatestModelAtomic(path))

    def reload_latest_model_if_newer(self, path: Optional[str] = None) -> bool:
        with self.lock:
            return bool(self.entry.reloadLatestModelIfNewer(path))

    def get_device_info(self) -> str:
        with self.lock:
            return str(self.entry.getDeviceInfo())

    def get_main_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "train_steps": int(self.entry.train_step_counter),
                "train_samples": int(self.entry.main_train_sample_counter),
            }

    def get_mulligan_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "train_steps": int(self.entry.mulligan_train_step_counter),
                "train_samples": int(self.entry.mulligan_train_sample_counter),
            }

    def get_health_stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "gpu_oom_count": int(self.entry.cuda_mgr.get_oom_count()),
            }

    def reset_health_stats(self) -> None:
        with self.lock:
            self.entry.cuda_mgr.reset_oom_count()

    def record_game_result(self, last_value_prediction: float, won: bool) -> None:
        with self.lock:
            self.entry.record_value_prediction(last_value_prediction, won)

    def get_value_head_metrics(self) -> Dict[str, object]:
        with self.lock:
            metrics = dict(self.entry.get_value_metrics())
            metrics["use_gae"] = bool(getattr(self.entry, "use_gae", False))
            return metrics


def merge_segments(requests: List[List[bytes]]) -> List[bytes]:
    if not requests:
        return []
    count = len(requests[0])
    return [b"".join(request[i] for request in requests) for i in range(count)]


def feature_array_from_bytes(features_bytes: bytes):
    import numpy as np

    if not features_bytes:
        return np.zeros((0,), dtype="<f4")
    return np.frombuffer(features_bytes, dtype="<f4")
