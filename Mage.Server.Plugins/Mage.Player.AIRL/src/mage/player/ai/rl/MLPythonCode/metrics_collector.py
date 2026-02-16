import os
import time
import math
import torch
from logging_utils import logger, LogCategory


class MetricsCollector:
    """Tracks training metrics, value head quality, GAE parameters, and entropy scheduling."""

    def __init__(self):
        # Training step counters
        self.train_step_counter = 0
        self.mulligan_train_step_counter = 0
        self.score_call_counter = 0
        self.infer_counter = 0  # For periodic GC during inference
        self.main_train_sample_counter = 0
        self.mulligan_train_sample_counter = 0

        # Timing metrics (EMA with alpha=0.1)
        self.train_time_ms_ema = None
        self.infer_time_ms_ema = None
        self.mulligan_time_ms_ema = None
        self.timing_alpha = 0.1

        # Entropy schedule configuration
        self.entropy_start = float(os.getenv('ENTROPY_START', '0.20'))
        self.entropy_end = float(os.getenv('ENTROPY_END', '0.03'))
        self.entropy_decay_steps = int(
            os.getenv('ENTROPY_DECAY_STEPS', '50000'))

        # GAE (Generalized Advantage Estimation) configuration
        self.gae_lambda = float(os.getenv('GAE_LAMBDA', '0.95'))
        self.use_gae = bool(int(os.getenv('USE_GAE', '1')))

        # Auto-GAE: Start with MC returns, auto-switch to GAE once value head is healthy
        self.gae_auto_enable = bool(int(os.getenv('GAE_AUTO_ENABLE', '0')))
        self.gae_auto_threshold = float(
            os.getenv('GAE_AUTO_THRESHOLD', '0.65'))
        self.gae_auto_min_samples = int(
            os.getenv('GAE_AUTO_MIN_SAMPLES', '200'))

        # GAE lambda schedule
        self.gae_lambda_high = float(os.getenv('GAE_LAMBDA_HIGH', '0.99'))
        self.gae_lambda_low = float(os.getenv('GAE_LAMBDA_LOW', '0.90'))
        self.gae_lambda_decay_steps = int(
            os.getenv('GAE_LAMBDA_DECAY_STEPS', '5000'))
        self.current_gae_lambda = self.gae_lambda_high if self.use_gae else 1.0
        self.gae_enabled_step = 0 if self.use_gae else None

        # Value head quality tracking for auto-GAE
        self.value_wins = []
        self.value_losses = []
        self.value_window_size = 100

        # Latest training loss components (for Prometheus export)
        self.latest_total_loss = 0.0
        self.latest_policy_loss = 0.0
        self.latest_value_loss = 0.0
        self.latest_entropy = 0.0
        self.latest_entropy_coef = 0.0
        self.latest_clip_frac = 0.0
        self.latest_approx_kl = 0.0
        self.latest_batch_size = 0
        self.latest_advantage_mean = 0.0

    def get_entropy_coefficient(self):
        """
        Calculate entropy coefficient with linear decay.
        Formula: coeff = start - (start - end) * min(1, steps / decay_steps)
        """
        if self.train_step_counter >= self.entropy_decay_steps:
            return self.entropy_end
        progress = self.train_step_counter / max(1, self.entropy_decay_steps)
        coeff = self.entropy_start - (self.entropy_start - self.entropy_end) * progress
        return coeff

    def record_value_prediction(self, value_pred, won):
        """Record a value prediction for auto-GAE tracking."""
        if won:
            self.value_wins.append(float(value_pred))
            if len(self.value_wins) > self.value_window_size:
                self.value_wins.pop(0)
        else:
            self.value_losses.append(float(value_pred))
            if len(self.value_losses) > self.value_window_size:
                self.value_losses.pop(0)

        self._check_auto_gae()

    def _check_auto_gae(self):
        """Check if value head is healthy enough to switch from MC returns to GAE."""
        if not self.gae_auto_enable or self.use_gae:
            return

        win_samples = len(self.value_wins)
        loss_samples = len(self.value_losses)

        if win_samples < self.gae_auto_min_samples // 2 or loss_samples < self.gae_auto_min_samples // 2:
            return

        win_correct = sum(1 for v in self.value_wins if v > 0)
        loss_correct = sum(1 for v in self.value_losses if v < 0)

        win_accuracy = win_correct / win_samples if win_samples > 0 else 0.0
        loss_accuracy = loss_correct / loss_samples if loss_samples > 0 else 0.0
        accuracy = min(win_accuracy, loss_accuracy)

        if accuracy >= self.gae_auto_threshold:
            self.use_gae = True
            if self.gae_enabled_step is None:
                self.gae_enabled_step = self.train_step_counter
            self.current_gae_lambda = self.gae_lambda_high
            logger.info(
                LogCategory.MODEL_TRAIN,
                "Auto-GAE triggered! Value accuracy %.1f%% (win=%.1f%%, loss=%.1f%%) >= %.1f%% threshold. Switching from MC returns to GAE.",
                accuracy * 100, win_accuracy * 100, loss_accuracy *
                100, self.gae_auto_threshold * 100
            )

    def get_value_metrics(self):
        """Return current value head metrics for external monitoring."""
        win_samples = len(self.value_wins)
        loss_samples = len(self.value_losses)
        total = win_samples + loss_samples

        if win_samples == 0 or loss_samples == 0:
            return {"accuracy": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "samples": 0}

        win_correct = sum(1 for v in self.value_wins if v > 0)
        loss_correct = sum(1 for v in self.value_losses if v < 0)

        win_accuracy = win_correct / win_samples
        loss_accuracy = loss_correct / loss_samples
        accuracy = min(win_accuracy, loss_accuracy)

        avg_win = sum(self.value_wins) / win_samples
        avg_loss = sum(self.value_losses) / loss_samples

        return {"accuracy": accuracy, "avg_win": avg_win, "avg_loss": avg_loss, "samples": total}

    def update_gae_lambda_schedule(self):
        """Update self.current_gae_lambda based on value quality + time since enabling."""
        if not self.use_gae:
            return
        if self.gae_enabled_step is None:
            self.gae_enabled_step = self.train_step_counter

        high = float(self.gae_lambda_high)
        low = float(self.gae_lambda_low)
        decay_steps = max(1, int(self.gae_lambda_decay_steps))

        steps_since = max(0, int(self.train_step_counter) -
                          int(self.gae_enabled_step))
        frac = min(1.0, steps_since / float(decay_steps))
        target = high + (low - high) * frac

        # Gate decay on value quality
        m = self.get_value_metrics()
        if m.get("samples", 0) < self.gae_auto_min_samples or m.get("accuracy", 0.0) < self.gae_auto_threshold:
            target = high

        lo = min(high, low)
        hi = max(high, low)
        self.current_gae_lambda = float(max(lo, min(hi, target)))

        if self.train_step_counter % 500 == 0:
            logger.info(LogCategory.MODEL_TRAIN,
                        "GAE schedule: enabled=%s lambda=%.3f (high=%.3f low=%.3f steps_since=%d acc=%.1f%% samples=%d)",
                        self.use_gae, self.current_gae_lambda, high, low, steps_since,
                        m.get("accuracy", 0.0) * 100.0, m.get("samples", 0))

    def update_timing_metric(self, kind: str, elapsed_ms: float):
        """Update timing EMA for a given operation type."""
        alpha = self.timing_alpha
        if kind == "train":
            self.train_time_ms_ema = elapsed_ms if self.train_time_ms_ema is None else \
                (alpha * elapsed_ms + (1 - alpha) * self.train_time_ms_ema)
        elif kind == "infer":
            self.infer_time_ms_ema = elapsed_ms if self.infer_time_ms_ema is None else \
                (alpha * elapsed_ms + (1 - alpha) * self.infer_time_ms_ema)
        elif kind == "mulligan":
            self.mulligan_time_ms_ema = elapsed_ms if self.mulligan_time_ms_ema is None else \
                (alpha * elapsed_ms + (1 - alpha) * self.mulligan_time_ms_ema)

    def get_timing_metrics(self):
        """Get current timing metrics for Prometheus export."""
        return {
            'train_time_ms': self.train_time_ms_ema if self.train_time_ms_ema is not None else 0.0,
            'infer_time_ms': self.infer_time_ms_ema if self.infer_time_ms_ema is not None else 0.0,
            'mulligan_time_ms': self.mulligan_time_ms_ema if self.mulligan_time_ms_ema is not None else 0.0,
        }

    def record_train_losses(self, total_loss, policy_loss, value_loss, entropy, 
                           entropy_coef, clip_frac, approx_kl, batch_size, advantage_mean):
        """Record the latest training loss components for Prometheus export."""
        self.latest_total_loss = float(total_loss)
        self.latest_policy_loss = float(policy_loss)
        self.latest_value_loss = float(value_loss)
        self.latest_entropy = float(entropy)
        self.latest_entropy_coef = float(entropy_coef)
        self.latest_clip_frac = float(clip_frac)
        self.latest_approx_kl = float(approx_kl)
        self.latest_batch_size = int(batch_size)
        self.latest_advantage_mean = float(advantage_mean)

    def get_training_loss_metrics(self):
        """Get latest training loss components for Prometheus export."""
        return {
            'total_loss': self.latest_total_loss,
            'policy_loss': self.latest_policy_loss,
            'value_loss': self.latest_value_loss,
            'entropy': self.latest_entropy,
            'entropy_coef': self.latest_entropy_coef,
            'clip_frac': self.latest_clip_frac,
            'approx_kl': self.latest_approx_kl,
            'batch_size': float(self.latest_batch_size),
            'advantage_mean': self.latest_advantage_mean,
        }

    def compute_gae(self, rewards, values, gamma=0.99, gae_lambda=None, dones=None):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor [batch_size] containing immediate rewards
            values: Tensor [batch_size] containing value predictions V(s_t)
            gamma: Discount factor
            gae_lambda: GAE lambda (overrides self.gae_lambda if provided)
            dones: Optional Tensor [batch_size] where 1 marks terminal steps

        Returns:
            advantages: Tensor [batch_size] containing GAE advantages
            returns: Tensor [batch_size] containing value targets (advantage + value)
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        # Calculate TD errors: δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        if dones is not None:
            d = dones.to(device=values.device).view(-1)
            next_values = next_values * (1.0 - d.float())

        deltas = rewards + gamma * next_values - values

        # Calculate GAE advantages by iterating backward
        gae = 0.0
        lam = self.gae_lambda if gae_lambda is None else float(gae_lambda)
        for t in reversed(range(batch_size)):
            if dones is not None and float(d[t].item()) != 0.0:
                gae = 0.0
            gae = deltas[t] + gamma * lam * gae
            advantages[t] = gae

        # Value targets = advantages + current value estimates
        returns = advantages + values

        return advantages, returns
