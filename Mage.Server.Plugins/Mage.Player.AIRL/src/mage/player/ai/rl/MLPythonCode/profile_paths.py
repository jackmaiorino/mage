"""
Shared helper for resolving profile-aware model/log directory paths.

When MODEL_PROFILE is set, all artifacts live under:
  rl/profiles/<profile>/models/
  rl/profiles/<profile>/logs/

Without MODEL_PROFILE the legacy flat layout is used (backward compat):
  rl/models/
  rl/logs/
"""
import os

_RL_BASE = 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl'


def profile_models_dir() -> str:
    profile = os.getenv('MODEL_PROFILE', '').strip()
    if profile:
        return os.getenv('RL_MODELS_DIR', f'{_RL_BASE}/profiles/{profile}/models')
    return os.getenv('RL_MODELS_DIR', f'{_RL_BASE}/models')


def profile_logs_dir() -> str:
    profile = os.getenv('MODEL_PROFILE', '').strip()
    if profile:
        return os.getenv('RL_LOGS_DIR', f'{_RL_BASE}/profiles/{profile}/logs')
    return os.getenv('RL_LOGS_DIR', f'{_RL_BASE}/logs')
