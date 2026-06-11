"""Re-export fp32 ONNX (versioned dir + .active_dir) for the baseline backup
and all live profiles. One-time repair after the fp16 calibration bug."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MLCODE = REPO / 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode'
PROFILES = REPO / 'Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles'
sys.path.insert(0, str(MLCODE))
os.environ['ONNX_EXPORT_FP16'] = '0'
from onnx_export import export_all_heads  # noqa: E402

TARGETS = [
    (REPO / 'local-training/backups/spy_value_baseline_20260531/model_latest.pt',
     REPO / 'local-training/backups/spy_value_baseline_20260531/onnx'),
]
for p in ['Pauper-Spy-Combo-Value', 'Pauper-Wildfire-Value',
          'Pauper-Rally-Anchor-Value', 'Pauper-Affinity-Anchor-Value']:
    TARGETS.append((PROFILES / p / 'models/model_latest.pt',
                    PROFILES / p / 'models/onnx'))

for model_path, onnx_dir in TARGETS:
    if not model_path.exists():
        print(f'SKIP (no model): {model_path}')
        continue
    stamp = datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%S_%f')
    out_dir = onnx_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'=== {model_path} -> {out_dir}')
    export_all_heads(str(model_path), str(out_dir))
    (out_dir / '.export_version').write_text('3')
    (onnx_dir / '.active_dir').write_text(stamp)
    print(f'    .active_dir -> {stamp}')
print('DONE')
