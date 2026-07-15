# Balanced-mirror baseline: latest Rally ckpt vs CP7-Rally MIRROR, both sides.
# Pass 1 = agent ON THE PLAY (default). Pass 2 = agent ON THE DRAW (EVAL_OPPONENT_ON_PLAY=1).
# Deterministic, seed 5151, n=128/side. Confirms play-side ~= gauntlet mirror (35-43%) and
# gives the draw-side + true balanced baseline for the mirror training experiment.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Rally-Anchor-Value\models"
$out = "local-training/mirror_branch_playdraw.log"
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "E:\mage-training\backups\rally_mirror_branch\model.pt" "$md\model.pt" -Force
Copy-Item "E:\mage-training\backups\rally_mirror_branch\model.pt" "$md\model_latest.pt" -Force
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_FEATURES_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
"=== RALLY MIRROR PLAY/DRAW $(Get-Date) md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) ===" | Out-File $out

# Pass 1: agent ON THE PLAY
Remove-Item Env:\EVAL_OPPONENT_ON_PLAY -ErrorAction SilentlyContinue
"=== PASS agent_on_PLAY $(Get-Date) ===" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_rally_gauntlet_registry.json --profiles Pauper-Rally-Anchor-Value `
  --opponents "rally" --skill 7 --games-per-matchup 256 --games-per-job 16 `
  --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "branch_mirror_play" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

# Pass 2: agent ON THE DRAW
$env:EVAL_OPPONENT_ON_PLAY="1"
"=== PASS agent_on_DRAW $(Get-Date) ===" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_rally_gauntlet_registry.json --profiles Pauper-Rally-Anchor-Value `
  --opponents "rally" --skill 7 --games-per-matchup 256 --games-per-job 16 `
  --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "branch_mirror_draw" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== MIRROR PLAY/DRAW DONE $(Get-Date) ===" | Out-File $out -Append
