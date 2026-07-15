# SPY ARM-2 NARROW GATE (Codex #57): trained-winning-list (22k) vs frozen old-list
# baseline, deterministic harness, Grixis skill-1, n=256, seed 5151 (dev block).
# Gate: win >= base + 8pp OR win > 55%.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$out = "local-training/spy_narrow_gate.log"
function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Kill-Train; Start-Sleep -Seconds 4
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_DIAG'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
"=== SPY NARROW GATE $(Get-Date) ===" | Out-File $out

# --- arm WIN: trained 22k model already in place; initiative features ON (trained with them) ---
$env:RL_INITIATIVE_FEATURES_ENABLE="1"
"=== ARM win md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_pretest_reg_win.json --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 256 --games-per-job 16 `
  --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "spy_gate_win" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

# --- arm BASE: frozen old-list control; initiative features OFF (trained without) ---
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "E:\mage-training\backups\fresh128_control\model_best.pt" "$md\model.pt" -Force
Copy-Item "E:\mage-training\backups\fresh128_control\model_best.pt" "$md\model_latest.pt" -Force
$env:RL_INITIATIVE_FEATURES_ENABLE="0"
"=== ARM base md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out -Append
py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_pretest_reg_cur.json --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 256 --games-per-job 16 `
  --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "spy_gate_base" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== SPY NARROW GATE DONE $(Get-Date) ===" | Out-File $out -Append
