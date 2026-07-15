# SPY 60k GATE (Codex #59): full gauntlet, det, n=256/matchup, game logging for
# win-path decomposition. Gate vs 22k (uniform 47.0%): >=50% OR +3pp, Grixis holds,
# elves/terror not worse.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/spy_60k_gate.log"
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_DIAG'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:RL_INITIATIVE_FEATURES_ENABLE="1"
$md = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
"=== SPY 60K GATE md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out
py -3.12 scripts/run_cp7_eval_sweep.py --registry local-training/_pretest_reg_win.json --profiles Pauper-Spy-Combo-Value `
  --opponents "grixis,burn,faeries,terror,wildfire,caw,elves,rally" --skill 1 --games-per-matchup 256 --games-per-job 16 `
  --deterministic-eval --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "spy_60k_gate" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
"=== SPY 60K GATE DONE $(Get-Date) ===" | Out-File $out -Append
