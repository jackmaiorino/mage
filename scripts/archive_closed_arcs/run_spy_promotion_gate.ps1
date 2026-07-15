# SPY ARM-2 PROMOTION GATE (Codex #58): full 8-deck gauntlet, det, n=256/matchup,
# seed 5151. Arm win = 22k winning-list (+initiative feats); arm base = frozen
# old-list control. Primary: uniform gauntlet win vs base; secondary: Grixis holds +8pp.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$out = "local-training/spy_promotion_gate.log"
$OPP = "grixis,burn,faeries,terror,wildfire,caw,elves,rally"
function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','WIN_REPLAY_ENABLE','SIL_LOSS_COEF','VALUE_LOSS_COEF','ATTACK_BLOCK_POLICY_LOSS_COEF','HEAD_GRAD_DIAG_EVERY','RL_INITIATIVE_DIAG'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
"=== SPY PROMOTION GATE $(Get-Date) ===" | Out-File $out
$arms = @(
  @{ label="win";  src="E:\mage-training\backups\spy_winning_22k\model.pt";        reg="local-training/_pretest_reg_win.json"; init="1" },
  @{ label="base"; src="E:\mage-training\backups\fresh128_control\model_best.pt";  reg="local-training/_pretest_reg_cur.json"; init="0" }
)
foreach($a in $arms){
  Kill-Train; Start-Sleep -Seconds 4
  if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
  Copy-Item $a.src "$md\model.pt" -Force; Copy-Item $a.src "$md\model_latest.pt" -Force
  $env:RL_INITIATIVE_FEATURES_ENABLE=$a.init
  "=== ARM $($a.label) md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $a.reg --profiles Pauper-Spy-Combo-Value `
    --opponents $OPP --skill 1 --games-per-matchup 256 --games-per-job 16 `
    --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "spy_promo_$($a.label)" 2>&1 |
    Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  "=== $($a.label) DONE $(Get-Date) ===" | Out-File $out -Append
}
"=== SPY PROMOTION GATE DONE $(Get-Date) ===" | Out-File $out -Append
