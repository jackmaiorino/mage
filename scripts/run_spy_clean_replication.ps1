# CLEAN replication: confirm a freshly clean-compiled current working tree + pristine
# baseline weights still reproduce ~50% vs Grixis skill 1. Seeds chosen to pair
# seed-by-seed against this session's repl_s* runs (5151=0.484, 1234=0.430 low outlier,
# 9999=0.562 high, 4242=0.500 mid) -> if clean_s* match repl_s* within noise, nothing broke.
# Assumes `mvn ... clean compile` already ran (uses --skip-compile so the eval does NOT
# trigger an incremental recompile that could re-introduce stale-class behavior).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/clean_replication_RESULT.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak = "local-training\backups\spy_value_baseline_20260531"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"

$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"

# pristine baseline weights
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"=== CLEAN replication started $(Get-Date) (fresh clean-compile + pristine baseline) ===" | Out-File $out

$seeds = @(5151, 1234, 9999, 4242)
foreach ($s in $seeds) {
  $rid = "clean_s$s"
  "=== eval $rid (seed $s) start $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $s --run-id $rid 2>&1 |
    Select-String -Pattern "wr=|winrate|wins=" | ForEach-Object { $_.Line } | Out-File $out -Append
  "=== eval $rid done $(Get-Date) ===" | Out-File $out -Append
}
"=== ALL DONE $(Get-Date) ===" | Out-File $out -Append
