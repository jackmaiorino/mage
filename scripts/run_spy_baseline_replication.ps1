# Baseline replication: can we reproduce the ~50% Spy-vs-Grixis-skill1 number?
# Restores the canonical baseline, then runs CLEAN greedy evals (search off) across
# multiple seeds. 5151/9999 reproduce prior measurements (determinism check); the
# rest are fresh independent draws (across-seed spread). Same tool/opponent/skill/n
# as every prior baseline measurement (run_cp7_eval_sweep, grixis, skill 1, n=128).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/baseline_replication_RESULT.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak = "local-training\backups\spy_value_baseline_20260531"
$reg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"

# clean eval env (no search/MCTS/shaping; ONNX not TRT)
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\SPY_FINISH_GATE -ErrorAction SilentlyContinue
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"

# restore canonical baseline into the live profile dir (the eval snapshots from here)
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"=== baseline replication started $(Get-Date) (baseline restored from $bak) ===" | Out-File $out

$seeds = @(5151, 9999, 1234, 7777, 3030, 4242)
foreach ($s in $seeds) {
  $rid = "repl_s$s"
  "=== eval $rid (seed $s) start $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $s --run-id $rid 2>&1 |
    Select-String -Pattern "wr=|winrate|wins=" | ForEach-Object { $_.Line } | Out-File $out -Append
  "=== eval $rid done $(Get-Date) ===" | Out-File $out -Append
}
"=== ALL DONE $(Get-Date) ===" | Out-File $out -Append
