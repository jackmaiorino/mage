# DECOUPLE phase B (MIXED-MODE) + eval. Live PPO at full throughput (SEARCH_OP
# OFF, 48 runners ~3 eps/s) while candidate_q targets are REPLAYED from the
# dumped dataset to keep the shared encoder de-myopia'd. Tests whether
# execution-safe de-myopia converts to winrate at PPO volume the gate couldn't
# reach. Then eval de-myopia'd vs baseline at TWO seeds (n=256 pooled) + probes.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/optionA_decouple_mix_RESULT.log"
$tlog = "local-training/optionA_decouple_mix_train.log"
$telog= "local-training/optionA_decouple_mix_train.err"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$demy = "local-training\backups\spy_value_decouple_20260602"
$dump = "C:\Users\Jack\IdeaProjects\mage\local-training\cq_dump"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$refabs = (Resolve-Path "$bak\model_latest.pt").Path
$mins = [int]($env:MIX_MINUTES); if ($mins -le 0) { $mins = 360 }
$iters = [int]($mins / 15)

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

$nshards = (Get-ChildItem "$dump\cq_*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
"=== DECOUPLE mixed-mode started $(Get-Date) ; window ${mins}min ; replay shards=$nshards ===" | Out-File $out
if ($nshards -lt 1) { "ABORT: no npz shards in $dump" | Out-File $out -Append; exit 1 }
Kill-Train; Start-Sleep -Seconds 3

Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
"baseline restored (mixed run starts from baseline)" | Out-File $out -Append

# MIXED: live PPO full throughput (SEARCH_OP OFF) + candidate_q REPLAY de-myopia
$env:SEARCH_OP_ENABLE="0"
$env:CANDIDATE_Q_FROM_MCTS_TARGETS="1"; $env:CANDIDATE_Q_MCTS_SIGNED_TARGETS="1"
$env:CANDIDATE_Q_LOSS_COEF="4.0"; $env:CANDIDATE_Q_HUBER_BETA="0.25"; $env:CANDIDATE_Q_ONLY="0"
$env:CANDIDATE_Q_REPLAY_DIR="$dump"; $env:CANDIDATE_Q_REPLAY_EVERY="1"; $env:CANDIDATE_Q_REPLAY_STEPS="2"
$env:VALUE_USE_SEPARATE_CRITIC_ENCODER="0"; $env:VALUE_LOSS_COEF="1.0"; $env:VALUE_LOSS_COEF_WARMUP="5.0"
$env:REFERENCE_POLICY_KL_COEF="0.3"; $env:MCTS_REFERENCE_MODEL_PATH="$refabs"
Remove-Item Env:\CANDIDATE_Q_DUMP_DIR -ErrorAction SilentlyContinue
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
$env:MULTI_PLY_MCTS="0"; $env:MCTS_TRAINING_ENABLE="0"

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
"mixed-mode launched $(Get-Date)" | Out-File $out -Append

for ($i = 0; $i -lt $iters; $i++) {
  Start-Sleep -Seconds 900
  $trl = "local-training/local_pbt/trainer.log"
  $ep = ""
  if (Test-Path $trl) { $ep = (Select-String -Path $trl -Pattern "episodes:" -ErrorAction SilentlyContinue | Select-Object -Last 1).Line }
  "[$([int](($i+1)*15)) min] $ep   $(Get-Date)" | Out-File $out -Append
}

"mixed window done $(Get-Date); stopping" | Out-File $out -Append
Kill-Train; Start-Sleep -Seconds 8

if (Test-Path $demy) { Remove-Item $demy -Recurse -Force }
New-Item -ItemType Directory -Path $demy -Force | Out-Null
Copy-Item "$md\model.pt"        "$demy\model.pt"        -Force
Copy-Item "$md\model_latest.pt" "$demy\model_latest.pt" -Force
"decouple model backed up to $demy" | Out-File $out -Append

"=== weight diff vs baseline ===" | Out-File $out -Append
py -3.12 scripts/mtgrl/qhead_weight_diff.py 2>&1 | Out-File $out -Append

# --- EVAL at TWO seeds (search OFF), de-myopia'd then baseline ---
$env:SEARCH_OP_ENABLE="0"; $env:CANDIDATE_Q_ONLY="0"; $env:CANDIDATE_Q_BLEND="0"; $env:USE_TRT_INFERENCE="0"
Remove-Item Env:\CANDIDATE_Q_REPLAY_DIR -ErrorAction SilentlyContinue
Remove-Item Env:\REFERENCE_POLICY_KL_COEF -ErrorAction SilentlyContinue
Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue

function Run-Eval($rid, $seed) {
  "=== eval $rid (seed $seed) $(Get-Date) ===" | Out-File $out -Append
  py -3.12 scripts/run_cp7_eval_sweep.py `
    --registry $reg --profiles Pauper-Spy-Combo-Value --opponents grixis --skill 1 `
    --games-per-matchup 128 --parallel 4 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 |
    Select-String -Pattern "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
}

# de-myopia'd model (profile holds it now)
Run-Eval "decouple_demyopia_s5151" 5151
Run-Eval "decouple_demyopia_s9999" 9999
# baseline
Copy-Item "$bak\model.pt"        "$md\model.pt"        -Force
Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
Run-Eval "decouple_baseline_s5151" 5151
Run-Eval "decouple_baseline_s9999" 9999

# --- PROBES per seed ---
$gl = "local-training/local_pbt/cp7_eval_sweeps"
foreach ($s in @("5151","9999")) {
  $dGL = "$gl/decouple_demyopia_s$s/game_logs"
  $bGL = "$gl/decouple_baseline_s$s/game_logs"
  "=== DE-MYOPIA value-AUC seed $s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/value_auc_bootstrap.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
  "=== EXECUTION castable-Spy seed $s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/castable_spy_diagnostic.py --wm $dGL --baseline $bGL 2>&1 | Out-File $out -Append
  "=== FUNNEL demyopia seed $s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label DEMY$s $dGL 2>&1 | Out-File $out -Append
  "=== FUNNEL baseline seed $s ===" | Out-File $out -Append
  py -3.12 scripts/mtgrl/combo_funnel_diagnostic.py --label BASE$s $bGL 2>&1 | Out-File $out -Append
}
"=== DECOUPLE mixed-mode DONE $(Get-Date) ===" | Out-File $out -Append
