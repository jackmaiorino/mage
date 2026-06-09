# SUSTAINED faithful-diet training: can the model LEARN the orchestration (develop 3 creatures
# THEN self-mill) and break 0.52? Continue from baseline under the fixed faithful league diet
# (Gson-fixed parser holds 0.52, no diet-decay). Track the NO_BOARD rate (the dominant misplay:
# self-mill with no 3-creature board to flashback Dread Return -> deck out) as the LEADING
# metric -- if it falls, the model is learning even before winrate moves. PURE faithful diet
# (no candidate_q) to isolate whether training ALONE teaches the orchestration.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/league_train_RESULT.log"
$tlog = "local-training/league_train.log"; $telog = "local-training/league_train.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$bak  = "local-training\backups\spy_value_baseline_20260531"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 40 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 10 }
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'SEARCH_OP_ENABLE','OPPONENT_SAMPLER'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) {
  $m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim()
  ">>> $m" | Out-File $out -Append
}

"=== SUSTAINED faithful-diet TRAIN started $(Get-Date) ; ${nChunks}x${chunkMin}min ; metric=NO_BOARD ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
Copy-Item "$bak\model.pt" "$md\model.pt" -Force; Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }; Copy-Item "$bak\onnx" "$md\onnx" -Recurse -Force
# PIN: restore meta-opponent models from the latest pin set (they are a shared mutable
# resource -- trained-over across runs; rot caused the 0.53->0.29 collapse on 06-09)
$pinTs = (Get-Content "local-training\backups\meta_pins_LATEST.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
if ($pinTs) {
  foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
    $pin = "local-training\backups\meta_pins_$pinTs\$p.model_latest.pt"
    if (Test-Path $pin) {
      Copy-Item $pin "$prof\$p\models\model_latest.pt" -Force
      Copy-Item $pin "$prof\$p\models\model.pt" -Force
    }
  }
  "meta-opponents restored from pin set $pinTs" | Out-File $out -Append
} else { "WARNING: no meta pin set found -- opponents may be drifted" | Out-File $out -Append }
# re-promote all 4 profiles + seed Spy league_state so the faithful diet (META-RL + MIRROR + CP7) engages
py -3.12 -c "
import json,glob,os
base='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles'
for p in ['Pauper-Spy-Combo-Value','Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value']:
    f=f'{base}/{p}/logs/league/agent_status.json'
    if os.path.exists(f):
        d=json.load(open(f)); d['promoted']=True; d['baseline_wr']=max(0.45,d.get('baseline_wr',0)); json.dump(d,open(f,'w'),indent=2)
sp=f'{base}/Pauper-Spy-Combo-Value'
snaps=sorted(glob.glob(sp+'/models/snapshots/snapshot_step_*.pt')); keys=['snap:'+os.path.abspath(x) for x in snaps]
st={'promoted':True,'lastTickEpisode':464900,'championPolicyKey':(keys[-1] if keys else None),'recent':keys[-5:],'pool':keys,'baselineWr':{k:0.5 for k in keys[-5:]}}
json.dump(st,open(sp+'/logs/league/league_state.json','w'),indent=2)
print('promoted 4 profiles + seeded league_state')
" 2>&1 | Out-File $out -Append
Eval-Now "lt_c0" 5151; Metric "lt_c0"

$trainEnv = {
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  Remove-Item Env:\CANDIDATE_Q_LOSS_COEF -ErrorAction SilentlyContinue
  Remove-Item Env:\WORLD_MODEL_LOSS_COEF -ErrorAction SilentlyContinue
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="180000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
  $env:TRAIN_PROFILES="4"; $env:NUM_GAME_RUNNERS="64"; $env:TOTAL_EPISODES="99999999"
}
for ($c = 1; $c -le $nChunks; $c++) {
  & $trainEnv
  "=== chunk $c train start $(Get-Date) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  Eval-Now "lt_c$c" 5151; Metric "lt_c$c"
  # ABORT GATE: wr < 0.46 = collapse (diet rot or instability) -- stop, do not waste the night
  $csv = "$GL/lt_c$c/matchups.csv"
  if (Test-Path $csv) {
    $row = Import-Csv $csv | Select-Object -First 1
    $wr = [double]$row.wins / [double]$row.total
    if ($wr -lt 0.46) {
      "=== ABORT at chunk $c: wr=$wr < 0.46 (collapse gate). Restoring baseline + pinned meta. ===" | Out-File $out -Append
      Copy-Item "$bak\model_latest.pt" "$md\model_latest.pt" -Force; Copy-Item "$bak\model.pt" "$md\model.pt" -Force
      break
    }
  }
}
"=== SUSTAINED TRAIN DONE $(Get-Date) ===" | Out-File $out -Append
