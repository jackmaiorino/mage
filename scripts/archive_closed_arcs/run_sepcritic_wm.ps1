# SEPARATE-CRITIC world-model aux (Codex-endorsed CONVERSION experiment).
# Prior run proved de-myopia WORKS (value-AUC up) but did NOT convert to policy
# (P(cast Spy) flat, NO_BOARD doubled = upstream shared-encoder drift). Fix: put
# the VALUE path in its own critic encoder so WM de-myopia shapes VALUE only;
# the actor encoder is untouched (no shared drift), and the improved value reaches
# the policy ONLY through terminal-reward PPO advantages (thesis-clean conversion).
# Gradient routing VERIFIED (build_separate_critic_warmstart.py): WM/value grad ->
# critic encoder, 0.0 -> policy encoder. Continue-from a WARM-STARTED model whose
# critic encoder = copy of the competent teacher policy encoder.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/sepcritic_RESULT.log"
$tlog = "local-training/sepcritic.log"; $telog = "local-training/sepcritic.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$ck   = "E:\mage-training\backups\sepcritic_wm_20260618"
$ref  = "E:\mage-training\backups\sepcritic_ref.pt"
$teacher = "E:\mage-training\backups\sepcritic_warmstart.pt"
$teacherSrc = "E:\mage-training\backups\fresh128_control\model_best.pt"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$statsCsv = "$prof\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 16 }
New-Item -ItemType Directory -Force $ck | Out-Null

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Clean-Sweep($rid) {
  foreach ($sub in 'snapshot','db') { if (Test-Path "$GL/$rid/$sub") { Remove-Item "$GL/$rid/$sub" -Recurse -Force -ErrorAction SilentlyContinue } }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'OPPONENT_SAMPLER','SEARCH_OP_ENABLE','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','RL_WORLD_MODEL_LABELS_ENABLE'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"; $env:OPPONENT_SAMPLER="grixis"
  foreach($k in 'WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','RL_WORLD_MODEL_LABELS_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill 1 --games-per-matchup 96 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base $seed --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) { $m = (py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim(); ">>> $m" | Out-File $out -Append; return $m }
function TrainWinfrac {
  if (-not (Test-Path $statsCsv)) { return "n/a" }
  $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds
  $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count)
  return "winfrac=$([Math]::Round($w,3))"
}

"=== SEPARATE-CRITIC WM AUX started $(Get-Date) ; WM_COEF=$($env:WM_COEF) KL=$($env:KL_COEF) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4

# (Re)build the warm-started separate-critic starting model + verify grad routing.
if (-not (Test-Path $teacherSrc)) { "FATAL: fresh-128 teacher missing at $teacherSrc" | Out-File $out -Append; exit 1 }
$env:VALUE_USE_SEPARATE_CRITIC_ENCODER="1"; $env:WORLD_MODEL_DIM="18"; $env:CRITIC_NUM_LAYERS="2"
$bw = (py -3.12 scripts/mtgrl/build_separate_critic_warmstart.py --teacher $teacherSrc --out $teacher --save 2>&1 | Out-String)
$bw | Out-File $out -Append
if ($bw -notmatch "GRAD ROUTING OK") { "FATAL: grad routing verification failed -- aborting." | Out-File $out -Append; exit 1 }

if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $teacher "$md\model_latest.pt" -Force; Copy-Item $teacher "$md\model.pt" -Force
Copy-Item $teacher $ref -Force
# faithful diet: restore 128 meta pins + promote all 4 (META-RL clock via dim-fix) + mirror
foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  $pin = "local-training\backups\meta_pins_pristine\$p.model_latest.pt"
  if (Test-Path $pin) { Copy-Item $pin "$prof\$p\models\model_latest.pt" -Force; Copy-Item $pin "$prof\$p\models\model.pt" -Force }
}
py -3.12 -c "
import json,os
base='Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles'
for p in ['Pauper-Spy-Combo-Value','Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value']:
    f=f'{base}/{p}/logs/league/agent_status.json'; os.makedirs(os.path.dirname(f),exist_ok=True)
    d=json.load(open(f)) if os.path.exists(f) else {}
    d['promoted']=True; d['baseline_wr']=max(0.45,d.get('baseline_wr',0)); json.dump(d,open(f,'w'),indent=2)
print('continue-from sepcritic warm-start; all promoted -> META-RL clock + mirror + CP7')
" 2>&1 | Out-File $out -Append
# baseline eval of the warm-started model (policy == teacher, so ~0.427 expected)
Eval-Now "sc_c0" 5151; Metric "sc_c0"; Clean-Sweep "sc_c0"

$wmCoef = $env:WM_COEF; if (-not $wmCoef) { $wmCoef = "0.3" }
$klCoef = $env:KL_COEF; if (-not $klCoef) { $klCoef = "0.5" }
$evalChunks = @(1,2,3,4,6,8,10,12,14,16)
$bestWr = 0.0; $lowStreak = 0
for ($c = 1; $c -le $nChunks; $c++) {
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
  $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"; $env:CANDIDATE_Q_ONLY="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','SEARCH_OP_ENABLE','SEARCH_OP_ARBITER_CAST_FILTER'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_PROMOTE_WR="0.40"; $env:LEAGUE_POST_HEURISTIC_SKILL="3"; $env:LEAGUE_MODE=""
  # THE EXPERIMENT: separate critic encoder isolates value/WM de-myopia from the policy encoder.
  $env:VALUE_USE_SEPARATE_CRITIC_ENCODER="1"; $env:CRITIC_NUM_LAYERS="2"
  $env:RL_WORLD_MODEL_LABELS_ENABLE="1"; $env:WORLD_MODEL_DIM="18"; $env:WORLD_MODEL_LOSS_COEF=$wmCoef; $env:WORLD_MODEL_DIAG="1"
  $env:REFERENCE_POLICY_KL_COEF=$klCoef; $env:MCTS_REFERENCE_MODEL_PATH=($ref -replace '\\','/')
  $env:ENTROPY_START="0.10"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="60000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"

  "=== chunk $c train start $(Get-Date) (sepcritic WM=$wmCoef KL=$klCoef) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  "chunk $c done; in-train $(TrainWinfrac) ; freeGB=$([Math]::Round((Get-PSDrive C).Free/1GB,1))" | Out-File $out -Append
  if ($evalChunks -contains $c) {
    Eval-Now "sc_c$c" 5151; $m = Metric "sc_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $castSpy = if ($m -match 'cast_spy=(\d+)%') { [int]$Matches[1] } else { 100 }
    $finAvail = if ($m -match 'finisher_avail=(\d+)%') { [int]$Matches[1] } else { 100 }
    if ($castSpy -lt 45 -or $finAvail -lt 65) {
      "=== ABORT (funnel collapse): cast_spy=$castSpy% finisher_avail=$finAvail% (baseline 60/87). best=$bestWr preserved. ===" | Out-File $out -Append; break
    }
    $csv = "$GL/sc_c$c/matchups.csv"
    if (Test-Path $csv) { $row = Import-Csv $csv | Select-Object -First 1; $wr=[double]$row.wins/[double]$row.total
      if ($wr -gt $bestWr) { $bestWr=$wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append }
      if ($wr -lt 0.35) { $lowStreak++ } else { $lowStreak = 0 }
      if ($lowStreak -ge 2) { "=== ABORT: 2 evals <0.35. best=$bestWr preserved. ===" | Out-File $out -Append; break }
    }
    Clean-Sweep "sc_c$c"
  }
}
"=== SEPARATE-CRITIC WM AUX DONE $(Get-Date) ; best_wr=$bestWr ; ckpts $ck ===" | Out-File $out -Append
