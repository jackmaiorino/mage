# CURRICULUM fresh-train of the wide net (d_model=256/4L) — bootstrap combo
# discovery against the WEAKEST opponent (CP7 skill 0), graduate to skill 1.
# Tests the representational-capacity question once the net can actually play.
# Thesis-clean: terminal-only reward, weaker opponent is still a real CP7.
# Eval ALWAYS vs skill-1 Grixis (comparable to all prior results).
# Per-eval sweep cleanup so the disk does not refill (killed the last run).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/bignet_curr_RESULT.log"
$tlog = "local-training/bignet_curr.log"; $telog = "local-training/bignet_curr.err"
$reg  = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$ck   = "local-training\backups\bignet_curr_20260615"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$statsCsv = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 120 }
$phaseAChunks = [int]($env:PHASE_A_CHUNKS); if ($phaseAChunks -le 0) { $phaseAChunks = 14 }
New-Item -ItemType Directory -Force $ck | Out-Null

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Clean-Sweep($rid) {
  # keep matchups.csv (tiny), delete the heavy snapshot + game-log dirs
  $d = "$GL/$rid"
  foreach ($sub in 'snapshot','game_logs','logs','db','results') {
    if (Test-Path "$d/$sub") { Remove-Item "$d/$sub" -Recurse -Force -ErrorAction SilentlyContinue }
  }
}
function Eval-Now($rid, $seed) {
  $save=@{}; foreach($k in 'OPPONENT_SAMPLER','LADDER_SKILLS','SEARCH_OP_ENABLE'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
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
function RepSep($rid) {
  $r = (py -3.12 scripts/mtgrl/candq_dump_analyze.py "local-training/candq_dumps_v6" "$md\model_latest.pt" 2>&1 | Select-String 'detected|policy\(best\)|Q separation' | ForEach-Object { $_.Line.Trim() }) -join " | "
  "REPSEP $rid : $r" | Out-File $out -Append
}
function TrainWinfrac {
  # in-training winfrac over the last ~2000 episodes (the leading indicator for a fresh net)
  if (-not (Test-Path $statsCsv)) { return -1 }
  $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds
  $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count)
  return [Math]::Round($w,3)
}

"=== BIGNET CURRICULUM started $(Get-Date) ; ${nChunks}x${chunkMin}min ; PhaseA(skill0)=${phaseAChunks}ch ; d_model=256/4L ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null

$evalChunks = @(2,4,6,8,10,12,14,16,18,21,24,28,32,38,44,50,56,62,70,80,90,100,110,120)
$bestWr = 0.0
$dimChecked = $false
for ($c = 1; $c -le $nChunks; $c++) {
  # --- per-chunk env (skill changes at the phase boundary) ---
  $trainSkill = if ($c -le $phaseAChunks) { "0" } else { "1" }
  $env:MODEL_D_MODEL="256"; $env:MODEL_NUM_LAYERS="4"; $env:MODEL_NHEAD="8"; $env:MODEL_DIM_FEEDFORWARD="1024"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','CANDIDATE_Q_MCTS_SIGNED_TARGETS','CANDIDATE_Q_BLEND','CANDIDATE_Q_DUMP_DIR','CANDIDATE_Q_DETACH_ENCODER','SEARCH_OP_APPLY_OVERRIDE','SEARCH_OP_ARBITER_CAST_FILTER','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  $env:OPPONENT_SAMPLER="ladder"; $env:LADDER_SKILLS=$trainSkill; $env:LADDER_MIX_LOWER_P="0.0"; $env:LEAGUE_MODE=""
  $env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"

  "=== chunk $c train start $(Get-Date) (train-skill=$trainSkill) ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8

  # disk guard: bail cleanly if free space gets dangerously low
  $freeGB = [Math]::Round((Get-PSDrive C).Free/1GB,1)
  "chunk $c done; in-train winfrac(skill$trainSkill)=$(TrainWinfrac) ; freeGB=$freeGB" | Out-File $out -Append
  if ($freeGB -lt 5) { "=== ABORT: disk <5GB free, stopping to protect the machine ===" | Out-File $out -Append; break }

  if (-not $dimChecked -and (Test-Path "$md\model_latest.pt")) {
    $dim = (py -3.12 -c "import torch; d=torch.load(r'$md/model_latest.pt',map_location='cpu',weights_only=False)['state_dict']; print(d['cls_token'].shape[-1])" 2>&1 | Select-Object -Last 1)
    "DIM CHECK: d_model=$dim (expect 256)" | Out-File $out -Append; $dimChecked = $true
  }
  if ($evalChunks -contains $c) {
    Eval-Now "bc_c$c" 5151; Metric "bc_c$c"; RepSep "bc_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/bc_c$c/matchups.csv"
    if (Test-Path $csv) {
      $row = Import-Csv $csv | Select-Object -First 1
      $wr = [double]$row.wins / [double]$row.total
      if ($wr -gt $bestWr) { $bestWr = $wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append }
    }
    Clean-Sweep "bc_c$c"
  }
}
"=== BIGNET CURRICULUM DONE $(Get-Date) ; best_wr=$bestWr ; ckpts in $ck ===" | Out-File $out -Append
