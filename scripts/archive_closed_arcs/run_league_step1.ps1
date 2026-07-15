# LEAGUE STEP 1 (Codex #15, cheap config-only): continue-train the competent full-winning
# model against a STRONGER opponent regime -- CP clock skill 3->7, clock-weighted sampling,
# snapshot ring -- to test whether tougher sparring lifts ROBUSTNESS vs competent aggression.
# Baseline to beat: full-winning model vs CP-Grixis skill-5 = 0.269 (mirror skill-sweep).
# HONEST: this makes bad play LOSE; it will NOT by itself make the rare combo appear (that's
# Step 2 = search/demo discovery). Anti-collapse anchor = the external strong clock.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/league_step1_RESULT.log"
$tlog = "local-training/league_step1.log"; $telog = "local-training/league_step1.err"
$baseReg = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json"
$reg  = "local-training/_league_step1_registry.json"
$deck = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Winning.dek"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$prof = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles"
$ck   = "E:\mage-training\backups\league_step1_20260620"
$cont = "E:\mage-training\backups\brew_winning_20260619\model_best.pt"
$GL   = "local-training/local_pbt/cp7_eval_sweeps"
$statsCsv = "$prof\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$evalSkill = [int]($env:EVAL_SKILL); if ($evalSkill -le 0) { $evalSkill = 5 }
$chunkMin = [int]($env:CHUNK_MIN); if ($chunkMin -le 0) { $chunkMin = 45 }
$nChunks  = [int]($env:NUM_CHUNKS); if ($nChunks  -le 0) { $nChunks  = 40 }
New-Item -ItemType Directory -Force $ck | Out-Null

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
function Clean-Sweep($rid) { foreach ($sub in 'snapshot','db','results') { if (Test-Path "$GL/$rid/$sub") { Remove-Item "$GL/$rid/$sub" -Recurse -Force -ErrorAction SilentlyContinue } } }
function Eval-Now($rid) {
  $save=@{}; foreach($k in 'OPPONENT_SAMPLER','SEARCH_OP_ENABLE','LEAGUE_DEBUG'){ if(Test-Path "Env:\$k"){$save[$k]=(Get-Item "Env:\$k").Value} }
  $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"; $env:OPPONENT_SAMPLER="grixis"; Remove-Item Env:\LEAGUE_DEBUG -ErrorAction SilentlyContinue
  py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
    --opponents grixis --skill $evalSkill --games-per-matchup 128 --parallel 8 --eval-game-logging --replay-metadata `
    --skip-compile --replay-seed-base 5151 --run-id $rid 2>&1 | Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append
  foreach($k in $save.Keys){ Set-Item "Env:\$k" $save[$k] }
}
function Metric($rid) { ">>> $((py -3.12 scripts/mtgrl/orchestration_metric.py "$GL/$rid" --label $rid 2>&1 | Out-String).Trim())" | Out-File $out -Append }
function WinMode($rid) {
  $w = (py -3.12 scripts/mtgrl/win_mode.py "$GL/$rid" 2>&1 | Select-String "WR=|COMBO|BEATDOWN" | ForEach-Object { $_.Line.Trim() }) -join " | "
  "WINMODE $rid : $w" | Out-File $out -Append
}
function TrainWinfrac { if (-not (Test-Path $statsCsv)) { return "n/a" }; $rows = Get-Content $statsCsv -Tail 2000 | ConvertFrom-Csv -Header episode,turns,final_reward,opponent_type,winrate,episode_seconds; $w = ($rows | Where-Object {[double]$_.final_reward -gt 0}).Count / [Math]::Max(1,$rows.Count); return "winfrac=$([Math]::Round($w,3))" }

"=== LEAGUE STEP1 (continue full-winning, clock skill=7, eval vs grixis skill=$evalSkill) started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4
if (-not (Test-Path $cont)) { "FATAL: continue-from model missing $cont" | Out-File $out -Append; exit 1 }
# registry: Spy agent deck = Spy Winning
py -3.12 -c "
import json
reg=json.load(open(r'$baseReg'))
for e in reg:
    if e.get('profile')=='Pauper-Spy-Combo-Value': e.setdefault('train_env',{})['RL_AGENT_DECK_LIST']=r'$deck'
json.dump(reg,open(r'$reg','w'),indent=2)
print('league-step1 registry: agent deck -> Spy Winning')
" 2>&1 | Out-File $out -Append
$env:REGISTRY_PATH = (Resolve-Path $reg).Path
# CONTINUE-FROM: seed the profile model with the competent full-winning model (NOT fresh init)
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
New-Item -ItemType Directory -Force $md | Out-Null
Copy-Item $cont "$md\model_latest.pt" -Force; Copy-Item $cont "$md\model.pt" -Force
# faithful diet: meta pins + promote all 4 (META-RL clock + mirror)
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
print('all 4 promoted -> META-RL + mirror + strong CP clock')
" 2>&1 | Out-File $out -Append

$evalChunks = @(2,4,6,8,12,16,20,26,32,40)
$bestWr = 0.0
for ($c = 1; $c -le $nChunks; $c++) {
  $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
  $env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
  foreach($k in 'CANDIDATE_Q_LOSS_COEF','WORLD_MODEL_LOSS_COEF','REFERENCE_POLICY_KL_COEF','VALUE_USE_SEPARATE_CRITIC_ENCODER','RL_WORLD_MODEL_LABELS_ENABLE','WORLD_MODEL_DIM','RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  $env:CANDIDATE_Q_ONLY="0"
  # STRONGER opponent regime
  $env:OPPONENT_SAMPLER="league"; $env:LEAGUE_MODE=""
  $env:LEAGUE_POST_HEURISTIC_SKILL="5"          # clock 3 -> 5 (skill 7 = ~7x slower / 0.4 eps/s, impractical; skill5 ~= skill7 difficulty)
  $env:LEAGUE_POST_HEURISTIC_P="0.45"; $env:LEAGUE_POST_LOCAL_P="0.20"; $env:LEAGUE_POST_CROSS_P="0.35"  # ~45% clocks (throughput) / 20% mirror / 35% snapshots
  $env:LEAGUE_POOL_MAX="8"; $env:LEAGUE_PROMOTE_WR="0.55"                                  # snapshot ring + gated promotion
  $env:GAME_LOG_FREQUENCY="2000"                # minimize training game-log disk over a long run
  $env:ENTROPY_START="0.15"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="80000"     # continue-from: moderate exploration
  $env:ONNX_BATCH_TIMEOUT_MS="25"; $env:ONNX_BATCH_TIMEOUT_MAX_MS="50"
  $env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
  $env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"
  $env:REGISTRY_PATH = (Resolve-Path $reg).Path

  # DISK HYGIENE: clean prior training game logs + old ONNX staging; abort if disk too low (mid-chunk disk-full killed the prior run)
  Remove-Item "$prof\Pauper-Spy-Combo-Value\logs\games\training\*" -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem "$md\onnx" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -Skip 2 | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  $preFree = [Math]::Round((Get-PSDrive C).Free/1GB,1)
  if ($preFree -lt 6) { "=== ABORT (pre-chunk): disk ${preFree}GB < 6 ===" | Out-File $out -Append; break }
  "=== chunk $c train start $(Get-Date) ; preFreeGB=$preFree ===" | Out-File $out -Append
  Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
  Start-Sleep -Seconds ($chunkMin * 60)
  Kill-Train; Start-Sleep -Seconds 8
  $freeGB = [Math]::Round((Get-PSDrive C).Free/1GB,1)
  "chunk $c done; in-train $(TrainWinfrac) ; freeGB=$freeGB" | Out-File $out -Append
  if ($freeGB -lt 5) { "=== ABORT: disk <5GB ===" | Out-File $out -Append; break }
  if ($evalChunks -contains $c) {
    Eval-Now "ls_c$c"; Metric "ls_c$c"; WinMode "ls_c$c"
    Copy-Item "$md\model_latest.pt" "$ck\model_after_c$c.pt" -Force
    $csv = "$GL/ls_c$c/matchups.csv"
    if (Test-Path $csv) { $row = Import-Csv $csv | Select-Object -First 1; $wr = [double]$row.wins/[double]$row.total
      if ($wr -gt $bestWr) { $bestWr=$wr; Copy-Item "$md\model_latest.pt" "$ck\model_best.pt" -Force; "new best wr=$wr at chunk $c" | Out-File $out -Append } }
    Clean-Sweep "ls_c$c"
  }
}
"=== LEAGUE STEP1 DONE $(Get-Date) ; best_wr=$bestWr ; ckpts $ck ===" | Out-File $out -Append
