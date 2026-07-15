# Throughput + fresh-init sanity probe for the d_model=256/4-layer net, LOCAL.
# Fresh-inits the bigger net, runs ~12 min on CP7 diet, measures eps/min, restores.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out = "local-training/bignet_probe.log"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$csv = "$md\..\logs\stats\training_stats.csv"
$bak = "local-training\backups\candq_arm_20260612"

function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

"=== BIGNET THROUGHPUT PROBE started $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 4

# Fresh-init: clear the d_model=128 model so the 256 net initializes from scratch
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $md | Out-Null

# Bigger-net architecture + cheap heuristic diet (no NN opponents needed for a throughput read)
$env:MODEL_D_MODEL="256"; $env:MODEL_NUM_LAYERS="4"; $env:MODEL_NHEAD="8"
$env:OPPONENT_SAMPLER="ladder"; $env:LADDER_SKILLS="1"; $env:LADDER_MIX_LOWER_P="0.0"; $env:LEAGUE_MODE=""
foreach($k in 'CANDIDATE_Q_LOSS_COEF','CANDIDATE_Q_FROM_MCTS_TARGETS','SEARCH_OP_ENABLE','REFERENCE_POLICY_KL_COEF','MCTS_REFERENCE_MODEL_PATH','CANDIDATE_Q_DUMP_DIR'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
$env:ENTROPY_START="0.25"; $env:ENTROPY_END="0.03"; $env:ENTROPY_DECAY_STEPS="100000"
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"; $env:ONNX_EXPORT_ENABLE="1"
$env:TRAIN_PROFILES="1"; $env:NUM_GAME_RUNNERS="48"; $env:TOTAL_EPISODES="99999999"

"config: d_model=256 layers=4 nhead=8, 48 runners, ladder/CP7 diet" | Out-File $out -Append
Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" -RedirectStandardOutput "local-training/bignet_train.log" -RedirectStandardError "local-training/bignet_train.err" -WindowStyle Hidden

# warmup: wait for first episodes + confirm fresh-init worked (no shape errors)
Start-Sleep -Seconds 360
$n1 = 0; if (Test-Path $csv) { $n1 = (Get-Content $csv | Measure-Object -Line).Lines }
$t1 = Get-Date
"warmup done $(Get-Date): episodes so far=$n1" | Out-File $out -Append
$err = (Select-String -Path "local-training/bignet_train.err" -Pattern "size mismatch|shape|Error|Traceback" -ErrorAction SilentlyContinue | Select-Object -First 3)
if ($err) { "STARTUP ERRORS:" | Out-File $out -Append; $err | ForEach-Object { $_.Line.Substring(0,[Math]::Min(160,$_.Line.Length)) } | Out-File $out -Append }

# measurement window: 10 min
Start-Sleep -Seconds 600
$n2 = 0; if (Test-Path $csv) { $n2 = (Get-Content $csv | Measure-Object -Line).Lines }
$t2 = Get-Date
$mins = ($t2 - $t1).TotalMinutes
$rate = if ($mins -gt 0) { [Math]::Round(($n2 - $n1) / $mins, 1) } else { 0 }
"measure window: $($n2-$n1) episodes in $([Math]::Round($mins,1)) min => $rate eps/min" | Out-File $out -Append
"  => phase1 75k eps = $([Math]::Round(75000/$rate/60/24,1)) days ; full 425k = $([Math]::Round(425000/$rate/60/24,1)) days" | Out-File $out -Append
Kill-Train; Start-Sleep -Seconds 4

# restore small-net best so the live profile is the known-good 0.573 model
Remove-Item Env:\MODEL_D_MODEL,Env:\MODEL_NUM_LAYERS,Env:\MODEL_NHEAD -ErrorAction SilentlyContinue
if (Test-Path $md) { Remove-Item "$md\*" -Recurse -Force -ErrorAction SilentlyContinue }
Copy-Item "$bak\model_best.pt" "$md\model_latest.pt" -Force
Copy-Item "$bak\model_best.pt" "$md\model.pt" -Force
"=== restored small-net 0.573 best; PROBE DONE $(Get-Date) ===" | Out-File $out -Append
