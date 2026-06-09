# PROBE 3 (~6min): with LEAGUE_DEBUG=1, read the TRUE diet from lastOpponentType +
# meta-candidate count (the CSV opponent_type mislabels league snapshot opponents as
# SELFPLAY, so it's useless). Decisive: meta-candidates>0 + lastOpponentType showing
# META-RL/CROSS/LOCAL-SNAP = diet engaged; all H-CP7/SELFPLAY = still broken.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$tlog = "local-training/probe_dbg_train.log"; $telog = "local-training/probe_dbg_train.err"
function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3
"" | Out-File "local-training/local_pbt/trainer.log" -Encoding ascii  # clear so grep is fresh

$env:TRAIN_PROFILES="4"
$env:OPPONENT_SAMPLER="league"
$env:LEAGUE_DEBUG="1"
$env:LEAGUE_PROMOTE_WR="0.40"
$env:NUM_GAME_RUNNERS="48"
$env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
$env:PBT_EXPLOIT_INTERVAL_MINUTES="999"

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
Start-Sleep -Seconds 360
Kill-Train; Start-Sleep -Seconds 5

Write-Output "=== [LEAGUE_DBG] meta-candidate resolution ==="
Get-Content "local-training/local_pbt/trainer.log" -ErrorAction SilentlyContinue | Select-String -Pattern "LEAGUE_DBG.*(meta-candidates|meta-skip|cand )" | Select-Object -First 20 | ForEach-Object { $_.Line }
Write-Output "=== [LEAGUE_DBG] actual diet (lastOpponentType distribution) ==="
Get-Content "local-training/local_pbt/trainer.log" -ErrorAction SilentlyContinue |
  Select-String -Pattern "LEAGUE_DBG.*lastOpponentType=" |
  ForEach-Object { ($_.Line -replace '.*lastOpponentType=','') -replace '\(.*','' } |
  Group-Object | Sort-Object Count -Descending | ForEach-Object { "{0,6}  {1}" -f $_.Count, $_.Name }
Write-Output "=== any DBG at all? (count) ==="
(Get-Content "local-training/local_pbt/trainer.log" -ErrorAction SilentlyContinue | Select-String -Pattern "LEAGUE_DBG").Count