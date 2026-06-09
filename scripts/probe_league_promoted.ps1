# PROBE 2 (~9min): after seeding promoted=true + snapshot pool + LEAGUE_PROMOTE_WR=0.40,
# does OPPONENT_SAMPLER=league now engage the real diet (META/CROSS/LOCAL-SNAP) instead of
# falling to CP7-only Stage-0? Multi-profile so cross-profile snapshot models are loaded.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$f = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\logs\stats\training_stats.csv"
$tlog = "local-training/probe_league_train.log"; $telog = "local-training/probe_league_train.err"
function Kill-Train {
  Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt')
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Kill-Train; Start-Sleep -Seconds 3
$before = (Get-Content $f | Measure-Object -Line).Lines
Write-Output "CSV rows before: $before"

$env:TRAIN_PROFILES="4"
$env:OPPONENT_SAMPLER="league"
$env:LEAGUE_PROMOTE_WR="0.40"
$env:LEAGUE_BASELINE_GAMES_PER_MATCHUP="2"
$env:NUM_GAME_RUNNERS="48"
$env:SEARCH_OP_ENABLE="0"; $env:MCTS_TRAINING_ENABLE="0"; $env:MULTI_PLY_MCTS="0"
$env:INFER_CUDA_DEVICE="cuda:1"; $env:TRAIN_CUDA_DEVICE="cuda:0"
$env:PBT_EXPLOIT_INTERVAL_MINUTES="999"

Start-Process -FilePath "py" -ArgumentList "-3.12","scripts/run_local_pbt.py" `
  -RedirectStandardOutput $tlog -RedirectStandardError $telog -WindowStyle Hidden
Start-Sleep -Seconds 540
Kill-Train; Start-Sleep -Seconds 5

$after = (Get-Content $f | Measure-Object -Line).Lines
Write-Output "CSV rows after: $after  (added $($after-$before))"
Write-Output "=== opponent_type distribution of NEW rows (league + promoted) ==="
if ($after -gt $before) {
  Get-Content $f | Select-Object -Skip $before |
    ForEach-Object { ($_ -split ',')[3] } |
    ForEach-Object { ($_ -replace '\(.*','') } |
    Group-Object | Sort-Object Count -Descending |
    ForEach-Object { "{0,8}  {1}" -f $_.Count, $_.Name }
} else { Write-Output "NO NEW ROWS -- check $telog"; Get-Content $telog -Tail 10 -ErrorAction SilentlyContinue }