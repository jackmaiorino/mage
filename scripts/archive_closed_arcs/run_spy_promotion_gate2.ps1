# SPY PROMOTION GATE v2: local process-array. Determinism holds across SEPARATE
# processes (own JVM + own GPU-service port); only intra-process --parallel breaks it.
# Per arm: restore model once, run 8 per-opponent sweeps as 4-concurrent PS jobs.
# Verification: win-arm Grixis must reproduce the narrow gate exactly (120/256).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$md  = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$out = "local-training/spy_promotion_gate2.log"
$opps = @('grixis','burn','faeries','terror','wildfire','caw','elves','rally')
function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt|cp7_eval') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
"=== SPY PROMOTION GATE v2 $(Get-Date) ===" | Out-File $out
$arms = @(
  @{ label="win";  src="E:\mage-training\backups\spy_winning_22k\model.pt";       reg="local-training/_pretest_reg_win.json"; init="1" },
  @{ label="base"; src="E:\mage-training\backups\fresh128_control\model_best.pt"; reg="local-training/_pretest_reg_cur.json"; init="0" }
)
foreach($a in $arms){
  Kill-Train; Start-Sleep -Seconds 5
  if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
  Copy-Item $a.src "$md\model.pt" -Force; Copy-Item $a.src "$md\model_latest.pt" -Force
  "=== ARM $($a.label) md5=$((Get-FileHash "$md\model.pt" -Algorithm MD5).Hash) $(Get-Date) ===" | Out-File $out -Append
  $jobs = @()
  $i = 0
  foreach($opp in $opps){
    $port = 26150 + $i*10; $mport = 27150 + $i*10; $i++
    $jobs += Start-Job -ArgumentList $opp,$port,$mport,$a.label,$a.reg,$a.init -ScriptBlock {
      param($opp,$port,$mport,$label,$reg,$init)
      Set-Location "C:\Users\Jack\IdeaProjects\mage"
      $env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
      $env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
      $env:RL_INITIATIVE_FEATURES_ENABLE=$init
      py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
        --opponents $opp --skill 1 --games-per-matchup 256 --games-per-job 16 `
        --gpu-port $port --gpu-metrics-port $mport `
        --deterministic-eval --skip-compile --replay-seed-base 5151 --run-id "spy_promo2_$($label)_$opp" 2>&1 |
        Select-String "wr=" | ForEach-Object { $_.Line }
    }
    # stagger startups: concurrent sweep SETUP collides on shared profile state
    # (onnx export/snapshot file locks); 180s lets each finish setup first
    Start-Sleep -Seconds 180
    # cap concurrency at 4
    while ((Get-Job -State Running).Count -ge 4) { Start-Sleep -Seconds 30 }
  }
  Wait-Job $jobs | Out-Null
  foreach($j in $jobs){ Receive-Job $j | Out-File $out -Append; Remove-Job $j }
  "=== $($a.label) DONE $(Get-Date) ===" | Out-File $out -Append
}
"=== SPY PROMOTION GATE v2 DONE $(Get-Date) ===" | Out-File $out -Append
