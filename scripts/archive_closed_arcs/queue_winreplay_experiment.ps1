# Unattended pipeline for the win-replay paired experiment (Codex #32):
#   wait for Arm A (CONST-continue control) -> eval A -> SMOKE-GATE (verify SIL fires)
#   -> full Arm B (CONST + win-replay SIL) -> eval B.
# Aborts before the 6k Arm B if the smoke shows SIL is not firing (silent no-op guard).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$qlog = "local-training/queue_winreplay.log"
"=== QUEUE win-replay experiment $(Get-Date) ===" | Out-File $qlog

# 1. Wait for Arm A to finish.
"waiting for Arm A..." | Out-File $qlog -Append
while ($true) {
  Start-Sleep -Seconds 60
  $done = $false
  if (Test-Path "local-training/affinity_armA_constcontinue.log") {
    $txt = Get-Content "local-training/affinity_armA_constcontinue.log" -Raw -ErrorAction SilentlyContinue
    if ($txt -match "Arm A DONE") { $done = $true }
  }
  $java = (Get-Process java -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($done -and $java -eq 0) { break }
  if ($done) { Start-Sleep -Seconds 10; break }
}
"Arm A finished $(Get-Date)" | Out-File $qlog -Append

# 2. Eval Arm A.
"evaluating Arm A $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_armA_constcontinue/model.pt" `
  -Out "local-training/affinity_armA_eval_RESULT.log" -RunId "affinity_armA_eval"

# 3. SMOKE GATE: 300-ep win-replay run, verify [WIN_REPLAY] + [SIL_DIAG] fire.
"running Arm B SMOKE gate $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_armB_smoke.ps1"
$smokeOk = py -3.12 -c "
import sys,re,os
raw=open('local-training/affinity_armB_smoke.log','rb').read().decode('utf-16',errors='replace')
winrep = bool(re.search(r'\[WIN_REPLAY\].*replayed=([1-9]\d*)', raw))   # Java side: buffer replays active
# Python SIL fired: diag file with pos_adv_rows>0 (stdout not captured in local mode)
sil=False
p='local-training/_sil_smoke_diag.txt'
if os.path.exists(p):
    d=open(p).read()
    sil = bool(re.search(r'pos_adv_rows=([1-9]\d*)', d))
print('PASS' if (winrep and sil) else 'FAIL')
print('win_replay=%s sil_fired=%s' % (winrep, sil), file=sys.stderr)
"
"smoke result: $smokeOk" | Out-File $qlog -Append
if ($smokeOk -notmatch "PASS") {
  "SMOKE FAILED -- SIL not firing end-to-end. Aborting before full Arm B. Inspect affinity_armB_smoke.log." | Out-File $qlog -Append
  exit 2
}

# 4. Full Arm B.
"running full Arm B $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_armB_winreplay_sil.ps1"

# 5. Eval Arm B.
"evaluating Arm B $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_armB_winreplay_sil/model.pt" `
  -Out "local-training/affinity_armB_eval_RESULT.log" -RunId "affinity_armB_eval"

"=== QUEUE DONE $(Get-Date) ===" | Out-File $qlog -Append
