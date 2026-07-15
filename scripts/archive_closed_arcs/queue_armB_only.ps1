# Run full Arm B (CONST + win-replay SIL, dedup off) then eval it at matched seed.
# Plumbing already smoke-validated; Arm A (control) already done = 47.3%.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$qlog = "local-training/queue_armB.log"
"=== QUEUE Arm B only $(Get-Date) ===" | Out-File $qlog

"running full Arm B $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_armB_winreplay_sil.ps1"

"evaluating Arm B $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_armB_winreplay_sil/model.pt" `
  -Out "local-training/affinity_armB_eval_RESULT.log" -RunId "affinity_armB_eval"

"=== QUEUE Arm B DONE $(Get-Date) ===" | Out-File $qlog -Append
