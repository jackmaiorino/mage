# Curriculum v2: single MIXED-DIFFICULTY red-gradient run (drastic/medium/real + maintenance)
# from ref C 47.5%, then eval vs the REAL gauntlet. Compare real-Rally to the 18-23% baseline.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$qlog = "local-training/queue_curriculum_v2.log"
"=== QUEUE curriculum v2 (mixed red gradient) $(Get-Date) ===" | Out-File $qlog

"training mixed-gradient run $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_curr_stage1.ps1"

"eval vs real gauntlet $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_curr_stage1/model.pt" `
  -Out "local-training/affinity_curr_v2_eval_RESULT.log" -RunId "affinity_curr_v2_eval"

"=== QUEUE curriculum v2 DONE $(Get-Date) ===" | Out-File $qlog -Append
