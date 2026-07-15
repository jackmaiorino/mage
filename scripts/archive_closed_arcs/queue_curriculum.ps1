# Unattended curriculum experiment: Stage1 (slow-Rally bootstrap) -> eval -> Stage2 (real
# diet harden) -> eval. Compare Stage2 eval Rally matchup to the 47.3% control (Rally 23%).
# Decision: if Rally lifts meaningfully (e.g. 23 -> 40%+) the curriculum bootstrap works.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$qlog = "local-training/queue_curriculum.log"
"=== QUEUE curriculum $(Get-Date) ===" | Out-File $qlog

"Stage 1 (slow-Rally bootstrap) $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_curr_stage1.ps1"

"eval Stage 1 $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_curr_stage1/model.pt" `
  -Out "local-training/affinity_curr_stage1_eval_RESULT.log" -RunId "affinity_curr_stage1_eval"

"Stage 2 (real-diet harden) $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_curr_stage2.ps1"

"eval Stage 2 $(Get-Date)" | Out-File $qlog -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_curr_stage2/model.pt" `
  -Out "local-training/affinity_curr_stage2_eval_RESULT.log" -RunId "affinity_curr_stage2_eval"

"=== QUEUE curriculum DONE $(Get-Date) ===" | Out-File $qlog -Append
