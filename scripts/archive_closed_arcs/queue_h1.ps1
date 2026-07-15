# H1 unattended: hard-weighted train from ref C -> eval on UNIFORM 8-deck gauntlet (matched seed).
# Then apply Codex decision rule: Rally +>=8pp vs ref C, uniform mean flat/up, easy regression <=5pp.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$q = "local-training/queue_h1.log"
"=== QUEUE H1 $(Get-Date) ===" | Out-File $q
"training H1 $(Get-Date)" | Out-File $q -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_h1.ps1"
"eval H1 on uniform gauntlet $(Get-Date)" | Out-File $q -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_h1/model.pt" `
  -Out "local-training/affinity_h1_eval_RESULT.log" -RunId "affinity_h1_eval"
"=== QUEUE H1 DONE $(Get-Date) ===" | Out-File $q -Append
