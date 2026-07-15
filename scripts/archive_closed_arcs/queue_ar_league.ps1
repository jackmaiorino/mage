# A/R league first gate (Codex #40): train league to ~15k -> eval Affinity on the UNIFORM
# fixed CP gauntlet. Apply gate: Rally >=8pp over ref C (20%) & uniform flat/up -> continue;
# else accept scoped floor. (Eval also reports the Rally slice = the key metric.)
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$q = "local-training/queue_ar_league.log"
"=== QUEUE A/R league first-gate $(Get-Date) ===" | Out-File $q
"training A/R league to 15k $(Get-Date)" | Out-File $q -Append
$env:LEAGUE_SMOKE = "15000"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_ar_league.ps1"
"eval Affinity (league@15k) on uniform gauntlet $(Get-Date)" | Out-File $q -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
  -Backup "E:/mage-training/backups/affinity_ar_league/model.pt" `
  -Out "local-training/affinity_ar_league_eval_RESULT.log" -RunId "affinity_ar_league_eval"
"=== QUEUE A/R league DONE $(Get-Date) ===" | Out-File $q -Append
