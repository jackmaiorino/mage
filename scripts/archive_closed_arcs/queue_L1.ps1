# L1 unattended gated pipeline (Codex #38): up to 10 chunks of 10k-ep uniform clean training
# from ref C, eval each on the uniform 8-deck gauntlet, apply abort gates. Tests whether
# Affinity-vs-Rally CLIMBS with a long clean run (undertrained) or stays flat (wall).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$q = "local-training/queue_L1.log"
"=== QUEUE L1 long clean run $(Get-Date) ===" | Out-File $q
for ($ck = 1; $ck -le 10; $ck++) {
  "--- L1 chunk $ck (=$($ck*10)k eps) train $(Get-Date) ---" | Out-File $q -Append
  $env:L1_CHUNK = "$ck"
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_L1_chunk.ps1"
  "--- L1 chunk $ck eval (uniform gauntlet) $(Get-Date) ---" | Out-File $q -Append
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/run_affinity_arm_eval.ps1" `
    -Backup "E:/mage-training/backups/affinity_L1_ck$ck/model.pt" `
    -Out "local-training/affinity_L1_ck${ck}_eval_RESULT.log" -RunId "affinity_L1_ck${ck}_eval"
  $decision = (py -3.12 scripts/mtgrl/l1_gate.py "local-training/affinity_L1_ck*_eval_RESULT.log" $ck 2>>$q)
  "GATE after ck$ck = $decision" | Out-File $q -Append
  if ($decision -match "STOP") { "L1 STOPPING: $decision" | Out-File $q -Append; break }
}
"=== QUEUE L1 DONE $(Get-Date) ===" | Out-File $q -Append
