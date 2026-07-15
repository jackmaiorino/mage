# QUEUE: wait for the Wildfire clean-run to free the GPU, then run the constant-entropy
# Affinity experiment (paper recipe) + eval. Chains on the single local GPU.
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$q = "local-training/queue_const_entropy.log"
"=== QUEUE start $(Get-Date) -- waiting for Wildfire clean-run ===" | Out-File $q
# wait until Wildfire DONE or java absent for 2 consecutive checks
$dead = 0
for ($i=0; $i -lt 300; $i++) {
  if (Select-String -Path "local-training\wildfire_cleanrun.log" -Pattern "WILDFIRE CLEAN-RUN DONE" -Quiet -ErrorAction SilentlyContinue) { "Wildfire DONE detected" | Out-File $q -Append; break }
  $java = (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'java.exe' } | Measure-Object).Count
  if ($java -eq 0) { $dead++ } else { $dead = 0 }
  if ($dead -ge 2) { "Wildfire JVM gone x2" | Out-File $q -Append; break }
  Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 10
"=== running const-entropy experiment $(Get-Date) ===" | Out-File $q -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Jack\IdeaProjects\mage\scripts\run_affinity_const_entropy.ps1"
"=== running const-entropy eval $(Get-Date) ===" | Out-File $q -Append
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Jack\IdeaProjects\mage\scripts\run_affinity_const_entropy_eval.ps1"
"=== QUEUE done $(Get-Date) ===" | Out-File $q -Append
