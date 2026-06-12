# Package the two tarballs a rented Linux box needs:
#   local-training/cloud/mage-src.tar.gz     (git archive of HEAD -- commit first!)
#   local-training/cloud/mage-models.tar.gz  (best/baseline checkpoints + fp32 onnx + meta pins)
# Upload both + run: tar xzf mage-src.tar.gz && bash scripts/cloud/provision_rental_box.sh --models-tarball mage-models.tar.gz
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force "local-training\cloud" | Out-Null

"=== source archive (git HEAD: $(git log --oneline -1)) ==="
git archive --format=tar.gz -o "local-training/cloud/mage-src.tar.gz" HEAD
Get-Item "local-training\cloud\mage-src.tar.gz" | Select-Object Name,Length

"=== models archive ==="
$prof = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
$onnxActive = Get-Content "local-training/backups/spy_value_baseline_20260531/onnx/.active_dir"
# Stage with the on-box directory layout
$stage = "local-training\cloud\_stage"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
foreach ($p in 'Pauper-Spy-Combo-Value','Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  New-Item -ItemType Directory -Force "$stage\$prof\$p\models" | Out-Null
}
New-Item -ItemType Directory -Force "$stage\local-training\backups\spy_value_baseline_20260531\onnx\$onnxActive" | Out-Null

# Spy: best checkpoint as the live model; baseline kept as a restorable backup
Copy-Item "local-training\backups\sustained_20260611\model_best.pt" "$stage\$prof\Pauper-Spy-Combo-Value\models\model_latest.pt"
Copy-Item "local-training\backups\sustained_20260611\model_best.pt" "$stage\$prof\Pauper-Spy-Combo-Value\models\model.pt"
Copy-Item "local-training\backups\spy_value_baseline_20260531\model_latest.pt" "$stage\local-training\backups\spy_value_baseline_20260531\model_latest.pt"
Copy-Item "local-training\backups\spy_value_baseline_20260531\model.pt" "$stage\local-training\backups\spy_value_baseline_20260531\model.pt"
# fp32 onnx for the baseline backup (runners restore from here) + the live profile
Copy-Item "local-training\backups\spy_value_baseline_20260531\onnx\$onnxActive\*" "$stage\local-training\backups\spy_value_baseline_20260531\onnx\$onnxActive\" -Recurse
Copy-Item "local-training\backups\spy_value_baseline_20260531\onnx\.active_dir" "$stage\local-training\backups\spy_value_baseline_20260531\onnx\.active_dir"
# Meta opponents (pristine pins)
foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  Copy-Item "local-training\backups\meta_pins_pristine\$p.model_latest.pt" "$stage\$prof\$p\models\model_latest.pt"
  Copy-Item "local-training\backups\meta_pins_pristine\$p.model_latest.pt" "$stage\$prof\$p\models\model.pt"
}
# Pin set marker so runner scripts' pin-restore finds a set
New-Item -ItemType Directory -Force "$stage\local-training\backups\meta_pins_cloud" | Out-Null
foreach ($p in 'Pauper-Wildfire-Value','Pauper-Rally-Anchor-Value','Pauper-Affinity-Anchor-Value') {
  Copy-Item "local-training\backups\meta_pins_pristine\$p.model_latest.pt" "$stage\local-training\backups\meta_pins_cloud\$p.model_latest.pt"
}
"cloud" | Out-File "$stage\local-training\backups\meta_pins_LATEST.txt" -Encoding ascii -NoNewline

tar -czf "local-training/cloud/mage-models.tar.gz" -C $stage .
Remove-Item $stage -Recurse -Force
Get-Item "local-training\cloud\mage-models.tar.gz" | Select-Object Name,Length
"=== DONE: upload both tarballs from local-training/cloud/ ==="
