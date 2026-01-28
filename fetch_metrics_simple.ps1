# Simple metrics fetcher
$outputFile = "prometheus_metrics.txt"

Write-Host "Attempting to fetch metrics..."
Write-Host "If training is NOT running, start it first!"
Write-Host ""

# Try the metrics endpoint
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/metrics" -UseBasicParsing -TimeoutSec 5
    $response.Content | Out-File -FilePath $outputFile -Encoding UTF8
    Write-Host "SUCCESS - Metrics saved to $outputFile"
} catch {
    Write-Host "FAILED - Could not connect to http://localhost:9090/metrics"
    Write-Host ""
    Write-Host "Please start training first, then run this script again"
    Write-Host ""
    Write-Host "Or manually run: curl http://localhost:9090/metrics > prometheus_metrics.txt"
}
