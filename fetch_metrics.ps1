# Fetch Prometheus metrics for analysis
$metricsUrl = "http://localhost:9090/metrics"
$outputFile = "prometheus_metrics.txt"

Write-Host "Fetching metrics from $metricsUrl..."

try {
    $response = Invoke-WebRequest -Uri $metricsUrl -UseBasicParsing
    
    # Filter for mage-related metrics
    $mageMetrics = $response.Content -split "`n" | Where-Object { 
        $_ -match "^mage_" -or $_ -match "^# HELP mage_" -or $_ -match "^# TYPE mage_"
    }
    
    $mageMetrics | Out-File -FilePath $outputFile -Encoding UTF8
    
    Write-Host "OK - Metrics saved to $outputFile"
    Write-Host "Total lines: $($mageMetrics.Count)"
    
    # Show summary of key metrics
    Write-Host ""
    Write-Host "Key Metrics:"
    $mageMetrics | Select-String "mage_infer_time_ms|mage_train_time_ms|mage_infer_latency|mage_train_queue_depth|mage_infer_batch|mage_infer_timeouts_total|mage_infer_flushes" | ForEach-Object {
        Write-Host $_
    }
    
} catch {
    Write-Host "Error fetching metrics: $_"
    Write-Host "Make sure training is running and metrics endpoint is available at $metricsUrl"
    exit 1
}
