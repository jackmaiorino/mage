# Fetch last known values from Prometheus
$prometheusUrl = "http://localhost:9091"
$outputFile = "historical_metrics.txt"

Write-Host "Fetching last known metric values from Prometheus..."

$metrics = @(
    "mage_infer_time_ms",
    "mage_train_time_ms", 
    "mage_mulligan_time_ms",
    "mage_infer_batch_avg_size",
    "mage_train_queue_depth",
    "mage_infer_timeouts_total",
    "mage_infer_flushes_full_total",
    "mage_infer_flushes_timeout_total",
    "mage_gpu_memory_used_bytes",
    "mage_gpu_memory_total_bytes",
    "mage_infer_latency_avg_ms"
)

$results = @()

foreach ($metric in $metrics) {
    try {
        $url = "$prometheusUrl/api/v1/query?query=$metric"
        $response = Invoke-RestMethod -Uri $url -Method Get
        
        if ($response.data.result.Count -gt 0) {
            $value = $response.data.result[0].value[1]
            $timestamp = $response.data.result[0].value[0]
            $results += "$metric = $value (at $timestamp)"
            Write-Host "$metric = $value"
        } else {
            Write-Host "$metric = NO DATA"
        }
    } catch {
        Write-Host "$metric = ERROR: $_"
    }
}

$results | Out-File -FilePath $outputFile -Encoding UTF8
Write-Host ""
Write-Host "Results saved to $outputFile"
