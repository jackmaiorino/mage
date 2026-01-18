# 🚀 Mage AI Training - Production Monitoring Guide

## 📊 Complete Monitoring Stack Overview

Your **production Kubernetes deployment** now includes a comprehensive monitoring solution designed specifically for distributed AI training workloads.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mage Workers  │    │  Mage Learners  │    │  Fluent Bit     │
│   (CPU Nodes)   │    │  (GPU Nodes)    │    │  (Log Agents)   │
│                 │    │                 │    │                 │
│ • Episodes/sec  │    │ • GPU Memory    │    │ • Log Parsing   │
│ • Thread Count  │    │ • Batch Size    │    │ • Error Filter  │
│ • Error Rate    │    │ • Training Loss │    │ • Aggregation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Prometheus    │
                    │  (Metrics DB)   │
                    │                 │
                    │ • Scrapes :9090 │
                    │ • Stores 30d    │
                    │ • Alerts        │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │    Grafana      │
                    │ (Visualization) │
                    │                 │
                    │ • Dashboards    │
                    │ • Real-time     │
                    │ • Multi-cluster │
                    └─────────────────┘
```

## 🎯 Key Metrics Tracked

### **Worker Metrics** (CPU Nodes)
- `mage_episodes_completed_total` - Episodes generated per worker
- `mage_worker_episodes_total` - Per-worker episode counts
- `mage_jvm_memory_used_bytes` - JVM memory utilization
- `mage_errors_total` - Error rates per component

### **Learner Metrics** (GPU Nodes)  
- `mage_training_loss` - Current model training loss
- `mage_optimal_batch_size` - GPU memory-based batch size
- `mage_current_batch_size` - Actual batch size used
- `mage_samples_processed_total` - Training samples consumed
- `mage_gpu_memory_used_bytes` - Real-time GPU VRAM usage
- `mage_training_updates_total` - Neural network update count
- `mage_win_rate` - Model evaluation performance
- `mage_checkpoints_saved_total` - Model checkpoint frequency

### **System Metrics**
- `container_cpu_usage_seconds_total` - Pod CPU utilization
- `container_memory_usage_bytes` - Pod memory usage
- `kube_pod_container_status_restarts_total` - Pod stability
- `nvidia_gpu_memory_total_bytes` - GPU hardware capacity

## 🚨 Built-in Alerts

### **Performance Alerts**
- **LowEpisodeGenerationRate**: Workers generating <0.1 episodes/sec
- **HighGPUMemoryUsage**: GPU memory >90% (prevents OOM crashes)
- **TrainingLossDivergence**: Loss increasing >1.0 over 10min

### **Reliability Alerts**  
- **FrequentPodRestarts**: >3 restarts per hour (indicates instability)
- **PythonBridgeErrors**: Java-Python communication failures
- **CheckpointFailures**: Model saving errors

## 🚀 Deployment Instructions

### **1. Deploy Monitoring Stack**
```bash
# Deploy to your EKS cluster
kubectl apply -f k8s/monitoring-stack.yaml
kubectl apply -f k8s/monitoring-configs.yaml

# Or use the deployment script
./scripts/deploy-monitoring.sh
```

### **2. Deploy AI Training Workload**
```bash
# Deploy GPU-enabled training
kubectl apply -f k8s/gpu-deployment.yaml

# Verify metrics endpoints
kubectl get pods -l app=mage-learner -o wide
kubectl get pods -l app=mage-worker -o wide
```

### **3. Access Dashboards**
```bash
# Grafana (Main Dashboard)
kubectl port-forward svc/grafana 3000:3000 -n mage-monitoring
# Visit: http://localhost:3000 (admin/admin123)

# Prometheus (Raw Metrics)  
kubectl port-forward svc/prometheus 9090:9090 -n mage-monitoring
# Visit: http://localhost:9090
```

## 📈 Dashboard Features

### **Mage AI Training Dashboard**
- **Real-time Episode Generation**: Track worker throughput across all nodes
- **GPU Memory Optimization**: Monitor VRAM usage vs optimal batch sizes
- **Training Progress**: Loss trends, win rates, and model checkpoints
- **Resource Utilization**: CPU, memory, and GPU usage per pod
- **Error Tracking**: Component-specific error rates and types

### **Key Views**
1. **🎮 Episode Generation Rate** - Workers performance monitoring
2. **🔥 Training Loss Trends** - Model learning progress
3. **💾 GPU Memory Dashboard** - VRAM optimization tracking  
4. **⚡ System Resources** - Infrastructure utilization
5. **🚨 Error Rate Analysis** - Reliability monitoring
6. **📊 Throughput Table** - Per-pod performance comparison

## 🔧 Advanced Configuration

### **Prometheus Configuration**
- **Scrape Interval**: 15 seconds (real-time monitoring)
- **Retention**: 30 days of metrics history
- **Auto-discovery**: Kubernetes service discovery for pod targets
- **GPU Metrics**: NVIDIA GPU exporter integration

### **Custom Metrics**
All Java components expose Prometheus metrics on port `:9090/metrics`:
```java
// Record custom events
RLTrainer.metrics.recordEpisodeCompleted();
RLTrainer.metrics.recordTrainingBatch(batchSize, loss);  
RLTrainer.metrics.recordGpuMemory(used, total);
```

### **Log Aggregation**
Fluent Bit automatically:
- **Collects** logs from all `mage-*` pods
- **Parses** structured training logs (episodes, GPU, errors)
- **Filters** relevant training events
- **Exports** to Prometheus for alerting

## 🎯 Production Best Practices

### **Scaling Monitoring**
```yaml
# Prometheus for large clusters
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"  
    cpu: "4000m"

# Grafana for multiple users
replicas: 2
```

### **Alert Routing**
```yaml
# Example alert manager config
route:
  group_by: ['alertname', 'cluster']
  routes:
  - match:
      severity: critical
    receiver: 'team-pager'
  - match:
      severity: warning  
    receiver: 'team-email'
```

### **Multi-Cluster Setup**
```yaml
# Federation for multiple EKS clusters
- job_name: 'federate'
  scrape_interval: 15s
  honor_labels: true
  metrics_path: '/federate'
  params:
    'match[]':
      - '{job=~"mage-.*"}'
  static_configs:
    - targets:
      - 'prometheus-cluster-1:9090'
      - 'prometheus-cluster-2:9090'
```

## 🛠️ Troubleshooting

### **Common Issues**

**Metrics Not Appearing**
```bash
# Check pod annotations
kubectl describe pod <mage-pod-name>

# Verify metrics endpoint
kubectl port-forward <pod-name> 9090:9090
curl http://localhost:9090/metrics
```

**GPU Metrics Missing**
```bash
# Check NVIDIA device plugin
kubectl get nodes -o json | jq '.items[].status.allocatable'

# Verify GPU pod scheduling
kubectl describe pod <learner-pod> | grep nvidia.com/gpu
```

**High Memory Usage**
```bash
# Check Prometheus retention
kubectl logs prometheus-xxx -n mage-monitoring

# Reduce scrape frequency
kubectl edit configmap prometheus-config -n mage-monitoring
```

### **Performance Tuning**

**For High Episode Throughput**
- Increase worker replicas: `kubectl scale deployment mage-worker --replicas=10`
- Tune `NUM_GAME_RUNNERS` per worker pod
- Monitor `mage_episodes_completed_total` rate

**For GPU Utilization**
- Monitor `mage_optimal_batch_size` vs `mage_current_batch_size`
- Adjust `MAX_SAMPLES_PER_BATCH` based on GPU memory
- Track `nvidia_gpu_memory_used_bytes` percentage

## 📞 Support

**Dashboard Issues**: Check Grafana logs and datasource connectivity
**Missing Metrics**: Verify pod labels and Prometheus target discovery  
**Alert Fatigue**: Tune alert thresholds in `mage_rules.yml`
**Performance**: Review resource limits and retention policies

---

## 🎉 Result

With this monitoring stack, you'll have **complete visibility** into your distributed AI training pipeline:

✅ **Real-time performance tracking** across all workers and learners  
✅ **GPU memory optimization** preventing costly OOM crashes  
✅ **Proactive alerting** for performance and reliability issues  
✅ **Historical analysis** for training optimization  
✅ **Production-ready** scaling and multi-cluster support

Your AI training infrastructure is now **enterprise-grade** with comprehensive observability! 🚀