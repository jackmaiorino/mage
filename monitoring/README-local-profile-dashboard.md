# Local MTGRL Grafana Dashboard

This repo includes a profile-aware Grafana dashboard:

- Dashboard file: `monitoring/grafana/provisioning/dashboards/mage-training-meta-overview.json`
- Prometheus host config: `monitoring/prometheus-host.yml`
- CSV exporter: `scripts/export_training_profile_metrics.py`

The dashboard uses three local scrape targets:

- `host.docker.internal:9090` - `RLTrainer` metrics
- `host.docker.internal:27100` - shared GPU service metrics
- `host.docker.internal:9102` - profile CSV exporter metrics

Start the profile exporter before or after training:

```powershell
.\scripts\start_profile_metrics_exporter.ps1
```

Start the local Prometheus/Grafana stack:

```powershell
docker compose -f docker-compose-observe-local.yml up -d
```

Open Grafana at `http://localhost:3000` with `admin/admin123`, then open
`Mage AI Training / Mage Training - Meta Overview`.

If Prometheus was already running before `monitoring/prometheus-host.yml` changed,
restart it or reload its config:

```powershell
docker compose -f docker-compose-observe-local.yml restart prometheus grafana
```
