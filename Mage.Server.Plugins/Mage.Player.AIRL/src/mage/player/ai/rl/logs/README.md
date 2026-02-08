# RL Training Logs

Consolidated directory for all RL training logs, organized by type.

## Directory Structure

```
logs/
├── health/              # Training health monitoring
│   └── training_health.csv
├── stats/               # Episode-level statistics
│   ├── training_stats.csv
│   ├── evaluation_stats.csv
│   └── mulligan_stats.csv
├── games/               # Detailed game logs (one file per game)
│   ├── training/        # Training game logs
│   └── evaluation/      # Evaluation game logs
└── league/              # League self-play training
    ├── events.log
    └── status.txt
```

## Log File Descriptions

### Health Monitoring

**training_health.csv** - Tracks system health metrics every 60 seconds
- `timestamp` - When the sample was taken
- `uptime_min` - Training uptime in minutes
- `games_killed` - Games terminated by health monitor (stuck/infinite loops)
- `rl_activation_failures` - Invalid action selections
- `gpu_ooms` - GPU out-of-memory errors
- `python_errors` - Python bridge errors
- `model_nans` - Model NaN output occurrences

### Episode Statistics

**training_stats.csv** - Per-episode training metrics
- Episode outcomes, rewards, model performance

**evaluation_stats.csv** - Evaluation results
- Win rates vs different opponent skill levels

**mulligan_stats.csv** - Mulligan decision tracking
- Hand quality, keep/mulligan decisions, Q-values

### Game Logs

**games/training/** - Detailed logs for training games (when enabled via `GAME_LOG_FREQUENCY`)
- Full game state at each decision point
- Candidate actions with model scores
- Action selections and outcomes

**games/evaluation/** - Detailed logs for evaluation games
- Similar format to training logs
- Used for debugging and analysis

### League Training

**league/events.log** - League training event history
- Snapshot promotions
- Policy updates
- Performance metrics

**league/status.txt** - Current league state
- Active snapshot policies
- Win rates and rankings

## Environment Variables

Control log locations and behavior:

```bash
# Base directory (default: src/mage/player/ai/rl/logs)
RL_LOGS_DIR=path/to/logs

# Individual log paths (override defaults)
HEALTH_LOG_PATH=path/to/health.csv
STATS_PATH=path/to/training_stats.csv
EVAL_STATS_PATH=path/to/evaluation_stats.csv
MULLIGAN_STATS_PATH=path/to/mulligan_stats.csv
TRAINING_GAME_LOGS_DIR=path/to/training
EVAL_GAME_LOGS_DIR=path/to/evaluation
LEAGUE_EVENTS_LOG_PATH=path/to/league/events.log
LEAGUE_STATUS_PATH=path/to/league/status.txt

# Logging frequency
HEALTH_LOG_INTERVAL_SEC=60           # Health log write interval
GAME_LOG_FREQUENCY=200               # Log every Nth game (0=disabled)

# Game health monitoring
GAME_TIMEOUT_SEC=300                 # Kill stuck games after N seconds
GAME_HEALTH_MONITOR=1                # Enable/disable health monitoring
HEALTH_REPEAT_THRESHOLD=50           # Message spam threshold
HEALTH_REPEAT_WINDOW_MS=5000         # Spam detection window
```

## Usage

### Viewing Logs

```bash
# Health monitoring
cat logs/health/training_health.csv | tail -20

# Training progress
cat logs/stats/training_stats.csv | tail -50

# Recent game logs
ls -ltr logs/games/training/ | tail -10
```

### Analyzing Health Issues

```powershell
# Check for stuck games
Import-Csv logs/health/training_health.csv | Where-Object { $_.games_killed -gt 0 }

# Check for activation failures
Import-Csv logs/health/training_health.csv | Where-Object { $_.rl_activation_failures -gt 0 }

# Check for GPU OOMs
Import-Csv logs/health/training_health.csv | Where-Object { $_.gpu_ooms -gt 0 }
```

## Maintenance

### Cleaning Old Logs

```bash
# Remove old game logs (keep last N)
cd logs/games/training
ls -t | tail -n +1000 | xargs rm

# Archive old stats
gzip logs/stats/training_stats.csv
```

### Rotating Health Logs

Health logs are appended continuously. Rotate periodically:

```bash
mv logs/health/training_health.csv logs/health/training_health_$(date +%Y%m%d).csv
```

## Migration

Logs were migrated from the old structure on 2026-02-04:
- `models/training_health.log` → `logs/health/training_health.csv` (converted to pure CSV)
- `models/training_stats.csv` → `logs/stats/training_stats.csv`
- `models/mulligan_stats.csv` → `logs/stats/mulligan_stats.csv`
- `traininggamelogs/` → `logs/games/training/`
- `evalgamelogs/` → `logs/games/evaluation/`
- `models/snapshots/league_*.log` → `logs/league/`

The migration script is available at: `migrate_logs.ps1`
