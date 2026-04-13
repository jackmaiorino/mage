@echo off
REM Training script for remote GPU node (Haley's PC, RTX 4060)
REM Run from: C:\Users\haley\mage
REM Usage: scripts\run_remote_training.bat

setlocal

REM -- Paths --
set "PATH=C:\Users\haley\apache-maven-3.9.9\bin;C:\Program Files\Git\cmd;%PATH%"
set "JAVA_HOME=C:\Program Files\Java\jdk-23"

REM -- GPU service ports (offset from local to avoid confusion) --
set GPU_SERVICE_PORT=26200
set GPU_SERVICE_METRICS_PORT=27200

REM -- Training config --
set TRAIN_PROFILES=4
set NUM_GAME_RUNNERS=32
set TOTAL_EPISODES=5000000
set WINRATE_WINDOW=200
set PY_BATCH_TIMEOUT_MS=25
set PY_BATCH_MAX_SIZE=256
set PY_SERVICE_MODE=shared_gpu
set GAME_LOG_FREQUENCY=500
set ORCHESTRATED_RUN=1

REM -- PBT settings --
set PBT_EXPLOIT_INTERVAL=60
set PBT_MIN_EPISODES=200
set PBT_EPISODE_DELTA=100
set PBT_MIN_WINNER_GAP=0.02
set PBT_MIN_WINNER_WR=0.03
set PBT_MUTATION_PCT=0.20

REM -- GPU memory: 4060 has 8GB, leave room for ONNX inference --
set CUDA_MEM_FRACTION=0.3

REM -- ONNX hybrid mode --
set USE_TRT_INFERENCE=1

REM -- Do NOT set INFER_SERVICE_ENDPOINT (local training only) --
set INFER_SERVICE_ENDPOINT=

REM -- Entropy/PPO defaults --
set PPO_VF_CLIP=0
set CRITIC_LR=5e-4
set ENTROPY_START=0.15
set ENTROPY_END=0.02
set ENTROPY_DECAY_STEPS=500000

cd /d C:\Users\haley\mage
echo Starting training on remote node (RTX 4060)...
echo GPU service port: %GPU_SERVICE_PORT%
echo Metrics port: %GPU_SERVICE_METRICS_PORT%
echo Profiles: %TRAIN_PROFILES%, Runners: %NUM_GAME_RUNNERS%

py -3.12 scripts/run_local_pbt.py
