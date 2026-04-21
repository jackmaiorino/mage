@echo off
REM Remote GPU service, training-only (Haley's RTX 4060).
REM Inference stays on local PC via ONNX (hybrid mode).

set GPU_SERVICE_PORT=26100
set GPU_SERVICE_METRICS_PORT=27100
set GPU_SERVICE_BIND_HOST=0.0.0.0
set GPU_SERVICE_ROLE=both

set TRAIN_CUDA_DEVICE=cuda:0
set INFER_CUDA_DEVICE=cpu
set TRAIN_WORKER_THREADS=2
set SCORE_WORKER_THREADS=1

set CUDA_MEM_FRACTION=0.7
set LEARNER_BATCH_MAX_EPISODES=16
set LEARNER_BATCH_MAX_STEPS=8192

set PPO_VF_CLIP=0
set CRITIC_LR=5e-4
set ENTROPY_START=0.10
set ENTROPY_END=0.005
set ENTROPY_DECAY_STEPS=530000

set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

cd /d C:\Users\haley\mage
py -3.12 Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode\gpu_service_host.py
