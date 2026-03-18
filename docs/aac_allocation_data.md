# AAC Allocation Application

## Research (Lay) Abstract

This project investigates whether deep reinforcement learning (RL) agents can discover and execute complex multi-step strategies in adversarial card games without human-provided heuristics. Using Magic: The Gathering as the research domain -- a game with the largest known state space of any popular strategy game (~10^16,000 possible game states) -- we train neural network agents via self-play to learn emergent strategic behavior.

The specific research question: can an RL agent independently discover and reliably execute a "combo" strategy (a precise sequence of 3+ cards played in a specific order to achieve an instant win) when trained only on win/loss rewards? This tests the limits of credit assignment in RL, as the agent must learn to associate early-game setup decisions with delayed game-ending payoffs across dozens of intermediate actions.

The research has implications for sequential decision-making under uncertainty, long-horizon credit assignment, and AI planning in combinatorially complex domains. The XMage open-source game engine provides a complete rules implementation, enabling reproducible experiments.

## Past Results (Developmental Allocation)

With the 50 kSU developmental allocation (Q1 2026), we:

1. **Established the training pipeline**: Built a multi-node architecture separating GPU inference/training from CPU game simulation, enabling linear scaling with compute resources.
2. **Demonstrated combo discovery**: The RL agent learned to execute the Spy Combo line (Balustrade Spy -> self-mill -> Dread Return -> Lotleth Giant) achieving 5-11% winrate from a random-initialization baseline within hours of training.
3. **Optimized resource efficiency**: Through systematic benchmarking, identified that runner-per-CPU-core ratio of 2x is optimal (3.3x throughput improvement over naive oversubscription), A100 GPUs provide identical performance to H100 at 3.2x lower cost, and a single GPU handles both inference and training for 10+ satellite nodes.
4. **Developed Population-Based Training**: Implemented PBT with automatic hyperparameter exploitation across 3 concurrent profiles, enabling efficient exploration of the training landscape.

The developmental allocation was consumed within the first two weeks of active experimentation, limiting our ability to train agents to convergence.

## Milestones

1. **Spy Combo convergence** (Q2 2026): Train the agent to >50% winrate with the Spy Combo deck via sustained PBT. Requires ~10 training runs of 3 hours each (~60,000 episodes per run, 600,000 total).
2. **Deck generalization** (Q2-Q3 2026): Extend training to 3-4 additional deck archetypes (Elves, Affinity, Rally, Wildfire) to test whether learned strategies transfer. Requires ~5 runs per deck, 4 decks = 20 runs.
3. **Meta-game evaluation** (Q3-Q4 2026): Evaluate trained agents in round-robin tournaments to measure strategic diversity and meta-game emergence. Requires ~5 evaluation runs.
4. **Scaling analysis** (ongoing): Characterize how training efficiency scales with model size (128-dim to 512-dim transformers). Requires ~5 comparative runs at larger model sizes.

## Computational Methodology

## Architecture

```
GPU Head Node (1x GPU, 16 CPUs)
  +-- Python GPU service (PyTorch transformer, batched inference + PPO training)
  +-- PBT Orchestrator (profile selection, hyperparameter exploitation)
  +-- Minimal local game runners (winrate monitoring)

CPU Satellite Nodes (10 x 32 CPUs each)
  +-- 2 JVMs per node, 32 game runners per JVM (2x CPU core count)
  +-- Connect to GPU head via TCP for inference + training data submission
  +-- Each runner simulates full MTG games, queries GPU for action selection
```

## Resource Requirements

### GPU Usage

The neural network is a small transformer (d_model=128, 2 layers, 4 heads, ~500K parameters). A single GPU handles both inference and training simultaneously:

| Metric | Measured Value |
|--------|---------------|
| GPU compute utilization (nvidia-smi) | 10% |
| GPU memory used | 1.9 GB / 80 GB (2.4%) |
| GPU power draw | 125 W / 700 W (17.8%) |
| Inference batch service time (avg) | 14 ms |
| Inference batch service time (p95) | 31 ms |
| Training batch service time (avg) | 237 ms |
| Training batch service time (p95) | 391 ms |
| Score worker duty cycle | 60% |

The H100 is overprovisioned for this workload; an A100 would perform identically at 3.2x lower billing cost. The bottleneck is Python-side data marshaling and CPU-bound game simulation, not GPU compute. **We request A100 GPUs** for cost efficiency.

### CPU Usage (Satellite Nodes)

Game simulation is CPU-bound (Java XMage game engine). Each satellite runs 2 JVMs with 32 game runner threads each, achieving ~2x CPU core oversubscription.

**Runner count optimization (measured 2026-03-17):**

| Runners/JVM | Satellites | Total Runners | Games/sec | Status |
|-------------|-----------|--------------|-----------|--------|
| 32 | 10 | 640 | **5.7** | Optimal: no timeouts |
| 100 | 20 | 4000 | ~1.75 | Thread thrashing, game kills |

Reducing runners from 100 to 32 per JVM (matching 2x CPU cores) yielded **3.3x higher game throughput** with 6x fewer total runners. The oversubscribed configuration caused CPU starvation, leading to cascading inference timeouts and killed games.

### Memory Usage

| Component | Measured |
|-----------|---------|
| GPU head node | 33.6 GB / 64 GB (52.5%) |
| CPU satellite node (per JVM) | ~4-8 GB |

### Storage

Training artifacts per run (models, logs, metrics): ~500 MB per run
Runtime bundle (Java JARs + Python): ~70 MB
Long-term model archives for PBT experiments: ~5-10 GB

## Job Performance Metrics

### Optimized Run (Job 18599480)

gpu-h100 partition, 1x H100, 10 CPU satellite nodes (32 CPUs, 32 runners/JVM each):

```
GPU Service Configuration:
  Batch max size:            1024
  Batch timeout:             25 ms
  Score worker threads:      3 (auto-scaled to match profile count)
  Train worker threads:      3 (parallel per-profile training)

Inference Pipeline:
  Score batches executed:    21,869
  Score samples processed:   565,539
  Avg inference batch size:  9.7 samples (p95: 34)
  Peak batch size:           256+ samples (during bursts)
  Inference throughput:      501 samples/sec (sustained), 806 (peak)
  Inference latency (p50):   42 ms
  Inference latency (p95):   236 ms
  Score duty cycle:          60%
  Score failures:            0

Training Pipeline:
  Train batches executed:    1,591
  Avg train batch size:      1.7 episodes, 68 steps
  Train service time (p50):  212 ms
  Train service time (p95):  391 ms
  Training backlog:          0 episodes (training keeps up with game simulation)
  Train failures:            0

Connectivity:
  Active TCP connections:    84
  Model publish operations:  139+
  Model reload operations:   139+

Episode Throughput:
  Games completed/sec:       5.7 (across 3 profiles)
  Games/hour:                ~20,500
  Games/3h run:              ~61,500
  Profiles training:         3 concurrent (Spy Combo A, Spy Combo B, Affinity)
  Winrates observed:         Spy-B 11.3%, Spy-A 4.4%, Affinity 4.3% (early training)
```

### Scaling Efficiency

| Config | CPU Cores | GPU Duty | Samples/sec | Games/sec |
|--------|----------|---------|-------------|-----------|
| Head only (16 CPUs) | 16 | 11% | 37 | ~0.3 |
| + 10 satellites (32 runners/JVM) | 336 | 60% | 501 | 5.7 |

Linear scaling from 1 to 10 satellites. GPU duty reaches 60% at 10 satellites, indicating headroom for additional CPU nodes with a larger allocation.

## GPU Justification

While GPU compute utilization is low per-sample, the GPU provides critical low-latency batched inference:

- **Without GPU**: Each game action requires ~500ms CPU inference. A 50-action game takes ~25s of inference time. Runners are inference-bound.
- **With GPU batching**: Effective per-action latency is 42ms (p50). Inference time per game drops to ~2s. Runners become CPU-bound (game simulation), enabling 10-12x higher throughput per CPU core.
- **Single GPU serves 640+ concurrent runners** across 10 nodes via TCP, amortizing GPU cost across massive parallelism.

The GPU is a throughput multiplier for the CPU fleet, not a compute bottleneck.

## Billing Analysis

| Partition | CPU Weight | Mem Weight (/GB) | GPU Weight |
|-----------|-----------|-----------------|-----------|
| gpu-h100 | 6.0 | 1.125 | 144.0/H100 |
| gpu-a100 | 1.5 | 0.375 | 48.0/A100 |
| standard | 1.0 | 0.25 | -- |

### Cost per 3-hour training run (optimal configuration)

| Component | Resources | SU Factor/hr | 3h Cost (SU) |
|-----------|----------|-------------|-------------|
| A100 GPU head | 1 A100, 16 CPU, 64G | 96 | 288 |
| 10x CPU satellites (30min, resubmitted 6x) | 10 x 32 CPU, 32G | 40 | 1,200 |
| **Total** | | | **~1,488** |

With H100 instead of A100, the GPU head costs 1,368 SU (4.75x more) for identical performance.

### Allocation Request Justification

| Allocation Size | Training Runs | Episodes | Research Capacity |
|----------------|--------------|---------|-------------------|
| 50 kSU (current) | ~33 runs | ~2M | Sufficient for single-deck convergence |
| 100 kSU (requested) | ~67 runs | ~4.1M | Multi-deck training + deck construction experiments |

RL training for complex card game strategy requires millions of episodes for policy convergence. Our current 50 kSU developmental allocation is exhausted within days of active training. A 200-550 kSU allocation would enable sustained multi-day training campaigns needed for the agent to learn complex multi-step combo sequences.

## SU Justification (Per Milestone)

### Job Types

| Job Type | Partition | GPUs | CPUs | Memory | Walltime | SU Factor/hr | SU Cost/Job |
|----------|-----------|------|------|--------|----------|-------------|-------------|
| GPU head (A100) | gpu-a100 | 1x A100 | 16 | 64G | 3h | 96 | 288 |
| CPU satellite | standard | -- | 32 | 32G | 30min | 40 | 20 |
| Evaluation run | gpu-a100 | 1x A100 | 16 | 64G | 1h | 96 | 96 |

A typical training session consists of 1 GPU head job (3h) + 10 satellite jobs resubmitted every 30 minutes (6 rounds x 10 satellites = 60 satellite jobs). Total per session: 288 + (60 x 20) = **1,488 SU (~1.5 kSU)**.

### Per-Milestone Breakdown

| Milestone | Sessions | SU/Session | Total SU | Justification |
|-----------|----------|------------|----------|---------------|
| 1. Spy Combo convergence | 10 | 1,488 | 14,880 | RL convergence requires ~600K episodes. At 20,500 eps/hr, 10 x 3h sessions = 615K episodes. |
| 2. Deck generalization (4 decks) | 20 | 1,488 | 29,760 | 5 sessions per deck x 4 decks. Each deck requires independent training from scratch. |
| 3. Meta-game evaluation | 5 | 96 | 480 | Round-robin tournaments: GPU head only (no satellites needed). |
| 4. Deck construction + emergent strategy | 5 | 1,488 | 7,440 | Joint deck-building + play optimization with extended training horizons. |
| **Total** | **40** | | **~52,560 SU** |

We request **100 kSU** to provide headroom for failed runs, debugging, parameter tuning, and occasional H100 usage when A100 nodes are unavailable (H100 costs ~4.75x more per session).

### How Estimates Were Derived

- **Episodes per session**: Measured at 5.7 episodes/sec sustained = 20,500/hr = 61,500/session (3h). See "Job Performance Metrics" section.
- **Sessions per milestone**: Based on RL training literature, policy convergence for complex strategy games requires 500K-2M episodes. Our 10-session estimate for Spy Combo (615K episodes) is conservative.
- **SU cost per session**: SU = SU_factor * walltime_hours. SU factors from Zaratan TRESBillingWeights (A100: CPU=1.5, Mem=0.375/GB, GPU=48.0). See "Billing Analysis" section.

## Disk Space Justification

| Storage Type | Amount | Justification |
|-------------|--------|---------------|
| Scratch (HPFS) | 500 GB | Each training run produces ~500 MB of artifacts (model checkpoints, training logs, game logs, PBT population snapshots). With 40 planned runs and retention of recent experiments for comparison, 500 GB provides adequate working space. |
| SHELL | 1 TB | Long-term archival of converged model weights, training curves, and evaluation results across all deck archetypes. Final models (~50 MB each) plus training histories enable reproducibility. |

## Requested Allocation

- **Compute**: 100 kSU/year (supports ~67 training sessions across 4 milestones, with H100 fallback headroom)
- **Scratch storage**: 500 GB (active training artifacts)
- **SHELL storage**: 1 TB (long-term model archives and reproducibility data)

## Efficiency Improvements Implemented

1. **Multi-node architecture**: Separates CPU-bound game simulation from GPU inference/training, allowing independent scaling
2. **Runner count optimization**: Empirically determined 2x CPU core oversubscription as optimal (3.3x throughput improvement over naive oversubscription)
3. **Batched inference**: Groups requests across profiles with 25ms timeout, achieving 9.7 avg batch size (up to 256+ during bursts)
4. **Parallel train workers**: 3 concurrent training threads with per-profile exclusion eliminate training backlog
5. **Auto-scaling score workers**: Dynamically scales inference threads to match registered profile count
6. **A100 preference**: Identified A100 as providing identical performance to H100 at 3.2x lower cost for this workload
7. **Population-Based Training**: Explores hyperparameter space efficiently by copying successful configurations and perturbing losers
