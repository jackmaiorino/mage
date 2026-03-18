Faculty Advisor: (your PI -- leave blank if you're faculty)

Request Type: Renewal (upgrading from developmental 50 kSU allocation to larger 250 kSU allocation)

Past Results:
Our developmental allocation focused on building and optimizing the multi-node HPC training infrastructure. Key results:

1. Multi-node architecture: Designed and implemented a GPU head + CPU satellite architecture that separates neural network inference/training (GPU-bound) from game simulation (CPU-bound), enabling independent scaling of each resource. This required developing a custom TCP-based inference batching protocol, a PBT orchestrator, and satellite auto-discovery via shared environment files.

2. 20x throughput improvement through systematic benchmarking: Through iterative profiling on Zaratan, we improved training throughput from 0.3 episodes/sec (single-node baseline) to 5.7 episodes/sec (10 satellites). Key optimizations discovered on the cluster:
   - Runner-per-core ratio: Empirically determined that 2x CPU oversubscription is optimal. Naive oversubscription (100 runners on 16 cores) caused thread thrashing and 3.3x LOWER throughput -- a finding that required HPC-scale testing to discover.
   - Score worker auto-scaling: Identified and fixed a bottleneck where a single inference worker thread starved 640+ game runners, causing cascading timeouts and killed games.
   - Training pipeline parallelization: Implemented 3 parallel train workers with per-profile exclusion, eliminating a training backlog that grew unboundedly at scale.
   - Batch size optimization: Increased inference batch ceiling from 64 to 1024, reducing Python GIL contention and enabling burst processing of 256+ samples in 92ms.

3. GPU cost optimization: Benchmarked A100 vs H100 performance for our workload (small transformer, 128-dim, 2 layers). Both GPUs show <10% compute utilization -- the bottleneck is Python-side data marshaling, not GPU compute. A100 provides identical training throughput at 3.2x lower billing cost, informing our resource strategy for the larger allocation.

4. Early RL results: The agent learned to execute the Spy Combo line (Balustrade Spy -> self-mill -> Dread Return -> Lotleth Giant) within hours of training, reaching 5-13% winrate from random initialization. This validates that PBT can discover rare optimal minima in the policy space, but convergence to >50% requires sustained training beyond what the developmental allocation supports.

5. Projected scaling: With the pipeline optimizations discovered during this allocation, we project that additional satellite nodes (enabled by a larger allocation) can push throughput to ~19 episodes/sec, a 63x improvement over our initial single-node baseline and sufficient for convergence-scale training campaigns.

Research Title: Reinforcement Learning for Multi-Step Strategy Discovery in Adversarial Card Games

Research (Lay) Abstract:
This project investigates whether deep reinforcement learning (RL) agents can discover and execute complex multi-step strategies in adversarial card games without human-provided heuristics. We use Magic: The Gathering (MTG) as the research domain -- a game with the largest known state space of any popular strategy game, involving hidden information, stochastic elements, and combinatorially complex decision trees.

The specific research question is whether an RL agent can independently discover and reliably execute "combo" strategies -- precise sequences of 3 or more cards played in a specific order to achieve an instant win -- when trained only on win/loss outcome rewards. This tests the limits of credit assignment in reinforcement learning, as the agent must learn to associate early-game setup decisions (e.g., keeping specific cards in its opening hand) with delayed game-ending payoffs that occur dozens of intermediate actions later.

We train a transformer-based policy/value network via Proximal Policy Optimization (PPO) and Population-Based Training (PBT), using the open-source XMage game engine as a complete rules simulator. The system uses a multi-node architecture: a single GPU node runs the neural network service (batched inference and training), while multiple CPU satellite nodes run parallel game simulations. Each game simulation queries the GPU for action selection, enabling the agent to play thousands of games simultaneously and learn from the outcomes.

Preliminary results from our developmental allocation show the agent learning to execute a specific combo line (Balustrade Spy targeting self to mill the entire library, then Flashback Dread Return targeting Lotleth Giant for lethal damage) within hours of training, achieving 5-11% winrate from random initialization. However, reaching convergence (>50% winrate) and generalizing to multiple deck strategies requires sustained training campaigns of hundreds of thousands of episodes, which exceeds the developmental allocation budget.

Our longer-term research goal extends beyond fixed-deck training: once we demonstrate that the agent can reliably learn niche combo strategies from pre-built decks, we plan to give the agent the ability to construct its own decks from a card pool and discover emergent strategies through self-play. The agent must jointly optimize deck composition and play policy -- a fundamentally harder co-optimization problem that may reveal novel synergies human players have not identified. This research has implications for sequential decision-making under uncertainty, long-horizon credit assignment, co-evolutionary dynamics, and AI planning in combinatorially complex adversarial domains.

Desired Start Date: (leave blank for immediate)

Estimated RAM Per CPU Core (in GB): 2

Requested Allocation Type: Larger allocation

Requested Cluster: Zaratan
We use the gpu-a100 partition for the GPU head node and the standard partition for CPU satellite nodes. A100 GPUs provide identical performance to H100 for our small transformer model at 3.2x lower billing cost, as verified by benchmarking on the cluster.

Processor Need: Both CPU and GPU

Desired End Date: (leave blank for 1 year default)

Additional High-Performance Scratch Disk Space (TB): 0.4 (total 500 GB with the default 100 GB)

Additional SHELL (Medium-Term) Disk Space (TB): 0 (default 1 TB is sufficient)

Requested kSU: 250

Software to be used:
- Java 8+ (OpenJDK, loaded via module)
- Python 3.8+ (loaded via module)
- PyTorch 2.x (pip-installed, CUDA-enabled)
- Maven 3.x (loaded via module, for Java compilation)
- NumPy, Py4J (pip-installed Python libraries)
No special version restrictions. All software is self-installed within the runtime bundle.

SU Justification:
Our training pipeline uses two job types run concurrently:

JOB TYPE 1: GPU Head Node
- Partition: gpu-a100 (preferred) or gpu-h100 (fallback)
- Resources: 1x A100 GPU, 16 CPU cores, 64 GB RAM
- SU factor: 96/hr on A100 (GPU: 1x48 + CPU: 16x1.5 + Mem: 64x0.375), 312/hr on H100 (GPU: 1x144 + CPU: 16x6 + Mem: 64x1.125)
- Typical walltime: 3 hours
- SU cost per job: 288 SU (A100) or 936 SU (H100)

JOB TYPE 2: CPU Satellite Node
- Partition: standard
- Resources: 32 CPU cores, 32 GB RAM
- SU factor: 40/hr (32x1.0 CPU + 32x0.25 mem)
- Typical walltime: 3 hours (matches GPU head)
- SU cost per job: 120 SU
- We run 10 satellites concurrently for the duration of the session
- Satellite cost per session: 10 x 120 = 1,200 SU

COST PER SESSION: 1,488 SU with A100, 2,136 SU with H100
We plan ~140 sessions over one year (40 milestone sessions + ~100 additional for hyperparameter sweeps, model scaling, and iterative tuning at ~2-3 sessions/week). A100 nodes are heavily contended (0/19 available during our benchmarking), so we estimate ~50% H100 fallback: 70 x 1,488 + 70 x 2,136 = 253,680 SU. We request 250 kSU.

MILESTONE BREAKDOWN:

Milestone 1 - Spy Combo convergence (10 sessions): RL convergence requires ~600,000 episodes. At 5.7 eps/sec (measured), ten 3-hour sessions produce ~615,000 episodes. Cost: 14,880 SU.

Milestone 2 - Deck generalization, 4 decks (20 sessions): Independent training per archetype, 5 sessions each. Cost: 29,760 SU.

Milestone 3 - Meta-game evaluation (5 sessions): Round-robin tournaments, GPU head only, 1-hour each. Cost: 480 SU.

Milestone 4 - Deck construction and emergent strategy discovery (5 sessions): Joint deck-building + play optimization with extended training horizons. Cost: 7,440 SU.

Milestone subtotal (A100-only): 52,560 SU for 40 sessions.

SCALING TO 250 kSU:

The 52.5 kSU baseline assumes 100% A100 availability and no iteration beyond milestone minimums. We request 250 kSU to account for:

1. GPU availability: A100 nodes are heavily contended (0/19 available during our benchmarking). H100 fallback costs 1.44x more per session (2,136 vs 1,488 SU). Mixed-GPU milestone cost: ~72 kSU.

2. Additional sessions (~100 beyond milestones): Hyperparameter sweeps (~60 sessions), model scaling experiments at 2x-4x dimensions (~20 sessions), extended deck construction runs (~20 sessions). At ~1,800 SU avg: ~180 kSU.

Grand total: ~72 kSU (milestones) + ~180 kSU (additional) = ~252 kSU, supporting sustained 2-3 sessions/week over one year.

Disk Space Justification:
Scratch (requesting 0.4 TB additional, 0.5 TB total):
Each training session produces ~500 MB of artifacts: model checkpoints (action model + mulligan model per profile, ~50 MB each), training statistics CSVs, game logs, PBT population snapshots, and orchestrator status files. With 40 planned training sessions and the need to retain recent experiments for comparison and PBT exploitation (copying winning model weights to losing profiles), we need ~20 GB of active working data plus headroom for concurrent experiments. The 500 GB total provides adequate working space with room for temporary game log output during debugging sessions.

We do not request additional SHELL storage; the default 1 TB is sufficient for long-term archival of converged model weights (~50 MB per trained deck) and training curves.

Data Classification Level: Low Risk (Level 1)

Data Sensitivity Notes:
All data is synthetic game simulation output (neural network weights, training statistics, game logs). No personally identifiable information, protected data, or sensitive research data is stored or processed. All code is open-source (XMage game engine).

Code Use and Scalability:
Our system uses two codes: a Java game simulation engine (XMage, CPU-bound) and a Python/PyTorch neural network service (GPU-accelerated).

SCALING BENCHMARKS (measured on Zaratan, Job 18599480):

CPU Scaling (game simulation throughput):
  Head only (16 CPUs):        0.3 games/sec,  37 inference samples/sec, 11% GPU duty
  + 10 satellites (336 CPUs): 5.7 games/sec, 501 inference samples/sec, 60% GPU duty

Scaling from 16 to 336 CPUs (21x) yielded 19x throughput improvement, demonstrating near-linear CPU scaling. The GPU is not the bottleneck.

Runner-per-core optimization:
We empirically determined that 2x CPU core oversubscription (32 runners on 16 cores per JVM) is optimal. Higher oversubscription (100 runners on 16 cores) caused thread thrashing and 3.3x LOWER throughput despite more runners, due to CPU starvation causing cascading inference timeouts.

GPU utilization:
nvidia-smi reports 10% compute utilization and 1.9 GB / 80 GB memory usage. The transformer model (128-dim, 2 layers, ~500K parameters) is small; GPU value comes from low-latency batched inference (14ms avg, 31ms p95) serving 640+ concurrent game runners. An A100 provides identical performance to an H100 for this workload.

Inference batches average 9.7 samples (p95: 34), with burst batches up to 256 samples processed in 92ms. The GPU handles both inference and PPO training concurrently with zero training backlog.

Memory usage:
GPU head node: 33.6 GB / 64 GB (52.5%). We request 64 GB (4 GB/core on 16 cores) to accommodate the Java game engine overhead alongside the Python GPU service. Measured usage confirms this is appropriate.
CPU satellite nodes: ~4-8 GB per JVM, 2 JVMs per node. 32 GB total (1 GB/core on 32 cores) is sufficient. This is at or below the standard memory-per-core ratio.

Milestones:
1. Combo strategy convergence via PBT (Q2 2026): Use Population-Based Training to navigate the RL loss landscape and converge on a policy that reliably executes a rare, high-reward multi-step combo (Balustrade Spy -> Dread Return -> Lotleth Giant). This combo represents a very rare optimal minimum in the policy space -- the agent must learn a precise sequence of setup decisions (mulligan strategy, card ordering, self-targeting) where any deviation leads to failure. PBT's population diversity is critical for discovering these narrow optima that single-agent training would miss. Success metric: >50% winrate with consistent combo execution. Estimated: 10 training sessions, ~600,000 episodes.

2. Generalization across strategy archetypes (Q2-Q3 2026): Extend PBT training to 4 additional deck archetypes (Elves, Grixis Affinity, Mono-Red Rally, Jund Wildfire), each representing a fundamentally different strategic paradigm (aggro, synergy-based, midrange). This tests whether PBT can discover optimal minima across qualitatively different reward landscapes -- combo decks have sparse, delayed rewards while aggro decks have dense, immediate rewards. Estimated: 20 training sessions, 5 per deck.

3. Meta-game emergence and Nash equilibrium approximation (Q3-Q4 2026): Pit trained deck specialists against each other in round-robin tournaments to study emergent counter-play dynamics. In game theory terms, this measures whether PBT-trained agents approximate a Nash equilibrium in the meta-game -- whether the population converges on stable strategy distributions or exhibits cyclic dominance. Estimated: 5 evaluation sessions.

4. Autonomous deck construction and emergent strategy discovery (Q4 2026): Leveraging the proven ability to discover rare combo optima from fixed decks, give the agent the ability to construct its own decks from a card pool and discover emergent strategies through self-play. This is a joint optimization over both the strategy space (play policy) and the combinatorial deck space (~10^30 possible 60-card decks from a 500-card pool). PBT is essential here -- different population members can explore radically different deck constructions, with exploitation propagating discovered synergies across the population. Success metric: the agent discovers novel synergies or combo lines not present in human-designed decks. Estimated: 5 training sessions with extended episode horizons.
