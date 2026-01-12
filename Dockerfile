# syntax=docker/dockerfile:1

###########################
# 1. Build stage          #
###########################
FROM maven:3.9.6-eclipse-temurin-11 AS build

# We copy the entire multi-module Maven workspace and compile it once.
WORKDIR /workspace

# Copy source -- optimisation: keep this single layer, but use BuildKit cache for dependencies and parallel build
COPY . .

# Use Maven local repo cache so subsequent builds are fast, parallel compile on all CPUs
RUN mvn -q -T 1C -Dmaven.test.skip=true install \
    && mvn -q dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=/workspace/deps -DincludeTypes=jar

###########################
# 2. Runtime stage        #
###########################
# (Fixed base image to avoid linter issues; override with multi-stage build if needed)
FROM eclipse-temurin:17-jre-jammy AS runtime

# --- Python runtime (needed for ML bridge) ---
# Optional: install CUDA-enabled torch wheels when TORCH_CHANNEL=gpu
ARG TORCH_CHANNEL=cpu
# Install python and build essentials first (no cache mount needed here)
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python-is-python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python wheels with BuildKit pip cache (torch wheels are large)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${TORCH_CHANNEL}" = "gpu" ]; then \
    pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio; \
    else \
    pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio; \
    fi && \
    pip3 install --no-cache-dir py4j

# Create a non-root user for safety (optional but good practice)
RUN useradd -ms /bin/bash mage && mkdir -p /app && chown -R mage:mage /app
USER mage
WORKDIR /app

# --- Java artefacts ---
# Copy compiled classes, packaged jars, and the runtime dependency set produced above
COPY --chown=mage:mage --from=build /workspace/ /app/

# HEALTHCHECK to ensure RL process is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD pgrep -f RLTrainer || exit 1

# Default environment (override with `-e` at docker run)
ENV MODE=train \
    TOTAL_EPISODES=100
ENV DECKS_DIR=/app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper

# Set absolute paths for models and stats to avoid relative path issues inside the container
ENV MODEL_PATH=/app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/model.pt \
    STATS_PATH=/app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/training_stats.csv \
    EPISODE_COUNTER_PATH=/app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/episodes.txt \
    EMBEDDING_MAPPING_PATH=/app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/mapping.json

# Create the storage directory and initial mapping file to prevent startup errors
RUN mkdir -p /app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage && \
    echo "{}" > /app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/mapping.json && \
    chown -R mage:mage /app/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage

# Calculate the classpath at runtime: every compiled classes dir + all jars in target folders.
RUN printf '%s:' $(find . -type f \( -name "*.jar" -o -path "*/target/classes" \) ) > /tmp/classpath && \
    sed -i 's/:$//' /tmp/classpath

# ENTRYPOINT evaluates the dynamically generated classpath and starts RLTrainer.
ENTRYPOINT ["bash","-c","java -cp $(cat /tmp/classpath) mage.player.ai.rl.RLTrainer $MODE"] 