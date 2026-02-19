package mage.player.ai.rl;

/**
 * Interface for Python ML model entry point. This interface defines the methods
 * that will be called from Java to Python.
 */
public interface PythonEntryPoint {

    /**
     * Initialize the Python ML model
     */
    void initializeModel();

    /**
     * Save the current model state
     *
     * @param path Path to save the model
     */
    void saveModel(String path);

    /**
     * Load a saved model state
     *
     * @param path Path to load the model from
     */
    void loadModel(String path);

    /**
     * Atomically save a 'latest weights' file (tmp -> replace).
     *
     * @param path Path to save latest weights to
     * @return true if saved
     */
    boolean saveLatestModelAtomic(String path);

    /**
     * Reload latest weights if the file exists and is newer than the currently
     * loaded one.
     *
     * @param path Path to latest weights file
     * @return true if reloaded
     */
    boolean reloadLatestModelIfNewer(String path);

    /**
     * Calculate optimal batch size based on available GPU memory
     *
     * @return optimal number of samples per batch for current GPU
     */
    int getOptimalBatchSize();

    /**
     * Return a short diagnostic string about runtime device placement. Useful
     * for confirming CUDA vs CPU at runtime.
     */
    String getDeviceInfo();

    /**
     * Score a fixed-size padded candidate set for each state.
     * <p>
     * Return format: for each batch item, {@code maxCandidates} float32 policy
     * probabilities followed by 1 float32 value estimate.
     */
    byte[] scoreCandidatesFlat(
            byte[] sequencesBytes,
            byte[] masksBytes,
            byte[] tokenIdsBytes,
            byte[] candidateFeaturesBytes,
            byte[] candidateIdsBytes,
            byte[] candidateMaskBytes,
            int batchSize,
            int seqLen,
            int dModel,
            int maxCandidates,
            int candFeatDim
    );

    /**
     * Score candidates using a specific policy key.
     * <p>
     * policyKey: - "train" (current training model) - "snap:&lt;id&gt;"
     * (snapshot opponent policy)
     */
    byte[] scoreCandidatesPolicyFlat(
            byte[] sequencesBytes,
            byte[] masksBytes,
            byte[] tokenIdsBytes,
            byte[] candidateFeaturesBytes,
            byte[] candidateIdsBytes,
            byte[] candidateMaskBytes,
            String policyKey,
            String headId,
            int pickIndex,
            int minTargets,
            int maxTargets,
            int batchSize,
            int seqLen,
            int dModel,
            int maxCandidates,
            int candFeatDim
    );

    /**
     * Predict mulligan decision from hand features.
     *
     * @param features Mulligan feature vector (32-dim float array)
     * @return Probability of keeping the hand (0.0 = mulligan, 1.0 = keep)
     */
    float predictMulligan(float[] features);

    /**
     * Return raw two-headed mulligan scores (Q_keep, Q_mull) as little-endian
     * float32 bytes.
     *
     * @param features Mulligan feature vector (68-dim float array)
     * @return byte[8] containing 2 float32 values: [Q_keep, Q_mull]
     */
    byte[] predictMulliganScores(float[] features);

    /**
     * Train the mulligan model with a batch of training data.
     *
     * @param features Batch of mulligan features
     * @param decisions Batch of mulligan decisions (1=keep, 0=mulligan)
     * @param outcomes Batch of game outcomes (win/loss)
     * @param gameLengths Batch of game lengths in turns (int32 array) for survival reward
     * @param batchSize Size of the batch
     */
    void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes, byte[] gameLengths, byte[] earlyLandScores, byte[] overrides, int batchSize);

    /**
     * Save the mulligan model to disk.
     */
    void saveMulliganModel();

    /**
     * Train on a batch of padded candidate decision steps.
     *
     * @param chosenIndexBytes int32[batchSize] chosen candidate index
     * @param discountedReturnsBytes float32[batchSize] value target
     */
    void trainCandidatesFlat(
            byte[] sequencesBytes,
            byte[] masksBytes,
            byte[] tokenIdsBytes,
            byte[] candidateFeaturesBytes,
            byte[] candidateIdsBytes,
            byte[] candidateMaskBytes,
            byte[] chosenIndicesBytes,
            byte[] chosenCountBytes,
            byte[] oldLogpTotalBytes,
            byte[] oldValueBytes,
            byte[] discountedReturnsBytes,
            int batchSize,
            int seqLen,
            int dModel,
            int maxCandidates,
            int candFeatDim
    );

    /**
     * Train on a batch containing multiple episodes concatenated together. The
     * dones array marks terminal steps (1 at end-of-episode, else 0) so that
     * GAE/returns are computed per-episode without leaking across boundaries.
     *
     * @param donesBytes int32[batchSize] 1=end-of-episode, 0=continuation
     * @param rewardsBytes float32[batchSize] immediate rewards
     */
    void trainCandidatesMultiFlat(
            byte[] sequencesBytes,
            byte[] masksBytes,
            byte[] tokenIdsBytes,
            byte[] candidateFeaturesBytes,
            byte[] candidateIdsBytes,
            byte[] candidateMaskBytes,
            byte[] rewardsBytes,
            byte[] chosenIndicesBytes,
            byte[] chosenCountBytes,
            byte[] oldLogpTotalBytes,
            byte[] oldValueBytes,
            byte[] donesBytes,
            byte[] headIdxBytes,
            int batchSize,
            int seqLen,
            int dModel,
            int maxCandidates,
            int candFeatDim
    );

    /**
     * Get main model training statistics. Returns a map with 'train_steps'
     * (int) and 'train_samples' (int).
     */
    java.util.Map<String, Integer> getMainModelTrainingStats();

    /**
     * Get mulligan model training statistics. Returns a map with 'train_steps'
     * (int) and 'train_samples' (int).
     */
    java.util.Map<String, Integer> getMulliganModelTrainingStats();

    /**
     * Get training health statistics. Returns a map with 'gpu_oom_count' (int).
     */
    java.util.Map<String, Integer> getHealthStats();

    /**
     * Reset training health statistics counters.
     */
    void resetHealthStats();

    /**
     * Record game result for value head quality tracking and auto-GAE. This is
     * called after each game ends to track whether the value head correctly
     * predicts wins (positive) vs losses (negative).
     *
     * @param lastValuePrediction The final value prediction from the model
     * @param won True if the RL player won the game
     */
    void recordGameResult(float lastValuePrediction, boolean won);

    /**
     * Get current value head quality metrics. Returns a map with 'accuracy',
     * 'avg_win', 'avg_loss', 'samples', 'use_gae'.
     */
    java.util.Map<String, Object> getValueHeadMetrics();

    /**
     * Get auto-batching telemetry (caps/splits/headroom) for
     * Grafana/Prometheus. Returns a map with keys like: - worker, role -
     * infer_safe_max, train_safe_max_episodes - infer_mb_per_sample,
     * train_mb_per_step - free_mb, total_mb, desired_free_mb -
     * infer_splits_cap/paging/oom, train_splits_cap/paging/oom
     */
    java.util.Map<String, Object> getAutoBatchMetrics();

    /**
     * Get training loss components for Grafana/Prometheus. Returns a map with
     * keys: total_loss, policy_loss, value_loss, entropy, entropy_coef,
     * clip_frac, approx_kl, batch_size, advantage_mean
     */
    java.util.Map<String, Object> getTrainingLossMetrics();

    /**
     * Acquire GPU lock (called by Java before learner training burst). Learner
     * will hold this lock while draining the training queue.
     */
    void acquireGPULock();

    /**
     * Release GPU lock (called by Java after learner training burst).
     */
    void releaseGPULock();
}
