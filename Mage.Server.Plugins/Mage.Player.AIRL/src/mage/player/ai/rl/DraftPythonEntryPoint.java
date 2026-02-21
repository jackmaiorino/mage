package mage.player.ai.rl;

/**
 * Py4J interface for the draft model Python process.
 * Handles inference (scoring picks/construction) and training.
 */
public interface DraftPythonEntryPoint {

    /** Initialize the draft model (creates or loads checkpoint). */
    void initializeDraftModel();

    /** Save draft model to path. */
    void saveDraftModel(String path);

    /** Load draft model from path. */
    void loadDraftModel(String path);

    /**
     * Score all cards in the current pack (pick head).
     *
     * @param stateBytes      float32[seqLen * dimPerToken] little-endian
     * @param maskBytes       int32[seqLen] attention mask (0=real, 1=pad)
     * @param tokenIdBytes    int32[seqLen] card vocab IDs
     * @param candFeatBytes   float32[numCands * dimPerToken] candidate features
     * @param candIdBytes     int32[numCands] candidate vocab IDs
     * @param numCands        number of cards in pack
     * @param seqLen          state sequence length (always MAX_LEN)
     * @param dimPerToken     feature dim per token (always DraftStateBuilder.DIM_PER_TOKEN)
     * @return float32[numCands + 1] little-endian: scores[0..numCands-1], value
     */
    byte[] scoreDraftPick(
            byte[] stateBytes,
            byte[] maskBytes,
            byte[] tokenIdBytes,
            byte[] candFeatBytes,
            byte[] candIdBytes,
            int numCands,
            int seqLen,
            int dimPerToken
    );

    /**
     * Score all non-land cards in the drafted pool (construction head).
     *
     * Same byte format as scoreDraftPick, but candidates are the pool cards.
     * Returns float32[numCands + 1] (scores + value).
     */
    byte[] scoreDraftConstruction(
            byte[] stateBytes,
            byte[] maskBytes,
            byte[] tokenIdBytes,
            byte[] candFeatBytes,
            byte[] candIdBytes,
            int numCands,
            int seqLen,
            int dimPerToken
    );

    /**
     * Train the draft model on a completed episode.
     *
     * @param stateBytes       float32[numSteps * seqLen * dim] all states (picks + construction)
     * @param maskBytes        int32[numSteps * seqLen]
     * @param tokenIdBytes     int32[numSteps * seqLen]
     * @param candFeatBytes    float32[numSteps * maxCands * dim]
     * @param candIdBytes      int32[numSteps * maxCands]
     * @param candMaskBytes    int32[numSteps * maxCands] 0=real,1=masked
     * @param chosenIdxBytes   int32[numSteps] chosen candidate index per step
     * @param oldLogpBytes     float32[numSteps] log-probabilities at collection time
     * @param oldValueBytes    float32[numSteps] value estimates at collection time
     * @param rewardBytes      float32[numSteps] per-step rewards (terminal reward for last step)
     * @param doneBytes        int32[numSteps] 1 = end of episode
     * @param headIdxBytes     int32[numSteps] 0=pick_head, 1=construction_head
     * @param numSteps         total number of decision steps
     * @param seqLen           state sequence length
     * @param dimPerToken      feature dim
     * @param maxCands         max candidates per step
     */
    void trainDraftBatch(
            byte[] stateBytes,
            byte[] maskBytes,
            byte[] tokenIdBytes,
            byte[] candFeatBytes,
            byte[] candIdBytes,
            byte[] candMaskBytes,
            byte[] chosenIdxBytes,
            byte[] oldLogpBytes,
            byte[] oldValueBytes,
            byte[] rewardBytes,
            byte[] doneBytes,
            byte[] headIdxBytes,
            int numSteps,
            int seqLen,
            int dimPerToken,
            int maxCands
    );

    /** Get training statistics: train_steps, train_samples. */
    java.util.Map<String, Integer> getDraftTrainingStats();

    /** Get training loss metrics for logging. */
    java.util.Map<String, Object> getDraftLossMetrics();
}
