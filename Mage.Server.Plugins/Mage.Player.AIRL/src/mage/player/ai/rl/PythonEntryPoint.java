package mage.player.ai.rl;

/**
 * Interface for Python ML model entry point.
 * This interface defines the methods that will be called from Java to Python.
 */
public interface PythonEntryPoint {
    /**
     * Initialize the Python ML model
     */
    void initializeModel();

    /**
     * Make batch predictions using the Python ML model
     * @param sequences Batch of state sequences [batch_size, seq_len, d_model]
     * @param masks Attention masks [batch_size, seq_len]
     * @return Concatenated array of policy_scores and value_scores [batch_size * 2]
     */
    float[] predictBatch(float[][][] sequences, float[][] masks);

    /**
     * Train the model with a batch of data
     * @param sequences Batch of state sequences [batch_size, seq_len, d_model]
     * @param masks Attention masks [batch_size, seq_len]
     * @param policyScores Target policy scores [batch_size]
     * @param valueScores Target value scores [batch_size]
     * @param actionTypes Action type indices [batch_size]
     * @param actionCombos Action combinations [batch_size, max_actions]
     * @param reward Reward signal
     */
    void train(float[][][] sequences, float[][] masks, float[] policyScores, 
              float[] valueScores, int[] actionTypes, int[][] actionCombos, float reward);

    /**
     * Save the current model state
     * @param path Path to save the model
     */
    void saveModel(String path);

    /**
     * Load a saved model state
     * @param path Path to load the model from
     */
    void loadModel(String path);
} 