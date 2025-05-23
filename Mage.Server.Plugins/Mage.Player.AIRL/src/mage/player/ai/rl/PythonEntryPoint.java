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
     * Make predictions for a batch of state sequences using direct byte array
     * conversion
     *
     * @param sequencesBytes Raw byte array of sequences [batch_size * seq_len *
     * d_model * 4]
     * @param masksBytes Raw byte array of masks [batch_size * seq_len * 4]
     * @param batchSize Number of sequences in the batch
     * @param seqLen Length of each sequence
     * @param dModel Dimension of the model
     * @return Raw byte array of predictions (action_probs followed by
     * value_scores)
     */
    byte[] predictBatchFlat(byte[] sequencesBytes, byte[] masksBytes,
            int batchSize, int seqLen, int dModel);

    /**
     * Make batch predictions using the Python ML model
     *
     * @param sequences Batch of state sequences [batch_size, seq_len, d_model]
     * @param masks Attention masks [batch_size, seq_len]
     * @param actionMasks Masks for valid actions [batch_size, num_actions]
     * @return Concatenated array of action_probs and value_scores [batch_size *
     * (num_actions + 1)]
     */
    float[] predictBatch(float[][][] sequences, float[][] masks, float[][] actionMasks);

    /**
     * Train the model with a batch of data using direct byte array conversion
     *
     * @param sequencesBytes Raw byte array of sequences [batch_size * seq_len *
     * d_model * 4]
     * @param masksBytes Raw byte array of masks [batch_size * seq_len * 4]
     * @param actionProbsBytes Raw byte array of action probabilities
     * [batch_size * num_actions * 4]
     * @param valueScoresBytes Raw byte array of value scores [batch_size * 4]
     * @param actionMasksBytes Raw byte array of action masks [batch_size *
     * num_actions * 4]
     * @param batchSize Number of sequences in the batch
     * @param seqLen Length of each sequence
     * @param dModel Dimension of the model
     * @param reward Reward signal
     */
    void trainFlat(byte[] sequencesBytes, byte[] masksBytes, byte[] actionProbsBytes,
            byte[] valueScoresBytes, byte[] actionMasksBytes,
            int batchSize, int seqLen, int dModel, float reward);

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
}
