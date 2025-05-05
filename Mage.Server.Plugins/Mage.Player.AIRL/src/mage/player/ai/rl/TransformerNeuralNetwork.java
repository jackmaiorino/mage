package mage.player.ai.rl;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SelfAttentionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Transformer‑based RL network, Java‑8‑compatible.  Consumes the sequence +
 * mask representation produced by {@link StateSequenceBuilder} and outputs a
 * single flat logits vector sized {@code numActions}.  Easily extendable to
 * multi‑head (CAST / ATTACK / BLOCK) by adding more vertices.
 */
public final class TransformerNeuralNetwork {

    public enum AskType { CAST, ATTACK, BLOCK }

    private static final Logger log = Logger.getLogger(TransformerNeuralNetwork.class);

    private final Random rnd = new Random();
    private double epsilon;
    private final double initialEpsilon;
    private final double finalEpsilon;
    private final int decaySteps;
    private int currentStep = 0;
    private final ComputationGraph net;
    private final int maxLen;  // Store maxLen as a class field
    private final int d;
    private final int[] headSizes;
    private final int numLayers;  // Store number of transformer layers

    // output layer names – must match builder below
    private static final String CAST_OUT  = "castOut";
    private static final String ATK_OUT   = "atkOut";
    private static final String BLK_OUT   = "blkPtr";


    public TransformerNeuralNetwork(int d, int maxLen, int[] headSizes, double initialEpsilon) {
        this.d = d;
        this.maxLen = maxLen;
        this.initialEpsilon = initialEpsilon;
        this.epsilon = initialEpsilon;
        this.finalEpsilon = 0.01;  // Final epsilon value
        this.decaySteps = 1000000;  // Number of steps to decay over
        this.headSizes = headSizes;
        this.numLayers = 3;  // Set number of transformer layers

        // Increase feature dimension from 128 to 256
        int enhancedD = 256;
        
        // Increase number of attention heads from 8 to 16
        int numHeads = 16;
        
        // Increase head size from 32 to 64
        int headSize = 64;

        // Create the network
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input_sequence")
                .setInputTypes(InputType.recurrent(d, maxLen));

        // Add multiple transformer layers
        String lastLayer = "input_sequence";
        for (int i = 0; i < numLayers; i++) {
            String layerName = "transformer_" + i;
            builder.addLayer(layerName,
                    new SelfAttentionLayer.Builder()
                            .nIn(maxLen)
                            .nOut(maxLen)
                            .nHeads(numHeads)
                            .headSize(headSize)
                            .projectInput(true)
                            .build(),
                    lastLayer);
            lastLayer = layerName;
        }

        // Add global pooling to reduce sequence dimension
        builder.addLayer("pool",
                new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build(),
                lastLayer);

        // Enhanced feed-forward network with more layers and larger dimensions
        builder.addLayer("ff1",
                new DenseLayer.Builder()
                        .nIn(d)  // Input size matches the feature dimension
                        .nOut(d * 2)
                        .activation(Activation.RELU)
                        .build(),
                "pool")
                .addLayer("ff2",
                        new DenseLayer.Builder()
                                .nIn(d * 2)
                                .nOut(d)
                                .activation(Activation.RELU)
                                .build(),
                        "ff1");

        // Output heads with increased capacity - using OutputLayer instead of DenseLayer
        builder.addLayer("cast_head",
                new OutputLayer.Builder()
                        .nIn(d)  // Input size matches the feature dimension
                        .nOut(headSizes[0])
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build(),
                "ff2")
                .addLayer("atk_head",
                        new OutputLayer.Builder()
                                .nIn(d)
                                .nOut(headSizes[1])
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "ff2")
                .addLayer("blk_head",
                        new OutputLayer.Builder()
                                .nIn(d)
                                .nOut(headSizes[2])
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "ff2")
                .setOutputs("cast_head", "atk_head", "blk_head");

        net = new ComputationGraph(builder.build());
        net.init();
    }

    // -------------------------------------------------------------------------
    // ε‑GREEDY PREDICT
    // -------------------------------------------------------------------------
    /**
     * Update epsilon value based on current step
     */
    public void updateEpsilon() {
        if (currentStep < decaySteps) {
            // Linear decay
            epsilon = initialEpsilon - (initialEpsilon - finalEpsilon) * ((double) currentStep / decaySteps);
            currentStep++;
            
            // Log epsilon every 1000 steps
            if (currentStep % 1000 == 0) {
                RLTrainer.threadLocalLogger.get().info("Epsilon decay: step=" + currentStep + ", epsilon=" + epsilon);
            }
        } else {
            epsilon = finalEpsilon;
        }
    }

    // -------------------------------------------------------------------------
    // ε‑GREEDY PREDICT
    // -------------------------------------------------------------------------
    /**
     * ε‑greedy inference for a single state.
     */
    public INDArray predict(StateSequenceBuilder.SequenceOutput s,
                            AskType ask, boolean exploration) {
        if (s == null) {
            throw new IllegalArgumentException("SequenceOutput cannot be null");
        }

        // Update epsilon if we're in exploration mode
        if (exploration) {
            updateEpsilon();
        }

        // ε-greedy branch ----------------------------------------------------
        if (exploration && rnd.nextDouble() < epsilon) {
            RLTrainer.threadLocalLogger.get().info("Random exploration with epsilon: " + epsilon);
            return randomOutput(ask);
        }

        // Ensure input shape is correct
        INDArray seq = s.sequence;
        INDArray mask = s.mask;
        
        // Log the actual shapes for debugging
        RLTrainer.threadLocalLogger.get().info("Input sequence shape: " + Arrays.toString(seq.shape()));
        RLTrainer.threadLocalLogger.get().info("Input mask shape: " + Arrays.toString(mask.shape()));

        // Reset network state before processing
        net.clearLayerMaskArrays();
        net.clear();

        // Ensure mask is [batch_size, sequence_length]
        if (mask.rank() == 2) {
            if (mask.shape()[1] < maxLen) {
                // Pad mask with zeros
                INDArray padded = Nd4j.zeros(mask.shape()[0], maxLen);
                padded.get(NDArrayIndex.all(), NDArrayIndex.interval(0, mask.shape()[1]))
                    .assign(mask);
                mask = padded;
            } else if (mask.shape()[1] > maxLen) {
                // Truncate mask to maxLen
                mask = mask.get(NDArrayIndex.all(), NDArrayIndex.interval(0, maxLen));
            }
        }

        // No need to set mask arrays as the network handles masking internally
        Map<String, INDArray> outs = net.feedForward(new INDArray[]{ seq }, false);

        switch (ask) {
            case CAST:   return outs.get("cast_head");
            case ATTACK: return outs.get("atk_head");
            case BLOCK:  return outs.get("blk_head");
            default:     throw new IllegalStateException("Unknown ask type");
        }
    }

    // ---------------------------------------------------------------------
    // training (supervised target per head) – minimal example
    // ---------------------------------------------------------------------
    public void fit(INDArray seq, INDArray mask,
                    INDArray castTargets, INDArray atkTargets, INDArray blkTargets) {
        // Log shapes for debugging
        RLTrainer.threadLocalLogger.get().info("Input sequence shape: " + Arrays.toString(seq.shape()));
        RLTrainer.threadLocalLogger.get().info("Cast targets shape: " + Arrays.toString(castTargets.shape()));

        // Ensure input shape is correct [batch_size, d, maxLen]
        if (seq.rank() == 2) {
            // Reshape to [batch_size, d, maxLen]
            seq = seq.reshape(seq.shape()[0], d, maxLen);
        }

        // Ensure mask shape is correct [batch_size, maxLen]
        if (mask.rank() == 2) {
            if (mask.shape()[1] < maxLen) {
                INDArray padded = Nd4j.zeros(mask.shape()[0], maxLen);
                padded.get(NDArrayIndex.all(), NDArrayIndex.interval(0, mask.shape()[1]))
                    .assign(mask);
                mask = padded;
            } else if (mask.shape()[1] > maxLen) {
                mask = mask.get(NDArrayIndex.all(), NDArrayIndex.interval(0, maxLen));
            }
        }

        // No need to set mask arrays as the network handles masking internally
        net.fit(new INDArray[]{ seq }, new INDArray[]{ castTargets, atkTargets, blkTargets });
    }

    // ---------------------------------------------------------------------
    // persistence helpers
    // ---------------------------------------------------------------------
    public void save(String path) throws IOException {
        org.deeplearning4j.util.ModelSerializer.writeModel(net, path, true);
    }

    public void load(String path) throws IOException {
        ComputationGraph loaded = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(path);
        net.setParams(loaded.params());
    }

    // ---------------------------------------------------------------------
    // Batch inference helper (returns CAST logits for a batch)
    // ---------------------------------------------------------------------
    public INDArray batchCastLogits(INDArray seqBatch, INDArray maskBatch) {
        // Ensure input shape is correct [batch_size, d, maxLen]
        if (seqBatch.rank() == 2) {
            // Reshape to [batch_size, d, maxLen]
            seqBatch = seqBatch.reshape(seqBatch.shape()[0], d, maxLen);
        }

        // Ensure mask has correct shape [batch_size, maxLen]
        if (maskBatch.rank() == 2) {
            if (maskBatch.shape()[1] < maxLen) {
                // Pad mask with zeros
                INDArray padded = Nd4j.zeros(maskBatch.shape()[0], maxLen);
                padded.get(NDArrayIndex.all(), NDArrayIndex.interval(0, maskBatch.shape()[1]))
                    .assign(maskBatch);
                maskBatch = padded;
            } else if (maskBatch.shape()[1] > maxLen) {
                // Truncate mask to maxLen
                maskBatch = maskBatch.get(NDArrayIndex.all(), NDArrayIndex.interval(0, maxLen));
            }
        }

        // No need to set mask arrays as the network handles masking internally
        INDArray[] outs = net.output(false, seqBatch);
        return outs[0]; // cast_head is configured as first output
    }

    // ---------------------------------------------------------------------
    // helper: produce random output for ε-greedy exploration
    // ---------------------------------------------------------------------
    public INDArray randomOutput(AskType ask) {
        String layer;
        switch (ask) {
            case CAST:   layer = "cast_head"; break;
            case ATTACK: layer = "atk_head"; break;
            case BLOCK:  layer = "blk_head"; break;
            default:     throw new IllegalStateException("Unknown ask type");
        }

        long nOut = ((org.deeplearning4j.nn.conf.layers.FeedForwardLayer)
                net.getLayer(layer).conf().getLayer()).getNOut();

        // Generate random probabilities that sum to 1
        INDArray random = Nd4j.rand(1, (int) nOut);
        random = random.div(random.sumNumber());
        return random.reshape(nOut);  // Reshape to 1D array
    }

    // ---------------------------------------------------------------------
    // Getters for RLModel
    // ---------------------------------------------------------------------
    public Random getRandom() {
        return rnd;
    }

    public double getEpsilon() {
        return epsilon;
    }
}
