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
import org.deeplearning4j.nn.conf.WeightInit;
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
    private final double epsilon;
    private final ComputationGraph net;
    private final int maxLen;  // Store maxLen as a class field
    private final int d;
    private final int[] headSizes;

    // output layer names – must match builder below
    private static final String CAST_OUT  = "castOut";
    private static final String ATK_OUT   = "atkOut";
    private static final String BLK_OUT   = "blkPtr";


    public TransformerNeuralNetwork(int d, int maxLen, int[] headSizes, double epsilon) {
        this.d = d;
        this.maxLen = maxLen;
        this.epsilon = epsilon;
        this.headSizes = headSizes;

        // Increase feature dimension from 128 to 256
        int enhancedD = 256;
        
        // Increase number of attention heads from 8 to 16
        int numHeads = 16;
        
        // Increase head size from 32 to 64
        int headSize = 64;
        
        // Add more transformer layers (from 1 to 3)
        int numLayers = 3;

        // Create the network
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input_sequence", "input_mask")
                .setInputTypes(InputType.recurrent(d, maxLen), InputType.recurrent(1, maxLen));

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
                    lastLayer, "input_mask");
            lastLayer = layerName;
        }

        // Enhanced feed-forward network with more layers and larger dimensions
        builder.addLayer("ff1",
                new DenseLayer.Builder()
                        .nIn(enhancedD)
                        .nOut(enhancedD * 2)
                        .activation(Activation.RELU)
                        .build(),
                lastLayer)
                .addLayer("ff2",
                        new DenseLayer.Builder()
                                .nIn(enhancedD * 2)
                                .nOut(enhancedD)
                                .activation(Activation.RELU)
                                .build(),
                        "ff1");

        // Output heads with increased capacity
        builder.addLayer("cast_head",
                new DenseLayer.Builder()
                        .nIn(enhancedD)
                        .nOut(headSizes[0])
                        .activation(Activation.SOFTMAX)
                        .build(),
                "ff2")
                .addLayer("atk_head",
                        new DenseLayer.Builder()
                                .nIn(enhancedD)
                                .nOut(headSizes[1])
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "ff2")
                .addLayer("blk_head",
                        new DenseLayer.Builder()
                                .nIn(enhancedD)
                                .nOut(headSizes[2])
                                .activation(Activation.SOFTMAX)
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
     * ε‑greedy inference for a single state.
     */
    public INDArray predict(StateSequenceBuilder.SequenceOutput s,
                            AskType ask, boolean exploration) {
        if (s == null) {
            throw new IllegalArgumentException("SequenceOutput cannot be null");
        }

        // ε-greedy branch ----------------------------------------------------
        if (exploration && rnd.nextDouble() < epsilon) {
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

        // apply mask, run forward pass, then clear
        net.setLayerMaskArrays(new INDArray[]{ mask }, null);
        Map<String, INDArray> outs = net.feedForward(new INDArray[]{ seq }, false);
        net.clearLayerMaskArrays();

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
        // Log the actual shapes for debugging
        System.out.println("Training sequence shape: " + Arrays.toString(seq.shape()));
        System.out.println("Training mask shape: " + Arrays.toString(mask.shape()));
        System.out.println("Cast targets shape: " + Arrays.toString(castTargets.shape()));
        System.out.println("Attack targets shape: " + Arrays.toString(atkTargets.shape()));
        System.out.println("Block targets shape: " + Arrays.toString(blkTargets.shape()));

        // Ensure input and target shapes match
        if (seq.shape()[0] != castTargets.shape()[0]) {
            // If we have more input samples than targets, truncate the input
            long minSamples = Math.min(seq.shape()[0], castTargets.shape()[0]);
            seq = seq.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
            mask = mask.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
            castTargets = castTargets.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
            atkTargets = atkTargets.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
            blkTargets = blkTargets.get(NDArrayIndex.interval(0, minSamples), NDArrayIndex.all());
        }

        net.setLayerMaskArrays(new INDArray[]{ mask }, null);
        net.fit(new INDArray[]{ seq }, new INDArray[]{ castTargets, atkTargets, blkTargets });
        net.clearLayerMaskArrays();
        
        // Reset network state after training
        net.clear();
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
        net.setLayerMaskArrays(new INDArray[]{ maskBatch }, null);
        INDArray[] outs = net.output(false, seqBatch);
        net.clearLayerMaskArrays();
        return outs[0]; // CAST_OUT is configured as first output
    }

    // ---------------------------------------------------------------------
    // helper: produce random output for ε-greedy exploration
    // ---------------------------------------------------------------------
    private INDArray randomOutput(AskType ask) {
        String layer;
        switch (ask) {
            case CAST:   layer = "cast_head"; break;
            case ATTACK: layer = "atk_head"; break;
            case BLOCK:  layer = "blk_head"; break;
            default:     throw new IllegalStateException("Unknown ask type");
        }

        long nOut = ((org.deeplearning4j.nn.conf.layers.FeedForwardLayer)
                net.getLayer(layer).conf().getLayer()).getNOut();

        return Nd4j.rand(1, (int) nOut);
    }
}
