package mage.player.ai.rl;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Reuses the existing little-endian flat tensor layouts from the local Py4J path.
 */
final class SharedGpuTensorSerde {

    private SharedGpuTensorSerde() {
    }

    static byte[][] buildScoreSegments(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask
    ) {
        return new byte[][]{
                floats2dToBytes(state.getSequence()),
                intsToBytes(state.getMask()),
                intsToBytes(state.getTokenIds()),
                floats2dToBytes(candidateFeatures),
                intsToBytes(candidateActionIds),
                intsToBytes(candidateMask)
        };
    }

    static byte[] packSegments(byte[]... segments) {
        byte[][] safeSegments = new byte[segments.length][];
        int totalBytes = Integer.BYTES;
        for (int i = 0; i < segments.length; i++) {
            byte[] safe = segments[i] == null ? new byte[0] : segments[i];
            safeSegments[i] = safe;
            totalBytes += Integer.BYTES + safe.length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(totalBytes);
        buffer.putInt(safeSegments.length);
        for (byte[] safe : safeSegments) {
            buffer.putInt(safe.length);
            buffer.put(safe);
        }
        return buffer.array();
    }

    static List<byte[]> unpackSegments(byte[] payload) throws IOException {
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(payload == null ? new byte[0] : payload));
        int count = in.readInt();
        List<byte[]> segments = new ArrayList<>(Math.max(0, count));
        for (int i = 0; i < count; i++) {
            int length = in.readInt();
            if (length < 0) {
                throw new IOException("Negative shared GPU tensor segment length: " + length);
            }
            byte[] segment = new byte[length];
            in.readFully(segment);
            segments.add(segment);
        }
        return segments;
    }

    static byte[] buildScorePayload(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask
    ) {
        return packSegments(buildScoreSegments(state, candidateActionIds, candidateFeatures, candidateMask));
    }

    static PythonMLBatchManager.PredictionResult decodePredictionResult(byte[] payload, int maxCandidates) {
        ByteBuffer buffer = ByteBuffer.wrap(payload == null ? new byte[0] : payload).order(ByteOrder.LITTLE_ENDIAN);
        float[] policy = new float[maxCandidates];
        for (int i = 0; i < maxCandidates && buffer.remaining() >= 4; i++) {
            policy[i] = buffer.getFloat();
        }
        float value = buffer.remaining() >= 4 ? buffer.getFloat() : 0.0f;
        return new PythonMLBatchManager.PredictionResult(policy, value);
    }

    static byte[] buildTrainPayload(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        if (trainingData == null || trainingData.isEmpty()) {
            return packSegments();
        }
        int batchSize = trainingData.size();
        int seqLen = trainingData.get(0).state.getSequence().length;
        int dModel = trainingData.get(0).state.getSequence()[0].length;
        int maxCandidates = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int candFeatDim = StateSequenceBuilder.TrainingData.CAND_FEAT_DIM;

        float[] sequences = new float[batchSize * seqLen * dModel];
        int[] masks = new int[batchSize * seqLen];
        int[] tokenIds = new int[batchSize * seqLen];
        float[] candFeat = new float[batchSize * maxCandidates * candFeatDim];
        int[] candIds = new int[batchSize * maxCandidates];
        int[] candMask = new int[batchSize * maxCandidates];
        int[] chosenIndices = new int[batchSize * maxCandidates];
        float[] rewardValues = new float[batchSize];
        int[] chosenCount = new int[batchSize];
        float[] oldLogpTotal = new float[batchSize];
        float[] oldValue = new float[batchSize];
        float[] sampleWeights = new float[batchSize];
        int[] dones = new int[batchSize];
        int[] headIdx = new int[batchSize];

        int seqOffset = 0;
        int maskOffset = 0;
        int tokenOffset = 0;
        int candFeatOffset = 0;
        int candOffset = 0;
        for (int i = 0; i < batchSize; i++) {
            StateSequenceBuilder.TrainingData item = trainingData.get(i);
            float[][] sequence = item.state.getSequence();
            for (float[] token : sequence) {
                System.arraycopy(token, 0, sequences, seqOffset, token.length);
                seqOffset += token.length;
            }
            System.arraycopy(item.state.getMask(), 0, masks, maskOffset, item.state.getMask().length);
            maskOffset += item.state.getMask().length;
            System.arraycopy(item.state.getTokenIds(), 0, tokenIds, tokenOffset, item.state.getTokenIds().length);
            tokenOffset += item.state.getTokenIds().length;

            for (float[] feature : item.candidateFeatures) {
                System.arraycopy(feature, 0, candFeat, candFeatOffset, feature.length);
                candFeatOffset += feature.length;
            }
            System.arraycopy(item.candidateActionIds, 0, candIds, candOffset, item.candidateActionIds.length);
            System.arraycopy(item.candidateMask, 0, candMask, candOffset, item.candidateMask.length);
            System.arraycopy(item.chosenIndices, 0, chosenIndices, candOffset, item.chosenIndices.length);
            candOffset += item.candidateActionIds.length;

            rewardValues[i] = rewards != null && i < rewards.size() && rewards.get(i) != null
                    ? rewards.get(i).floatValue()
                    : 0.0f;
            chosenCount[i] = item.chosenCount;
            oldLogpTotal[i] = item.oldLogpTotal;
            oldValue[i] = item.oldValue;
            sampleWeights[i] = 1.0f;
            dones[i] = (i == batchSize - 1) ? 1 : 0;
            headIdx[i] = actionTypeToHeadIdx(item.actionType);
        }

        return packSegments(
                floatsToBytes(sequences),
                intsToBytes(masks),
                intsToBytes(tokenIds),
                floatsToBytes(candFeat),
                intsToBytes(candIds),
                intsToBytes(candMask),
                floatsToBytes(rewardValues),
                intsToBytes(chosenIndices),
                intsToBytes(chosenCount),
                floatsToBytes(oldLogpTotal),
                floatsToBytes(oldValue),
                floatsToBytes(sampleWeights),
                intsToBytes(dones),
                intsToBytes(headIdx)
        );
    }

    static byte[] floatFeaturesToBytes(float[] values) {
        return floatsToBytes(values);
    }

    static byte[] floatsToBytes(float[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        putFloats(buffer, data);
        return buffer.array();
    }

    static byte[] intsToBytes(int[] data) {
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        putInts(buffer, data);
        return buffer.array();
    }

    private static byte[] floats2dToBytes(float[][] data) {
        ByteBuffer buffer = ByteBuffer.allocate(bytesForFloats2d(data)).order(ByteOrder.LITTLE_ENDIAN);
        putFloats2d(buffer, data);
        return buffer.array();
    }

    static int bytesForInts(int[] data) {
        return data == null ? 0 : data.length * 4;
    }

    static int bytesForFloats2d(float[][] data) {
        if (data == null || data.length == 0) {
            return 0;
        }
        int total = 0;
        for (float[] row : data) {
            if (row != null) {
                total += row.length * 4;
            }
        }
        return total;
    }

    static void putInts(ByteBuffer buffer, int[] data) {
        if (data == null) {
            return;
        }
        for (int value : data) {
            buffer.putInt(value);
        }
    }

    static void putFloats(ByteBuffer buffer, float[] data) {
        if (data == null) {
            return;
        }
        for (float value : data) {
            buffer.putFloat(value);
        }
    }

    static void putFloats2d(ByteBuffer buffer, float[][] data) {
        if (data == null) {
            return;
        }
        for (float[] row : data) {
            putFloats(buffer, row);
        }
    }

    private static int actionTypeToHeadIdx(StateSequenceBuilder.ActionType actionType) {
        if (actionType == null) {
            return 0;
        }
        switch (actionType) {
            case SELECT_TARGETS:
                return 1;
            case LONDON_MULLIGAN:
            case SELECT_CARD:
                return 2;
            case DECLARE_ATTACKS:
            case DECLARE_ATTACK_TARGET:
                return 3;
            case DECLARE_BLOCKS:
                return 4;
            default:
                return 0;
        }
    }
}
