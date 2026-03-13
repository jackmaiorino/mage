package mage.player.ai.rl;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Shared GPU backend client.
 *
 * Opens a pool of TCP connections to the local GPU host to avoid single-socket
 * contention at high thread counts. The higher-level PythonModel API remains
 * unchanged.
 */
public final class SharedGpuPythonModel implements PythonModel {

    private static final Logger logger = Logger.getLogger(SharedGpuPythonModel.class.getName());
    private static final int CONNECT_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("GPU_SERVICE_CONNECT_TIMEOUT_MS", 15_000));
    private static final int CONTROL_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("GPU_SERVICE_CONTROL_TIMEOUT_MS", 30_000));
    private static final int SCORE_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("PY_SCORE_TIMEOUT_MS", 60_000));
    private static final int OUTBOUND_QUEUE_CAPACITY = Math.max(128, EnvConfig.i32("GPU_SERVICE_OUTBOUND_QUEUE", 4096));
    private static final int LOCAL_SCORE_BATCH_MAX_SIZE = Math.max(1,
            EnvConfig.i32("GPU_SERVICE_LOCAL_BATCH_MAX_SIZE", PythonMLBatchManager.getConfiguredMaxBatchSize()));
    // Keep the JVM-side queue as a very thin staging buffer in shared-GPU mode.
    // The Python host owns the real cross-request batching window.
    private static final int LOCAL_SCORE_BATCH_TIMEOUT_MS = Math.max(1,
            EnvConfig.i32("GPU_SERVICE_LOCAL_BATCH_TIMEOUT_MS", 5));
    private static final int LOCAL_TRAIN_BATCH_MAX_EPISODES = Math.max(1,
            EnvConfig.i32("GPU_SERVICE_LOCAL_TRAIN_BATCH_MAX_EPISODES", EnvConfig.i32("LEARNER_BATCH_MAX_EPISODES", 8)));
    private static final int LOCAL_TRAIN_BATCH_TIMEOUT_MS = Math.max(1,
            EnvConfig.i32("GPU_SERVICE_LOCAL_TRAIN_BATCH_TIMEOUT_MS", 100));

    private static volatile SharedGpuPythonModel instance;
    private static final Object INSTANCE_LOCK = new Object();

    public static SharedGpuPythonModel getInstance() {
        SharedGpuPythonModel current = instance;
        if (current != null) {
            return current;
        }
        synchronized (INSTANCE_LOCK) {
            current = instance;
            if (current == null) {
                current = new SharedGpuPythonModel();
                instance = current;
            }
        }
        return current;
    }

    private static final int NUM_CHANNELS = Math.max(1, EnvConfig.i32("GPU_SERVICE_NUM_CHANNELS", 4));

    private final String profileId;
    private final String endpoint;
    private final String endpointHost;
    private final int endpointPort;
    private final MetricsCollector metrics = MetricsCollector.getInstance();
    private final AtomicLong requestIdSeq = new AtomicLong(1L);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private final Object predictionLock = new Object();
    private final Object trainLock = new Object();
    private final List<ScoreRequest> predictionQueue = new ArrayList<>();
    private final List<TrainRequest> trainQueue = new ArrayList<>();
    private final ScheduledExecutorService predictionScheduler;
    private final ScheduledExecutorService trainScheduler;
    private final Channel[] channels;
    private final AtomicLong channelRoundRobin = new AtomicLong(0);

    private volatile int trainQueueDepth;
    private volatile long droppedTrainEpisodes;

    private final class Channel {
        final int index;
        final ConcurrentHashMap<Long, CompletableFuture<SharedGpuProtocol.ResponseFrame>> pending = new ConcurrentHashMap<>();
        final BlockingQueue<OutboundRequest> outbound = new LinkedBlockingQueue<>(OUTBOUND_QUEUE_CAPACITY);
        final Object connectionLock = new Object();
        volatile Socket socket;
        volatile InputStream input;
        volatile OutputStream output;
        volatile Thread readerThread;
        volatile Thread writerThread;
        volatile boolean registered;

        Channel(int index) {
            this.index = index;
        }

        boolean isConnected() {
            Socket s = socket;
            return s != null && s.isConnected() && !s.isClosed();
        }

        void ensureReady() {
            if (shutdown.get()) {
                throw new IllegalStateException("Shared GPU client already shut down");
            }
            synchronized (connectionLock) {
                if (!isConnected()) {
                    connectLocked();
                }
                if (!registered) {
                    registerLocked();
                }
            }
        }

        void connectLocked() {
            closeConnection(new IOException("Reconnecting shared GPU channel " + index));
            try {
                Socket newSocket = new Socket();
                newSocket.setTcpNoDelay(true);
                newSocket.setKeepAlive(true);
                newSocket.connect(new InetSocketAddress(endpointHost, endpointPort), CONNECT_TIMEOUT_MS);
                socket = newSocket;
                input = newSocket.getInputStream();
                output = newSocket.getOutputStream();
                registered = false;
                startThreads();
                logger.info("Connected shared GPU channel " + index + " profile=" + profileId + " endpoint=" + endpoint);
            } catch (IOException e) {
                throw new IllegalStateException("Failed to connect channel " + index + " to shared GPU host at " + endpoint, e);
            }
        }

        void registerLocked() {
            Map<String, String> headers = buildRegisterHeaders();
            SharedGpuProtocol.ResponseFrame response = invokeOnChannel(this, SharedGpuProtocol.OP_REGISTER_PROFILE, headers, new byte[0], CONTROL_TIMEOUT_MS);
            if (response.status != SharedGpuProtocol.STATUS_OK) {
                throw new IllegalStateException(response.headers.getOrDefault("error", "Shared GPU profile registration failed on channel " + index));
            }
            registered = true;
        }

        void startThreads() {
            writerThread = new Thread(() -> writerLoop(this), "SharedGpuWriter-" + profileId + "-ch" + index);
            writerThread.setDaemon(true);
            writerThread.start();
            readerThread = new Thread(() -> readerLoop(this), "SharedGpuReader-" + profileId + "-ch" + index);
            readerThread.setDaemon(true);
            readerThread.start();
        }

        void closeConnection(IOException reason) {
            Socket s = socket;
            socket = null;
            input = null;
            output = null;
            registered = false;
            if (s != null) {
                try {
                    s.close();
                } catch (IOException ignored) {
                }
            }
            Thread r = readerThread;
            readerThread = null;
            if (r != null) {
                r.interrupt();
            }
            Thread w = writerThread;
            writerThread = null;
            if (w != null) {
                w.interrupt();
            }
            while (!outbound.isEmpty()) {
                OutboundRequest dropped = outbound.poll();
                if (dropped != null) {
                    CompletableFuture<SharedGpuProtocol.ResponseFrame> f = pending.remove(dropped.requestId);
                    if (f != null) {
                        f.completeExceptionally(reason);
                    }
                }
            }
        }

        void failPending(Throwable error) {
            List<Long> ids = new ArrayList<>(pending.keySet());
            for (Long id : ids) {
                CompletableFuture<SharedGpuProtocol.ResponseFrame> f = pending.remove(id);
                if (f != null) {
                    f.completeExceptionally(error);
                }
            }
        }
    }

    private static final class BatchKey {
        final String policyKey;
        final String headId;
        final int seqLen;
        final int dModel;
        final int maxCandidates;
        final int candFeatDim;

        private BatchKey(
                String policyKey,
                String headId,
                int seqLen,
                int dModel,
                int maxCandidates,
                int candFeatDim
        ) {
            this.policyKey = policyKey;
            this.headId = headId;
            this.seqLen = seqLen;
            this.dModel = dModel;
            this.maxCandidates = maxCandidates;
            this.candFeatDim = candFeatDim;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            BatchKey other = (BatchKey) o;
            return seqLen == other.seqLen
                    && dModel == other.dModel
                    && maxCandidates == other.maxCandidates
                    && candFeatDim == other.candFeatDim
                    && Objects.equals(policyKey, other.policyKey)
                    && Objects.equals(headId, other.headId);
        }

        @Override
        public int hashCode() {
            return Objects.hash(policyKey, headId, seqLen, dModel, maxCandidates, candFeatDim);
        }
    }

    private static final class ScoreRequest {
        final StateSequenceBuilder.SequenceOutput state;
        final int[] candidateActionIds;
        final float[][] candidateFeatures;
        final int[] candidateMask;
        final int originalCandidateCount;
        final BatchKey batchKey;
        final CompletableFuture<PythonMLBatchManager.PredictionResult> future;

        private ScoreRequest(
                StateSequenceBuilder.SequenceOutput state,
                int[] candidateActionIds,
                float[][] candidateFeatures,
                int[] candidateMask,
                int originalCandidateCount,
                BatchKey batchKey,
                CompletableFuture<PythonMLBatchManager.PredictionResult> future
        ) {
            this.state = state;
            this.candidateActionIds = candidateActionIds;
            this.candidateFeatures = candidateFeatures;
            this.candidateMask = candidateMask;
            this.originalCandidateCount = originalCandidateCount;
            this.batchKey = batchKey;
            this.future = future;
        }
    }

    private static final class TrainBatchKey {
        final int seqLen;
        final int dModel;
        final int maxCandidates;
        final int candFeatDim;

        private TrainBatchKey(int seqLen, int dModel, int maxCandidates, int candFeatDim) {
            this.seqLen = seqLen;
            this.dModel = dModel;
            this.maxCandidates = maxCandidates;
            this.candFeatDim = candFeatDim;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            TrainBatchKey other = (TrainBatchKey) o;
            return seqLen == other.seqLen
                    && dModel == other.dModel
                    && maxCandidates == other.maxCandidates
                    && candFeatDim == other.candFeatDim;
        }

        @Override
        public int hashCode() {
            return Objects.hash(seqLen, dModel, maxCandidates, candFeatDim);
        }
    }

    private static final class TrainRequest {
        final List<StateSequenceBuilder.TrainingData> trainingData;
        final List<Double> rewards;
        final TrainBatchKey batchKey;
        final int stepCount;
        final int episodeCount;

        private TrainRequest(
                List<StateSequenceBuilder.TrainingData> trainingData,
                List<Double> rewards,
                TrainBatchKey batchKey
        ) {
            this.trainingData = trainingData;
            this.rewards = rewards;
            this.batchKey = batchKey;
            this.stepCount = trainingData.size();
            // enqueueTraining is called once per finished game/trajectory.
            this.episodeCount = 1;
        }
    }

    private SharedGpuPythonModel() {
        this.profileId = EnvConfig.str("MODEL_PROFILE", "").trim();
        if (this.profileId.isEmpty()) {
            throw new IllegalStateException("PY_SERVICE_MODE=shared_gpu requires MODEL_PROFILE to be set");
        }
        this.predictionScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "SharedGpuLocalBatch-" + this.profileId);
            t.setDaemon(true);
            return t;
        });
        this.trainScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "SharedGpuLocalTrainBatch-" + this.profileId);
            t.setDaemon(true);
            return t;
        });
        this.endpoint = EnvConfig.str("GPU_SERVICE_ENDPOINT", "").trim();
        if (this.endpoint.isEmpty()) {
            throw new IllegalStateException("PY_SERVICE_MODE=shared_gpu requires GPU_SERVICE_ENDPOINT to be set");
        }
        int sep = this.endpoint.lastIndexOf(':');
        if (sep <= 0 || sep >= this.endpoint.length() - 1) {
            throw new IllegalStateException("Invalid GPU_SERVICE_ENDPOINT, expected host:port but got: " + this.endpoint);
        }
        this.endpointHost = this.endpoint.substring(0, sep);
        try {
            this.endpointPort = Integer.parseInt(this.endpoint.substring(sep + 1));
        } catch (NumberFormatException e) {
            throw new IllegalStateException("Invalid GPU_SERVICE_ENDPOINT port: " + this.endpoint, e);
        }
        this.channels = new Channel[NUM_CHANNELS];
        for (int i = 0; i < NUM_CHANNELS; i++) {
            this.channels[i] = new Channel(i);
        }
        logger.info("SharedGpuPythonModel created with " + NUM_CHANNELS + " channels for profile=" + profileId);
    }

    @Override
    public PythonMLBatchManager.PredictionResult scoreCandidates(
            StateSequenceBuilder.SequenceOutput state,
            int[] candidateActionIds,
            float[][] candidateFeatures,
            int[] candidateMask,
            String policyKey,
            String headId,
            int pickIndex,
            int minTargets,
            int maxTargets
    ) {
        Objects.requireNonNull(state, "state");
        int originalCandidateCount = candidateActionIds.length;
        int maxCand = StateSequenceBuilder.TrainingData.MAX_CANDIDATES;
        int candFeatDim = candidateFeatures.length == 0 ? 0 : candidateFeatures[0].length;
        int rawSeqLen = state.getSequence().length;
        int dModel = rawSeqLen == 0 ? 0 : state.getSequence()[0].length;
        int bucketedSeqLen = bucketSeqLen(rawSeqLen);

        // Pad candidates to MAX_CANDIDATES
        int[] paddedCandIds = padInts(candidateActionIds, maxCand);
        int[] paddedCandMask = padInts(candidateMask, maxCand);
        float[][] paddedCandFeats = padFloats2d(candidateFeatures, maxCand, candFeatDim);

        // Pad sequence to bucketed length
        StateSequenceBuilder.SequenceOutput paddedState = padSequence(state, bucketedSeqLen, dModel);

        BatchKey batchKey = new BatchKey(
                safe(policyKey, "train"),
                safe(headId, "action"),
                bucketedSeqLen,
                dModel,
                maxCand,
                candFeatDim
        );
        CompletableFuture<PythonMLBatchManager.PredictionResult> future = new CompletableFuture<>();
        metrics.recordInferenceRequest();
        boolean flushNow = false;
        synchronized (predictionLock) {
            predictionQueue.add(new ScoreRequest(paddedState, paddedCandIds, paddedCandFeats, paddedCandMask, originalCandidateCount, batchKey, future));
            if (predictionQueue.size() >= LOCAL_SCORE_BATCH_MAX_SIZE) {
                flushNow = true;
            } else if (predictionQueue.size() == 1) {
                predictionScheduler.schedule(this::safeFlushPredictionQueue, LOCAL_SCORE_BATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            }
        }
        if (flushNow) {
            flushPredictionQueue(true);
        }
        long startNanos = System.nanoTime();
        try {
            PythonMLBatchManager.PredictionResult result = future.get(SCORE_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            metrics.recordInferenceLatencyMs((System.nanoTime() - startNanos) / 1_000_000L);
            return result;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for shared GPU batch response", e);
        } catch (TimeoutException e) {
            metrics.recordInferenceTimeout();
            throw new IllegalStateException("Timed out waiting for shared GPU batch response", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new IllegalStateException("Shared GPU batch request failed", cause);
        }
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        if (trainingData == null || trainingData.isEmpty()) {
            return;
        }
        StateSequenceBuilder.TrainingData first = trainingData.get(0);
        TrainBatchKey batchKey = new TrainBatchKey(
                first.state.getSequence().length,
                first.state.getSequence().length == 0 ? 0 : first.state.getSequence()[0].length,
                StateSequenceBuilder.TrainingData.MAX_CANDIDATES,
                StateSequenceBuilder.TrainingData.CAND_FEAT_DIM
        );
        boolean flushNow = false;
        synchronized (trainLock) {
            trainQueue.add(new TrainRequest(
                    new ArrayList<>(trainingData),
                    copyRewards(rewards, trainingData.size()),
                    batchKey
            ));
            if (queuedTrainEpisodesLocked() >= LOCAL_TRAIN_BATCH_MAX_EPISODES) {
                flushNow = true;
            } else if (trainQueue.size() == 1) {
                trainScheduler.schedule(this::safeFlushTrainQueue, LOCAL_TRAIN_BATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            }
        }
        if (flushNow) {
            flushTrainQueue(true);
        }
    }

    @Override
    public float predictMulligan(float[] features) {
        Map<String, String> headers = singletonProfileHeaders();
        headers.put("feature_count", Integer.toString(features == null ? 0 : features.length));
        SharedGpuProtocol.ResponseFrame response = invoke(
                SharedGpuProtocol.OP_PREDICT_MULLIGAN,
                headers,
                SharedGpuTensorSerde.packSegments(SharedGpuTensorSerde.floatFeaturesToBytes(features == null ? new float[0] : features)),
                CONTROL_TIMEOUT_MS
        );
        return parseFloatHeader(response.headers, "value", 0.5f);
    }

    @Override
    public float[] predictMulliganScores(float[] features) {
        Map<String, String> headers = singletonProfileHeaders();
        headers.put("feature_count", Integer.toString(features == null ? 0 : features.length));
        SharedGpuProtocol.ResponseFrame response = invoke(
                SharedGpuProtocol.OP_PREDICT_MULLIGAN_SCORES,
                headers,
                SharedGpuTensorSerde.packSegments(SharedGpuTensorSerde.floatFeaturesToBytes(features == null ? new float[0] : features)),
                CONTROL_TIMEOUT_MS
        );
        ByteBuffer buffer = ByteBuffer.wrap(response.payload == null ? new byte[0] : response.payload).order(ByteOrder.LITTLE_ENDIAN);
        float keep = buffer.remaining() >= 4 ? buffer.getFloat() : 0.0f;
        float mull = buffer.remaining() >= 4 ? buffer.getFloat() : 0.0f;
        return new float[]{keep, mull};
    }

    @Override
    public void trainMulligan(
            byte[] features,
            byte[] decisions,
            byte[] outcomes,
            byte[] gameLengths,
            byte[] earlyLandScores,
            byte[] overrides,
            int batchSize
    ) {
        Map<String, String> headers = singletonProfileHeaders();
        headers.put("batch_size", Integer.toString(batchSize));
        byte[] payload = SharedGpuTensorSerde.packSegments(
                safeBytes(features),
                safeBytes(decisions),
                safeBytes(outcomes),
                safeBytes(gameLengths),
                safeBytes(earlyLandScores),
                safeBytes(overrides)
        );
        invoke(SharedGpuProtocol.OP_TRAIN_MULLIGAN, headers, payload, CONTROL_TIMEOUT_MS);
    }

    @Override
    public void saveMulliganModel() {
        invoke(SharedGpuProtocol.OP_SAVE_MULLIGAN_MODEL, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
    }

    @Override
    public void saveModel(String path) {
        Map<String, String> headers = singletonProfileHeaders();
        headers.put("path", path == null ? "" : path);
        invoke(SharedGpuProtocol.OP_SAVE_MODEL, headers, new byte[0], CONTROL_TIMEOUT_MS);
    }

    @Override
    public String getDeviceInfo() {
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_GET_DEVICE_INFO, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
        return response.headers.getOrDefault("device_info", "");
    }

    @Override
    public Map<String, Integer> getMainModelTrainingStats() {
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_GET_MAIN_STATS, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
        Map<String, Integer> result = new LinkedHashMap<>();
        result.put("train_steps", parseIntHeader(response.headers, "train_steps", 0));
        result.put("train_samples", parseIntHeader(response.headers, "train_samples", 0));
        return result;
    }

    @Override
    public Map<String, Integer> getMulliganModelTrainingStats() {
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_GET_MULLIGAN_STATS, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
        Map<String, Integer> result = new LinkedHashMap<>();
        result.put("train_steps", parseIntHeader(response.headers, "train_steps", 0));
        result.put("train_samples", parseIntHeader(response.headers, "train_samples", 0));
        return result;
    }

    @Override
    public Map<String, Integer> getHealthStats() {
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_GET_HEALTH_STATS, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
        Map<String, Integer> result = new LinkedHashMap<>();
        result.put("gpu_oom_count", parseIntHeader(response.headers, "gpu_oom_count", 0));
        trainQueueDepth = parseIntHeader(response.headers, "train_queue_depth", trainQueueDepth);
        droppedTrainEpisodes = parseIntHeader(response.headers, "dropped_train_episodes", (int) droppedTrainEpisodes);
        return result;
    }

    @Override
    public void resetHealthStats() {
        invoke(SharedGpuProtocol.OP_RESET_HEALTH_STATS, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
    }

    @Override
    public void recordGameResult(float lastValuePrediction, boolean won) {
        Map<String, String> headers = singletonProfileHeaders();
        headers.put("last_value_prediction", Float.toString(lastValuePrediction));
        headers.put("won", Boolean.toString(won));
        invoke(SharedGpuProtocol.OP_RECORD_GAME_RESULT, headers, new byte[0], CONTROL_TIMEOUT_MS);
    }

    @Override
    public Map<String, Object> getValueHeadMetrics() {
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_GET_VALUE_HEAD_METRICS, singletonProfileHeaders(), new byte[0], CONTROL_TIMEOUT_MS);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("accuracy", parseDoubleHeader(response.headers, "accuracy", 0.0));
        result.put("avg_win", parseDoubleHeader(response.headers, "avg_win", 0.0));
        result.put("avg_loss", parseDoubleHeader(response.headers, "avg_loss", 0.0));
        result.put("samples", parseIntHeader(response.headers, "samples", 0));
        result.put("use_gae", Boolean.parseBoolean(response.headers.getOrDefault("use_gae", "false")));
        return result;
    }

    @Override
    public void shutdown() {
        if (!shutdown.compareAndSet(false, true)) {
            return;
        }
        try {
            for (Channel ch : channels) {
                if (ch.isConnected()) {
                    try {
                        invokeOnChannel(ch, SharedGpuProtocol.OP_CLOSE_PROFILE, singletonProfileHeaders(), new byte[0], 5000);
                    } catch (Exception ignored) {
                    }
                }
            }
        } finally {
            IOException shutdownError = new IOException("Shared GPU client shutdown");
            predictionScheduler.shutdownNow();
            trainScheduler.shutdownNow();
            failQueuedPredictions(shutdownError);
            clearQueuedTrainRequests();
            for (Channel ch : channels) {
                ch.closeConnection(shutdownError);
                ch.failPending(shutdownError);
            }
        }
    }

    int getTrainQueueDepth() {
        return trainQueueDepth;
    }

    long getDroppedTrainEpisodes() {
        return droppedTrainEpisodes;
    }

    private Channel pickChannel() {
        int idx = (int) (channelRoundRobin.getAndIncrement() % channels.length);
        if (idx < 0) {
            idx += channels.length;
        }
        return channels[idx];
    }

    private SharedGpuProtocol.ResponseFrame invoke(int opcode, Map<String, String> headers, byte[] payload, int timeoutMs) {
        Channel ch = pickChannel();
        ch.ensureReady();
        return invokeOnChannel(ch, opcode, headers, payload, timeoutMs);
    }

    private CompletableFuture<SharedGpuProtocol.ResponseFrame> invokeAsync(
            int opcode,
            Map<String, String> headers,
            byte[] payload,
            int timeoutMs
    ) {
        Channel ch = pickChannel();
        ch.ensureReady();
        CompletableFuture<SharedGpuProtocol.ResponseFrame> responseFuture =
                invokeOnChannelAsync(ch, opcode, headers, payload, timeoutMs);
        CompletableFuture<SharedGpuProtocol.ResponseFrame> result = new CompletableFuture<>();
        responseFuture.whenComplete((response, error) -> {
            if (error != null) {
                Throwable cause = error instanceof ExecutionException && error.getCause() != null
                        ? error.getCause()
                        : error;
                result.completeExceptionally(cause);
                return;
            }
            if (response.status != SharedGpuProtocol.STATUS_OK) {
                result.completeExceptionally(new IllegalStateException(
                        response.headers.getOrDefault("error", "Shared GPU request failed")));
                return;
            }
            result.complete(response);
        });
        return result;
    }

    private void safeFlushPredictionQueue() {
        flushPredictionQueue(false);
    }

    private void safeFlushTrainQueue() {
        flushTrainQueue(false);
    }

    private void flushPredictionQueue(boolean dueToFull) {
        List<ScoreRequest> batch;
        synchronized (predictionLock) {
            if (predictionQueue.isEmpty()) {
                return;
            }
            batch = new ArrayList<>(predictionQueue);
            predictionQueue.clear();
        }

        Map<BatchKey, List<ScoreRequest>> grouped = new LinkedHashMap<>();
        for (ScoreRequest request : batch) {
            grouped.computeIfAbsent(request.batchKey, ignored -> new ArrayList<>()).add(request);
        }
        for (List<ScoreRequest> requests : grouped.values()) {
            int offset = 0;
            while (offset < requests.size()) {
                int end = Math.min(requests.size(), offset + LOCAL_SCORE_BATCH_MAX_SIZE);
                flushPredictionBatch(new ArrayList<>(requests.subList(offset, end)),
                        dueToFull || (end - offset) >= LOCAL_SCORE_BATCH_MAX_SIZE);
                offset = end;
            }
        }
    }

    private void flushPredictionBatch(List<ScoreRequest> batch, boolean dueToFull) {
        if (batch.isEmpty()) {
            return;
        }
        BatchKey key = batch.get(0).batchKey;
        try {
            metrics.recordInferBatchFlush(batch.size(), dueToFull);
        } catch (Exception ignored) {
        }

        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        headers.put("policy_key", key.policyKey);
        headers.put("head_id", key.headId);
        headers.put("pick_index", "0");
        headers.put("min_targets", "0");
        headers.put("max_targets", "0");
        headers.put("batch_size", Integer.toString(batch.size()));
        headers.put("seq_len", Integer.toString(key.seqLen));
        headers.put("d_model", Integer.toString(key.dModel));
        headers.put("max_candidates", Integer.toString(key.maxCandidates));
        headers.put("cand_feat_dim", Integer.toString(key.candFeatDim));

        try {
            byte[] payload = buildMergedScorePayload(batch);
            // The transport already multiplexes responses by request id, so keep
            // distinct local batch-key groups in flight instead of waiting for
            // each synchronous round-trip to finish before submitting the next.
            invokeAsync(
                    SharedGpuProtocol.OP_SCORE,
                    headers,
                    payload,
                    SCORE_TIMEOUT_MS
            ).whenComplete((response, error) -> {
                if (error != null) {
                    for (ScoreRequest request : batch) {
                        request.future.completeExceptionally(error);
                    }
                    return;
                }
                try {
                    ByteBuffer buffer = ByteBuffer.wrap(response.payload == null ? new byte[0] : response.payload)
                            .order(ByteOrder.LITTLE_ENDIAN);
                    int paddedCandCount = key.maxCandidates;
                    for (ScoreRequest request : batch) {
                        int expectedBytes = (paddedCandCount + 1) * 4;
                        if (buffer.remaining() < expectedBytes) {
                            throw new IllegalStateException("Shared GPU batch score response truncated");
                        }
                        float[] paddedPolicy = new float[paddedCandCount];
                        for (int i = 0; i < paddedCandCount; i++) {
                            paddedPolicy[i] = buffer.getFloat();
                        }
                        float value = buffer.getFloat();
                        int origCount = request.originalCandidateCount;
                        float[] policy = origCount == paddedCandCount ? paddedPolicy : Arrays.copyOf(paddedPolicy, origCount);
                        request.future.complete(new PythonMLBatchManager.PredictionResult(policy, value));
                    }
                } catch (Exception e) {
                    for (ScoreRequest request : batch) {
                        request.future.completeExceptionally(e);
                    }
                }
            });
        } catch (Exception e) {
            for (ScoreRequest request : batch) {
                request.future.completeExceptionally(e);
            }
        }
    }

    private void flushTrainQueue(boolean dueToFull) {
        List<TrainRequest> queued;
        synchronized (trainLock) {
            if (trainQueue.isEmpty()) {
                return;
            }
            queued = new ArrayList<>(trainQueue);
            trainQueue.clear();
        }

        Map<TrainBatchKey, List<TrainRequest>> grouped = new LinkedHashMap<>();
        for (TrainRequest request : queued) {
            grouped.computeIfAbsent(request.batchKey, ignored -> new ArrayList<>()).add(request);
        }
        for (List<TrainRequest> requests : grouped.values()) {
            List<TrainRequest> batch = new ArrayList<>();
            int episodes = 0;
            for (TrainRequest request : requests) {
                if (!batch.isEmpty() && (episodes + request.episodeCount) > LOCAL_TRAIN_BATCH_MAX_EPISODES) {
                    flushTrainBatch(new ArrayList<>(batch), dueToFull || episodes >= LOCAL_TRAIN_BATCH_MAX_EPISODES);
                    batch.clear();
                    episodes = 0;
                }
                batch.add(request);
                episodes += request.episodeCount;
            }
            flushTrainBatch(batch, dueToFull || episodes >= LOCAL_TRAIN_BATCH_MAX_EPISODES);
        }
    }

    private void flushTrainBatch(List<TrainRequest> batch, boolean dueToFull) {
        if (batch == null || batch.isEmpty()) {
            return;
        }
        TrainBatchKey key = batch.get(0).batchKey;
        List<StateSequenceBuilder.TrainingData> mergedTrainingData = new ArrayList<>();
        List<Double> mergedRewards = new ArrayList<>();
        for (TrainRequest request : batch) {
            mergedTrainingData.addAll(request.trainingData);
            mergedRewards.addAll(request.rewards);
        }

        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        headers.put("batch_size", Integer.toString(mergedTrainingData.size()));
        headers.put("seq_len", Integer.toString(key.seqLen));
        headers.put("d_model", Integer.toString(key.dModel));
        headers.put("max_candidates", Integer.toString(key.maxCandidates));
        headers.put("cand_feat_dim", Integer.toString(key.candFeatDim));
        headers.put("local_flush_reason", dueToFull ? "full" : "timeout");
        byte[] payload = SharedGpuTensorSerde.buildTrainPayload(mergedTrainingData, mergedRewards);
        try {
            SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_ENQUEUE_TRAIN, headers, payload, CONTROL_TIMEOUT_MS);
            trainQueueDepth = parseIntHeader(response.headers, "queue_depth", trainQueueDepth);
            droppedTrainEpisodes = parseIntHeader(response.headers, "dropped_train_episodes", (int) droppedTrainEpisodes);
        } catch (Exception e) {
            logger.warning("Shared GPU local train batch flush failed: " + e.getMessage());
            synchronized (trainLock) {
                List<TrainRequest> requeue = new ArrayList<>(batch.size() + trainQueue.size());
                requeue.addAll(batch);
                requeue.addAll(trainQueue);
                trainQueue.clear();
                trainQueue.addAll(requeue);
                trainScheduler.schedule(this::safeFlushTrainQueue, LOCAL_TRAIN_BATCH_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            }
        }
    }

    private byte[] buildMergedScorePayload(List<ScoreRequest> batch) {
        int sequenceBytes = 0;
        int maskBytes = 0;
        int tokenBytes = 0;
        int candidateFeatureBytes = 0;
        int candidateIdBytes = 0;
        int candidateMaskBytes = 0;
        for (ScoreRequest request : batch) {
            sequenceBytes += SharedGpuTensorSerde.bytesForFloats2d(request.state.getSequence());
            maskBytes += SharedGpuTensorSerde.bytesForInts(request.state.getMask());
            tokenBytes += SharedGpuTensorSerde.bytesForInts(request.state.getTokenIds());
            candidateFeatureBytes += SharedGpuTensorSerde.bytesForFloats2d(request.candidateFeatures);
            candidateIdBytes += SharedGpuTensorSerde.bytesForInts(request.candidateActionIds);
            candidateMaskBytes += SharedGpuTensorSerde.bytesForInts(request.candidateMask);
        }

        ByteBuffer[] merged = new ByteBuffer[]{
                ByteBuffer.allocate(sequenceBytes).order(ByteOrder.LITTLE_ENDIAN),
                ByteBuffer.allocate(maskBytes).order(ByteOrder.LITTLE_ENDIAN),
                ByteBuffer.allocate(tokenBytes).order(ByteOrder.LITTLE_ENDIAN),
                ByteBuffer.allocate(candidateFeatureBytes).order(ByteOrder.LITTLE_ENDIAN),
                ByteBuffer.allocate(candidateIdBytes).order(ByteOrder.LITTLE_ENDIAN),
                ByteBuffer.allocate(candidateMaskBytes).order(ByteOrder.LITTLE_ENDIAN)
        };
        for (ScoreRequest request : batch) {
            SharedGpuTensorSerde.putFloats2d(merged[0], request.state.getSequence());
            SharedGpuTensorSerde.putInts(merged[1], request.state.getMask());
            SharedGpuTensorSerde.putInts(merged[2], request.state.getTokenIds());
            SharedGpuTensorSerde.putFloats2d(merged[3], request.candidateFeatures);
            SharedGpuTensorSerde.putInts(merged[4], request.candidateActionIds);
            SharedGpuTensorSerde.putInts(merged[5], request.candidateMask);
        }
        return SharedGpuTensorSerde.packSegments(
                merged[0].array(),
                merged[1].array(),
                merged[2].array(),
                merged[3].array(),
                merged[4].array(),
                merged[5].array()
        );
    }

    private int queuedTrainEpisodesLocked() {
        int total = 0;
        for (TrainRequest request : trainQueue) {
            total += request.episodeCount;
        }
        return total;
    }

    private SharedGpuProtocol.ResponseFrame invokeOnChannel(Channel ch, int opcode, Map<String, String> headers, byte[] payload, int timeoutMs) {
        try {
            long requestId = requestIdSeq.getAndIncrement();
            CompletableFuture<SharedGpuProtocol.ResponseFrame> future = new CompletableFuture<>();
            ch.pending.put(requestId, future);
            OutboundRequest request = new OutboundRequest(opcode, requestId, headers, payload);
            if (!ch.outbound.offer(request, Math.max(1000, timeoutMs), TimeUnit.MILLISECONDS)) {
                ch.pending.remove(requestId);
                throw new IllegalStateException("Shared GPU outbound queue is full on channel " + ch.index);
            }
            SharedGpuProtocol.ResponseFrame response = future.get(timeoutMs, TimeUnit.MILLISECONDS);
            if (response.status != SharedGpuProtocol.STATUS_OK) {
                throw new IllegalStateException(response.headers.getOrDefault("error", "Shared GPU request failed"));
            }
            return response;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for shared GPU response", e);
        } catch (TimeoutException e) {
            throw new IllegalStateException("Timed out waiting for shared GPU response op=" + opcode + " ch=" + ch.index, e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new IllegalStateException("Shared GPU request failed", cause);
        }
    }

    private CompletableFuture<SharedGpuProtocol.ResponseFrame> invokeOnChannelAsync(
            Channel ch,
            int opcode,
            Map<String, String> headers,
            byte[] payload,
            int timeoutMs
    ) {
        try {
            long requestId = requestIdSeq.getAndIncrement();
            CompletableFuture<SharedGpuProtocol.ResponseFrame> future = new CompletableFuture<>();
            ch.pending.put(requestId, future);
            OutboundRequest request = new OutboundRequest(opcode, requestId, headers, payload);
            if (!ch.outbound.offer(request, Math.max(1000, timeoutMs), TimeUnit.MILLISECONDS)) {
                ch.pending.remove(requestId);
                future.completeExceptionally(new IllegalStateException("Shared GPU outbound queue is full on channel " + ch.index));
            }
            return future;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            CompletableFuture<SharedGpuProtocol.ResponseFrame> future = new CompletableFuture<>();
            future.completeExceptionally(new IllegalStateException("Interrupted while queueing shared GPU request", e));
            return future;
        }
    }

    private void writerLoop(Channel ch) {
        while (!shutdown.get()) {
            try {
                OutboundRequest request = ch.outbound.take();
                OutputStream currentOutput = ch.output;
                if (currentOutput == null) {
                    throw new IOException("Shared GPU socket output closed on channel " + ch.index);
                }
                SharedGpuProtocol.writeRequest(currentOutput, request.opcode, request.requestId, request.headers, request.payload);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                ch.closeConnection(e instanceof IOException ? (IOException) e : new IOException(e));
                failQueuedPredictions(e);
                ch.failPending(e);
                break;
            }
        }
    }

    private void readerLoop(Channel ch) {
        while (!shutdown.get()) {
            try {
                InputStream currentInput = ch.input;
                if (currentInput == null) {
                    throw new IOException("Shared GPU socket input closed on channel " + ch.index);
                }
                SharedGpuProtocol.ResponseFrame response = SharedGpuProtocol.readResponse(currentInput);
                CompletableFuture<SharedGpuProtocol.ResponseFrame> future = ch.pending.remove(response.requestId);
                if (future != null) {
                    future.complete(response);
                }
            } catch (EOFException eof) {
                ch.closeConnection(new IOException("Shared GPU host closed connection on channel " + ch.index, eof));
                failQueuedPredictions(eof);
                ch.failPending(eof);
                break;
            } catch (Exception e) {
                ch.closeConnection(e instanceof IOException ? (IOException) e : new IOException(e));
                failQueuedPredictions(e);
                ch.failPending(e);
                break;
            }
        }
    }

    private void failQueuedPredictions(Throwable error) {
        if (error == null) {
            return;
        }
        List<ScoreRequest> queued;
        synchronized (predictionLock) {
            if (predictionQueue.isEmpty()) {
                return;
            }
            queued = new ArrayList<>(predictionQueue);
            predictionQueue.clear();
        }
        for (ScoreRequest request : queued) {
            request.future.completeExceptionally(error);
        }
    }

    private void clearQueuedTrainRequests() {
        synchronized (trainLock) {
            trainQueue.clear();
        }
    }

    private Map<String, String> buildRegisterHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        headers.put("endpoint", endpoint);

        Map<String, String> env = new LinkedHashMap<>(System.getenv());
        env.put("MODEL_PROFILE", profileId);
        env.put("RL_MODELS_DIR", RLLogPaths.MODELS_BASE_DIR);
        env.put("RL_LOGS_DIR", RLLogPaths.LOGS_BASE_DIR);
        env.put("MODEL_PATH", RLLogPaths.MODEL_FILE_PATH);
        env.put("MTG_MODEL_PATH", RLLogPaths.MODEL_FILE_PATH);
        env.put("MODEL_LATEST_PATH", defaultLatestPath(RLLogPaths.MODEL_FILE_PATH));
        env.put("SNAPSHOT_DIR", RLLogPaths.SNAPSHOT_DIR);
        env.put("MULLIGAN_MODEL_PATH", RLLogPaths.MULLIGAN_MODEL_PATH);
        env.put("TRAINING_LOSSES_PATH", RLLogPaths.TRAINING_LOSSES_PATH);
        env.put("HEALTH_LOG_PATH", RLLogPaths.HEALTH_LOG_PATH);
        env.put("PY_BACKEND_MODE", "single");
        env.put("INFER_WORKERS", "0");
        env.put("MODEL_RELOAD_EVERY_MS", "0");
        env.put("MODEL_SYNC_EVERY_MS", "0");
        env.put("MODEL_SYNC_EVERY_TRAIN_STEPS", "0");
        env.put("PY_SERVICE_MODE", "shared_gpu");
        if (!env.containsKey("MULLIGAN_DEVICE") || env.get("MULLIGAN_DEVICE").trim().isEmpty()) {
            env.put("MULLIGAN_DEVICE", "cpu");
        }
        for (Map.Entry<String, String> entry : env.entrySet()) {
            headers.put("env." + entry.getKey(), entry.getValue());
        }
        return headers;
    }

    private static String defaultLatestPath(String modelPath) {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            return RLLogPaths.MODELS_BASE_DIR + "/model_latest.pt";
        }
        java.io.File file = new java.io.File(modelPath);
        java.io.File parent = file.getParentFile();
        if (parent == null) {
            return RLLogPaths.MODELS_BASE_DIR + "/model_latest.pt";
        }
        return new java.io.File(parent, "model_latest.pt").getPath();
    }

    private Map<String, String> singletonProfileHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        return headers;
    }

    private static byte[] safeBytes(byte[] data) {
        return data == null ? new byte[0] : data;
    }

    private static List<Double> copyRewards(List<Double> rewards, int expectedSize) {
        List<Double> copy = new ArrayList<>(expectedSize);
        for (int i = 0; i < expectedSize; i++) {
            copy.add(rewards != null && i < rewards.size() ? rewards.get(i) : 0.0d);
        }
        return copy;
    }

    private static String safe(String value, String fallback) {
        String trimmed = value == null ? "" : value.trim();
        return trimmed.isEmpty() ? fallback : trimmed;
    }

    static int bucketSeqLen(int seqLen) {
        if (seqLen <= 0) {
            return 0;
        }
        int bucket = 32;
        while (bucket < seqLen) {
            bucket <<= 1;
        }
        return bucket;
    }

    private static int[] padInts(int[] src, int targetLen) {
        if (src.length >= targetLen) {
            return src;
        }
        int[] padded = new int[targetLen];
        System.arraycopy(src, 0, padded, 0, src.length);
        return padded;
    }

    private static float[][] padFloats2d(float[][] src, int targetRows, int cols) {
        if (src.length >= targetRows) {
            return src;
        }
        float[][] padded = new float[targetRows][];
        System.arraycopy(src, 0, padded, 0, src.length);
        for (int i = src.length; i < targetRows; i++) {
            padded[i] = new float[cols];
        }
        return padded;
    }

    private static StateSequenceBuilder.SequenceOutput padSequence(
            StateSequenceBuilder.SequenceOutput state, int targetSeqLen, int dModel) {
        int rawLen = state.getSequence().length;
        if (rawLen >= targetSeqLen) {
            return state;
        }
        float[][] origSeq = state.getSequence();
        int[] origMask = state.getMask();
        int[] origTokenIds = state.getTokenIds();

        List<float[]> tokenList = new ArrayList<>(targetSeqLen);
        List<Integer> maskList = new ArrayList<>(targetSeqLen);
        List<Integer> tokenIdList = new ArrayList<>(targetSeqLen);
        for (int i = 0; i < rawLen; i++) {
            tokenList.add(origSeq[i]);
            maskList.add(origMask[i]);
            tokenIdList.add(origTokenIds[i]);
        }
        for (int i = rawLen; i < targetSeqLen; i++) {
            tokenList.add(new float[dModel]);
            maskList.add(0);
            tokenIdList.add(0);
        }
        return new StateSequenceBuilder.SequenceOutput(tokenList, maskList, tokenIdList);
    }

    private static int parseIntHeader(Map<String, String> headers, String key, int fallback) {
        try {
            return Integer.parseInt(headers.getOrDefault(key, Integer.toString(fallback)));
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static float parseFloatHeader(Map<String, String> headers, String key, float fallback) {
        try {
            return Float.parseFloat(headers.getOrDefault(key, Float.toString(fallback)));
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static double parseDoubleHeader(Map<String, String> headers, String key, double fallback) {
        try {
            return Double.parseDouble(headers.getOrDefault(key, Double.toString(fallback)));
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private static final class OutboundRequest {
        private final int opcode;
        private final long requestId;
        private final Map<String, String> headers;
        private final byte[] payload;

        private OutboundRequest(int opcode, long requestId, Map<String, String> headers, byte[] payload) {
            this.opcode = opcode;
            this.requestId = requestId;
            this.headers = headers == null ? Collections.<String, String>emptyMap() : new LinkedHashMap<>(headers);
            this.payload = payload == null ? new byte[0] : payload;
        }
    }
}
