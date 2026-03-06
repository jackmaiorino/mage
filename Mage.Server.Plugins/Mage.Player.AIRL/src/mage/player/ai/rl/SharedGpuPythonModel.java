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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Shared GPU backend client.
 *
 * One JVM keeps a single socket open to the local GPU host and multiplexes
 * requests across it. The higher-level PythonModel API remains unchanged.
 */
public final class SharedGpuPythonModel implements PythonModel {

    private static final Logger logger = Logger.getLogger(SharedGpuPythonModel.class.getName());
    private static final int CONNECT_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("GPU_SERVICE_CONNECT_TIMEOUT_MS", 15_000));
    private static final int CONTROL_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("GPU_SERVICE_CONTROL_TIMEOUT_MS", 30_000));
    private static final int SCORE_TIMEOUT_MS = Math.max(1000, EnvConfig.i32("PY_SCORE_TIMEOUT_MS", 60_000));
    private static final int OUTBOUND_QUEUE_CAPACITY = Math.max(128, EnvConfig.i32("GPU_SERVICE_OUTBOUND_QUEUE", 4096));

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

    private final String profileId;
    private final String endpoint;
    private final String endpointHost;
    private final int endpointPort;
    private final MetricsCollector metrics = MetricsCollector.getInstance();
    private final AtomicLong requestIdSeq = new AtomicLong(1L);
    private final ConcurrentHashMap<Long, CompletableFuture<SharedGpuProtocol.ResponseFrame>> pending = new ConcurrentHashMap<>();
    private final BlockingQueue<OutboundRequest> outbound = new LinkedBlockingQueue<>(OUTBOUND_QUEUE_CAPACITY);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private final Object connectionLock = new Object();

    private volatile Socket socket;
    private volatile InputStream input;
    private volatile OutputStream output;
    private volatile Thread readerThread;
    private volatile Thread writerThread;
    private volatile boolean registered;
    private volatile int trainQueueDepth;
    private volatile long droppedTrainEpisodes;

    private SharedGpuPythonModel() {
        this.profileId = EnvConfig.str("MODEL_PROFILE", "").trim();
        if (this.profileId.isEmpty()) {
            throw new IllegalStateException("PY_SERVICE_MODE=shared_gpu requires MODEL_PROFILE to be set");
        }
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
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        headers.put("policy_key", safe(policyKey, "train"));
        headers.put("head_id", safe(headId, "action"));
        headers.put("pick_index", Integer.toString(pickIndex));
        headers.put("min_targets", Integer.toString(minTargets));
        headers.put("max_targets", Integer.toString(maxTargets));
        headers.put("batch_size", "1");
        headers.put("seq_len", Integer.toString(state.getSequence().length));
        headers.put("d_model", Integer.toString(state.getSequence().length == 0 ? 0 : state.getSequence()[0].length));
        headers.put("max_candidates", Integer.toString(candidateActionIds.length));
        headers.put("cand_feat_dim", Integer.toString(candidateFeatures.length == 0 ? 0 : candidateFeatures[0].length));
        byte[] payload = SharedGpuTensorSerde.buildScorePayload(state, candidateActionIds, candidateFeatures, candidateMask);
        long startNanos = System.nanoTime();
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_SCORE, headers, payload, SCORE_TIMEOUT_MS);
        metrics.recordInferenceLatencyMs((System.nanoTime() - startNanos) / 1_000_000L);
        if ("1".equals(response.headers.get("flush_leader"))) {
            int batchSize = parseIntHeader(response.headers, "host_batch_size", 0);
            boolean dueToFull = "full".equalsIgnoreCase(response.headers.getOrDefault("host_flush_reason", ""));
            metrics.recordInferBatchFlush(batchSize, dueToFull);
        }
        return SharedGpuTensorSerde.decodePredictionResult(response.payload, candidateActionIds.length);
    }

    @Override
    public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        if (trainingData == null || trainingData.isEmpty()) {
            return;
        }
        StateSequenceBuilder.TrainingData first = trainingData.get(0);
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("profile_id", profileId);
        headers.put("batch_size", Integer.toString(trainingData.size()));
        headers.put("seq_len", Integer.toString(first.state.getSequence().length));
        headers.put("d_model", Integer.toString(first.state.getSequence().length == 0 ? 0 : first.state.getSequence()[0].length));
        headers.put("max_candidates", Integer.toString(StateSequenceBuilder.TrainingData.MAX_CANDIDATES));
        headers.put("cand_feat_dim", Integer.toString(StateSequenceBuilder.TrainingData.CAND_FEAT_DIM));
        byte[] payload = SharedGpuTensorSerde.buildTrainPayload(trainingData, rewards);
        SharedGpuProtocol.ResponseFrame response = invoke(SharedGpuProtocol.OP_ENQUEUE_TRAIN, headers, payload, CONTROL_TIMEOUT_MS);
        trainQueueDepth = parseIntHeader(response.headers, "queue_depth", trainQueueDepth);
        droppedTrainEpisodes = parseIntHeader(response.headers, "dropped_train_episodes", (int) droppedTrainEpisodes);
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
            if (isConnected()) {
                try {
                    invoke(SharedGpuProtocol.OP_CLOSE_PROFILE, singletonProfileHeaders(), new byte[0], 5000);
                } catch (Exception ignored) {
                }
            }
        } finally {
            closeConnection(new IOException("Shared GPU client shutdown"));
            failPending(new IOException("Shared GPU client shutdown"));
        }
    }

    int getTrainQueueDepth() {
        return trainQueueDepth;
    }

    long getDroppedTrainEpisodes() {
        return droppedTrainEpisodes;
    }

    private SharedGpuProtocol.ResponseFrame invoke(int opcode, Map<String, String> headers, byte[] payload, int timeoutMs) {
        ensureReady();
        return invokeConnected(opcode, headers, payload, timeoutMs);
    }

    private SharedGpuProtocol.ResponseFrame invokeConnected(int opcode, Map<String, String> headers, byte[] payload, int timeoutMs) {
        try {
            long requestId = requestIdSeq.getAndIncrement();
            CompletableFuture<SharedGpuProtocol.ResponseFrame> future = new CompletableFuture<>();
            pending.put(requestId, future);
            OutboundRequest request = new OutboundRequest(opcode, requestId, headers, payload);
            if (!outbound.offer(request, Math.max(1000, timeoutMs), TimeUnit.MILLISECONDS)) {
                pending.remove(requestId);
                throw new IllegalStateException("Shared GPU outbound queue is full");
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
            throw new IllegalStateException("Timed out waiting for shared GPU response op=" + opcode, e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new IllegalStateException("Shared GPU request failed", cause);
        }
    }

    private void ensureReady() {
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

    private boolean isConnected() {
        Socket current = socket;
        return current != null && current.isConnected() && !current.isClosed();
    }

    private void connectLocked() {
        closeConnection(new IOException("Reconnecting shared GPU client"));
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
            logger.info("Connected shared GPU client profile=" + profileId + " endpoint=" + endpoint);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to connect to shared GPU host at " + endpoint, e);
        }
    }

    private void registerLocked() {
        Map<String, String> headers = buildRegisterHeaders();
        SharedGpuProtocol.ResponseFrame response = invokeConnected(SharedGpuProtocol.OP_REGISTER_PROFILE, headers, new byte[0], CONTROL_TIMEOUT_MS);
        if (response.status != SharedGpuProtocol.STATUS_OK) {
            throw new IllegalStateException(response.headers.getOrDefault("error", "Shared GPU profile registration failed"));
        }
        registered = true;
    }

    private void startThreads() {
        writerThread = new Thread(this::writerLoop, "SharedGpuWriter-" + profileId);
        writerThread.setDaemon(true);
        writerThread.start();
        readerThread = new Thread(this::readerLoop, "SharedGpuReader-" + profileId);
        readerThread.setDaemon(true);
        readerThread.start();
    }

    private void writerLoop() {
        while (!shutdown.get()) {
            try {
                OutboundRequest request = outbound.take();
                OutputStream currentOutput = output;
                if (currentOutput == null) {
                    throw new IOException("Shared GPU socket output closed");
                }
                SharedGpuProtocol.writeRequest(currentOutput, request.opcode, request.requestId, request.headers, request.payload);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                closeConnection(e instanceof IOException ? (IOException) e : new IOException(e));
                failPending(e);
                break;
            }
        }
    }

    private void readerLoop() {
        while (!shutdown.get()) {
            try {
                InputStream currentInput = input;
                if (currentInput == null) {
                    throw new IOException("Shared GPU socket input closed");
                }
                SharedGpuProtocol.ResponseFrame response = SharedGpuProtocol.readResponse(currentInput);
                CompletableFuture<SharedGpuProtocol.ResponseFrame> future = pending.remove(response.requestId);
                if (future != null) {
                    future.complete(response);
                }
            } catch (EOFException eof) {
                closeConnection(new IOException("Shared GPU host closed connection", eof));
                failPending(eof);
                break;
            } catch (Exception e) {
                closeConnection(e instanceof IOException ? (IOException) e : new IOException(e));
                failPending(e);
                break;
            }
        }
    }

    private void failPending(Throwable error) {
        List<Long> ids = new ArrayList<>(pending.keySet());
        for (Long id : ids) {
            CompletableFuture<SharedGpuProtocol.ResponseFrame> future = pending.remove(id);
            if (future != null) {
                future.completeExceptionally(error);
            }
        }
    }

    private void closeConnection(IOException reason) {
        Socket currentSocket = socket;
        socket = null;
        input = null;
        output = null;
        registered = false;
        if (currentSocket != null) {
            try {
                currentSocket.close();
            } catch (IOException ignored) {
            }
        }
        Thread currentReader = readerThread;
        readerThread = null;
        if (currentReader != null) {
            currentReader.interrupt();
        }
        Thread currentWriter = writerThread;
        writerThread = null;
        if (currentWriter != null) {
            currentWriter.interrupt();
        }
        while (!outbound.isEmpty()) {
            OutboundRequest dropped = outbound.poll();
            if (dropped != null) {
                CompletableFuture<SharedGpuProtocol.ResponseFrame> future = pending.remove(dropped.requestId);
                if (future != null) {
                    future.completeExceptionally(reason);
                }
            }
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

    private static String safe(String value, String fallback) {
        String trimmed = value == null ? "" : value.trim();
        return trimmed.isEmpty() ? fallback : trimmed;
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
