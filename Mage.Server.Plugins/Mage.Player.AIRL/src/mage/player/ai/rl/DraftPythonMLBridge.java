package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.net.ServerSocketFactory;
import javax.net.SocketFactory;

import py4j.ClientServer;

/**
 * Py4J bridge for the draft model Python process.
 *
 * Starts a separate Python process running draft_py4j_entry_point.py on
 * DRAFT_PY4J_PORT (default 25335) to keep the draft and game models isolated.
 */
public class DraftPythonMLBridge implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(DraftPythonMLBridge.class.getName());

    public static final int DEFAULT_DRAFT_PORT = 25335;
    private static final int MAX_CONNECTION_RETRIES = 20;
    private static final int CONNECTION_RETRY_DELAY_MS = 2000;
    private static final int PYTHON_STARTUP_WAIT_MS = 2000;

    private static volatile DraftPythonMLBridge instance;
    private static final Object lock = new Object();

    private final String projectRoot;
    private final String venvPath;
    private final String scriptPath;
    private final int py4jPort;

    private ClientServer clientServer;
    private DraftPythonEntryPoint entryPoint;
    private Process pythonProcess;
    private boolean initialized = false;
    private final Object py4jLock = new Object();

    private DraftPythonMLBridge() {
        this.py4jPort = EnvConfig.i32("DRAFT_PY4J_PORT", DEFAULT_DRAFT_PORT);
        this.projectRoot = resolveProjectRoot();
        this.venvPath = resolveVenvPath(projectRoot);
        this.scriptPath = projectRoot
                + "/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/draft_py4j_entry_point.py";
        try {
            init();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize DraftPythonMLBridge", e);
        }
    }

    public static DraftPythonMLBridge getInstance() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new DraftPythonMLBridge();
                }
            }
        }
        return instance;
    }

    private void init() throws Exception {
        setupVenvIfNeeded();
        startPythonProcess();
        connectToGateway();
        entryPoint.initializeDraftModel();
        initialized = true;
        logger.info("DraftPythonMLBridge initialized on port " + py4jPort);
    }

    private void setupVenvIfNeeded() {
        File venv = new File(venvPath);
        if (venv.exists()) {
            return;
        }
        try {
            logger.info("Creating Python venv at: " + venvPath);
            String python = findSystemPython();
            ProcessBuilder pb = new ProcessBuilder(python, "-m", "venv", venvPath);
            pb.redirectErrorStream(true);
            Process p = pb.start();
            drainProcess(p, "[VENV]");
            p.waitFor();
        } catch (Exception e) {
            logger.warning("Failed to create venv (will try system Python): " + e.getMessage());
        }
    }

    private void startPythonProcess() throws Exception {
        File script = new File(scriptPath);
        if (!script.exists()) {
            throw new RuntimeException("Draft Python script not found: " + scriptPath);
        }

        String python = getPythonExec();
        ProcessBuilder pb = new ProcessBuilder(
                python, scriptPath,
                "--port", String.valueOf(py4jPort)
        );
        pb.environment().put("PY4J_PORT", String.valueOf(py4jPort));
        pb.environment().put("MTG_AI_LOG_LEVEL",
                System.getenv().getOrDefault("MTG_AI_LOG_LEVEL", "WARNING"));

        // Pass draft model profile paths
        String profileName = EnvConfig.str("DRAFT_MODEL_PROFILE",
                EnvConfig.str("MODEL_PROFILE", "VintageCube-Draft"));
        pb.environment().put("DRAFT_MODEL_PROFILE", profileName);

        String modelsDir = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/"
                + profileName + "/models";
        pb.environment().put("DRAFT_MODELS_DIR", projectRoot + "/" + modelsDir);

        pb.redirectErrorStream(true);
        pb.directory(new File(projectRoot
                + "/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode"));
        pythonProcess = pb.start();

        Thread outputThread = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(pythonProcess.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    logger.fine("[DRAFT_PY] " + line);
                }
            } catch (Exception ignored) {
            }
        });
        outputThread.setDaemon(true);
        outputThread.start();

        // Brief wait for startup
        long t0 = System.currentTimeMillis();
        while (System.currentTimeMillis() - t0 < PYTHON_STARTUP_WAIT_MS) {
            try {
                int code = pythonProcess.exitValue();
                throw new RuntimeException("Draft Python process exited immediately with code: " + code);
            } catch (IllegalThreadStateException e) {
                break; // still running, good
            }
        }
    }

    private void connectToGateway() throws Exception {
        Exception last = null;
        for (int attempt = 0; attempt < MAX_CONNECTION_RETRIES; attempt++) {
            try {
                clientServer = new ClientServer(
                        0,
                        InetAddress.getByName("127.0.0.1"),
                        py4jPort,
                        InetAddress.getByName("127.0.0.1"),
                        0, 0,
                        ServerSocketFactory.getDefault(),
                        SocketFactory.getDefault(),
                        null, false, true
                );
                clientServer.startServer();
                entryPoint = (DraftPythonEntryPoint) clientServer
                        .getPythonServerEntryPoint(new Class[]{DraftPythonEntryPoint.class});
                logger.info("Connected to draft Python model on port " + py4jPort);
                return;
            } catch (Exception e) {
                last = e;
                if (clientServer != null) {
                    try { clientServer.shutdown(); } catch (Exception ignored) {}
                    clientServer = null;
                }
                if (attempt < MAX_CONNECTION_RETRIES - 1) {
                    Thread.sleep(CONNECTION_RETRY_DELAY_MS);
                }
            }
        }
        throw new RuntimeException("Could not connect to draft Python process after "
                + MAX_CONNECTION_RETRIES + " attempts", last);
    }

    // -----------------------------------------------------------------------
    // Public inference / training API
    // -----------------------------------------------------------------------

    /**
     * Score cards in the current pack using the pick head.
     * Returns float[numCands + 1]: softmax scores + value estimate.
     */
    public float[] scoreDraftPick(
            float[] stateFlat, int[] maskFlat, int[] tokenIds,
            float[] candFeats, int[] candIds,
            int numCands, int seqLen, int dimPerToken
    ) {
        synchronized (py4jLock) {
            byte[] result = entryPoint.scoreDraftPick(
                    toBytes(stateFlat), toBytes(maskFlat), toBytes(tokenIds),
                    toBytes(candFeats), toBytes(candIds),
                    numCands, seqLen, dimPerToken
            );
            return fromBytes(result, numCands + 1);
        }
    }

    /**
     * Score non-land pool cards using the construction head.
     * Returns float[numCands + 1]: scores + value estimate.
     */
    public float[] scoreDraftConstruction(
            float[] stateFlat, int[] maskFlat, int[] tokenIds,
            float[] candFeats, int[] candIds,
            int numCands, int seqLen, int dimPerToken
    ) {
        synchronized (py4jLock) {
            byte[] result = entryPoint.scoreDraftConstruction(
                    toBytes(stateFlat), toBytes(maskFlat), toBytes(tokenIds),
                    toBytes(candFeats), toBytes(candIds),
                    numCands, seqLen, dimPerToken
            );
            return fromBytes(result, numCands + 1);
        }
    }

    /**
     * Train the draft model on a batch of decisions from a single draft episode.
     */
    public void trainDraftBatch(
            byte[] stateBytes, byte[] maskBytes, byte[] tokenIdBytes,
            byte[] candFeatBytes, byte[] candIdBytes, byte[] candMaskBytes,
            byte[] chosenIdxBytes, byte[] oldLogpBytes, byte[] oldValueBytes,
            byte[] rewardBytes, byte[] doneBytes, byte[] headIdxBytes,
            int numSteps, int seqLen, int dimPerToken, int maxCands
    ) {
        synchronized (py4jLock) {
            entryPoint.trainDraftBatch(
                    stateBytes, maskBytes, tokenIdBytes,
                    candFeatBytes, candIdBytes, candMaskBytes,
                    chosenIdxBytes, oldLogpBytes, oldValueBytes,
                    rewardBytes, doneBytes, headIdxBytes,
                    numSteps, seqLen, dimPerToken, maxCands
            );
        }
    }

    public Map<String, Integer> getTrainingStats() {
        synchronized (py4jLock) {
            return entryPoint.getDraftTrainingStats();
        }
    }

    public Map<String, Object> getLossMetrics() {
        synchronized (py4jLock) {
            return entryPoint.getDraftLossMetrics();
        }
    }

    public void saveDraftModel(String path) {
        synchronized (py4jLock) {
            try {
                Files.createDirectories(Paths.get(path).getParent());
            } catch (Exception e) {
                logger.warning("Failed to create dirs for draft model save: " + e.getMessage());
            }
            entryPoint.saveDraftModel(path);
        }
    }

    public void loadDraftModel(String path) {
        synchronized (py4jLock) {
            entryPoint.loadDraftModel(path);
        }
    }

    // -----------------------------------------------------------------------
    // Byte conversion helpers
    // -----------------------------------------------------------------------

    public static byte[] toBytes(float[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : arr) buf.putFloat(v);
        return buf.array();
    }

    public static byte[] toBytes(int[] arr) {
        ByteBuffer buf = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : arr) buf.putInt(v);
        return buf.array();
    }

    private static float[] fromBytes(byte[] bytes, int count) {
        float[] out = new float[count];
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count && buf.remaining() >= 4; i++) {
            out[i] = buf.getFloat();
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    private static String resolveProjectRoot() {
        // Walk up from cwd until pom.xml is found or we reach filesystem root
        File dir = new File(System.getProperty("user.dir")).getAbsoluteFile();
        while (dir != null) {
            if (new File(dir, "pom.xml").exists()) {
                return dir.getAbsolutePath();
            }
            dir = dir.getParentFile();
        }
        return System.getProperty("user.dir");
    }

    private String resolveVenvPath(String root) {
        String envPath = System.getenv("VENV_PATH");
        if (envPath != null && !envPath.isBlank()) return envPath;
        return root + "/venv";
    }

    private String getPythonExec() {
        // Windows
        File win = new File(venvPath, "Scripts/python.exe");
        if (win.exists()) return win.getAbsolutePath();
        // Unix
        File unix = new File(venvPath, "bin/python");
        if (unix.exists()) return unix.getAbsolutePath();
        return findSystemPython();
    }

    private static String findSystemPython() {
        // Prefer python3 on Unix
        for (String name : new String[]{"python3", "python"}) {
            try {
                Process p = new ProcessBuilder(name, "--version").start();
                p.waitFor();
                if (p.exitValue() == 0) return name;
            } catch (Exception ignored) {}
        }
        return "python";
    }

    private static void drainProcess(Process p, String prefix) {
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(p.getInputStream()))) {
            String line;
            while ((line = r.readLine()) != null) {
                logger.fine(prefix + " " + line);
            }
        } catch (Exception ignored) {}
    }

    @Override
    public void close() {
        if (clientServer != null) {
            try { clientServer.shutdown(); } catch (Exception ignored) {}
        }
        if (pythonProcess != null && pythonProcess.isAlive()) {
            pythonProcess.destroy();
        }
    }
}
