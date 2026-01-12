package mage.player.ai.rl;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;

import org.apache.log4j.Logger;

/**
 * GPU/CPU learner process. Polls a directory for serialized EpisodeData files,
 * aggregates them into batches, calls sharedModel.train, then deletes files.
 */
public class Learner {

    private static final Logger logger = Logger.getLogger(Learner.class);

    private final Path trajDir;
    private int maxSamplesPerBatch;  // Not final - can be updated by GPU detection
    private final int pollSeconds;
    private final int checkpointEvery;

    private int updateCounter = 0;

    public Learner(Path trajDir, int maxSamplesPerBatch, int pollSeconds, int checkpointEvery) {
        this.trajDir = trajDir;
        this.maxSamplesPerBatch = maxSamplesPerBatch;
        this.pollSeconds = pollSeconds;
        this.checkpointEvery = checkpointEvery;

        // Try to get optimal batch size from GPU memory detection
        try {
            int optimalBatchSize = RLTrainer.sharedModel.getOptimalBatchSize();
            if (optimalBatchSize > 0) {
                this.maxSamplesPerBatch = optimalBatchSize;
                logger.info("Using GPU-optimized batch size: " + optimalBatchSize + " samples");
                RLTrainer.metrics.recordOptimalBatchSize(optimalBatchSize);
            } else {
                logger.info("Using configured batch size: " + maxSamplesPerBatch + " samples");
            }
        } catch (Exception e) {
            logger.warn("Failed to get optimal batch size from GPU, using configured value: " + e.getMessage());
        }
    }

    public void run() throws IOException, ClassNotFoundException {
        Files.createDirectories(trajDir);
        while (true) {
            List<EpisodeData> episodes = loadEpisodeBatchBySampleCount();
            if (episodes.isEmpty()) {
                try {
                    TimeUnit.SECONDS.sleep(pollSeconds);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
                continue;
            }

            List<StateSequenceBuilder.TrainingData> allData = new ArrayList<>();
            List<Double> allReturns = new ArrayList<>();
            int consumed = episodes.size();

            for (EpisodeData ep : episodes) {
                allData.addAll(ep.trajectory);
                allReturns.addAll(ep.discountedReturns);
            }

            if (!allData.isEmpty()) {
                // Record metrics before training
                RLTrainer.metrics.recordSamplesProcessed(allData.size());

                RLTrainer.sharedModel.train(allData, allReturns);
                updateCounter++;

                // Record training metrics (assuming we can get loss somehow)
                // For now, we'll record batch size and update count
                RLTrainer.metrics.recordTrainingBatch(allData.size(), 0.0); // Loss would need to come from Python

                logger.info("Learner update " + updateCounter + " – consumed " + consumed + " episodes (" + allData.size() + " samples)");

                if (updateCounter % checkpointEvery == 0) {
                    try {
                        RLTrainer.sharedModel.saveModel(RLTrainer.MODEL_FILE_PATH);
                        logger.info("Checkpoint saved after update " + updateCounter);
                        RLTrainer.metrics.recordCheckpointSaved();
                    } catch (Exception e) {
                        logger.error("Failed to save checkpoint", e);
                        RLTrainer.metrics.recordError("checkpoint");
                    }
                }
                if (updateCounter % RLTrainer.EVAL_EVERY == 0) {
                    double winRate = RLTrainer.runEvaluation(10); // Run 10 evaluation games
                    logEvaluationResult(updateCounter, winRate);
                    RLTrainer.metrics.recordWinRate(winRate);
                }
            }
        }
    }

    private void logEvaluationResult(int updateCounter, double winRate) {
        try {
            Path statsPath = trajDir.resolve("evaluation_stats.csv");
            boolean writeHeader = !Files.exists(statsPath);
            StringBuilder sb = new StringBuilder();
            if (writeHeader) {
                sb.append("update_step,win_rate\n");
            }
            sb.append(updateCounter).append(',').append(String.format("%.4f", winRate)).append('\n');
            Files.write(statsPath, sb.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8),
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (IOException e) {
            logger.warn("Failed to write evaluation stats CSV", e);
        }
    }

    /**
     * Dynamic batching: Load episode data until total samples reach
     * maxSamplesPerBatch This eliminates race conditions by reading files only
     * once.
     */
    private List<EpisodeData> loadEpisodeBatchBySampleCount() throws IOException, ClassNotFoundException {
        List<EpisodeData> batch = new ArrayList<>();
        List<Path> filesToDelete = new ArrayList<>();
        int totalSamples = 0;

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(trajDir, "*.ser.gz")) {
            for (Path p : ds) {
                try (ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(p.toFile())))) {
                    EpisodeData ep = (EpisodeData) ois.readObject();
                    int episodeSamples = ep.trajectory.size();

                    // Add episode to batch if it doesn't exceed limit
                    if (totalSamples + episodeSamples <= maxSamplesPerBatch) {
                        batch.add(ep);
                        filesToDelete.add(p);
                        totalSamples += episodeSamples;
                    } else if (batch.isEmpty()) {
                        // Always include at least one episode, even if it's large
                        batch.add(ep);
                        filesToDelete.add(p);
                        totalSamples += episodeSamples;
                        break;
                    } else {
                        // Stop adding episodes once we'd exceed the limit
                        break;
                    }
                } catch (IOException e) {
                    logger.warn("Failed to read episode file " + p + ", skipping: " + e.getMessage());
                }
            }
        }

        // Delete processed files after successful reading
        for (Path p : filesToDelete) {
            try {
                Files.delete(p);
            } catch (IOException e) {
                logger.warn("Failed to delete processed file " + p, e);
            }
        }

        logger.info("Dynamic batch: " + batch.size() + " episodes, " + totalSamples + " samples (limit: " + maxSamplesPerBatch + ")");
        return batch;
    }
}
