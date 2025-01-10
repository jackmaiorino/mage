package mage.player.ai.rl;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BatchPredictionRequest {
    private static BatchPredictionRequest instance;

    public static synchronized BatchPredictionRequest getInstance(int activeGameRunners, long timeout, TimeUnit timeUnit) {
        if (instance == null) {
            instance = new BatchPredictionRequest(activeGameRunners, timeout, timeUnit);
        }
        return instance;
    }

    private final BlockingQueue<Request> requestQueue;
    private int activeGameRunners;
    private final long timeout;
    private final TimeUnit timeUnit;
    private long lastProcessTime = System.currentTimeMillis();
    private long totalPredictions = 0;
    private long startTime = System.currentTimeMillis();
    private final INDArray inputBatch;
    private static final Logger logger = Logger.getLogger(BatchPredictionRequest.class);

    public BatchPredictionRequest(int activeGameRunners, long timeout, TimeUnit timeUnit) {
        this.activeGameRunners = activeGameRunners;
        this.timeout = timeout;
        this.timeUnit = timeUnit;
        this.requestQueue = new LinkedBlockingQueue<>();
        this.inputBatch = Nd4j.create(RLTrainer.BATCH_SIZE, RLState.STATE_VECTOR_SIZE, 'c').assign(0);
        startBatchProcessor();
    }

    public synchronized void decrementActiveGameRunners() {
        this.activeGameRunners--;
    }

    public synchronized void incrementActiveGameRunners() {
        this.activeGameRunners++;
    }

    public INDArray predict(INDArray state) throws InterruptedException {
        Request request = new Request(state);
        requestQueue.put(request);
        return request.getPrediction();
    }

    private void startBatchProcessor() {
        Thread batchProcessor = new Thread(() -> {
            while (true) {
                try {
                    processBatch();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });

        batchProcessor.setName("BatchPredictManager");

        batchProcessor.setPriority(Thread.MAX_PRIORITY);

        batchProcessor.start();
    }

    private void processBatch() throws InterruptedException {
        if (activeGameRunners <= 0) {
            // If activeGameRunners is 0, wait for the timeout duration before retrying
            Thread.sleep(timeUnit.toMillis(1000));
            return;
        }

        long currentTime = System.currentTimeMillis();
        logger.info("Time since last processBatch: " + (currentTime - lastProcessTime) + " milliseconds");
        lastProcessTime = currentTime;

        // This allows the CPU to work while GPU is working
        int batchSize;
        if (activeGameRunners == 1) {
            batchSize = activeGameRunners;
        } else{
            batchSize = activeGameRunners / 2;
        }
        //int currentactiveGameRunners = Math.min(requestQueue.size(), RLTrainer.BATCH_SIZE);
        Request[] batch = new Request[batchSize];
        int count = 0;
        long batchStartTime = System.currentTimeMillis();

        while (count < batchSize && (System.currentTimeMillis() - batchStartTime) < timeUnit.toMillis(timeout)) {
            Request request = requestQueue.poll(timeout, timeUnit);
            if (request != null) {
                batch[count++] = request;
            }
        }

        logger.info("Time to fill batch: " + (System.currentTimeMillis() - batchStartTime) + " milliseconds");
        logger.warn("Batch size: " + count);

        if (count > 0) {
            totalPredictions += count;
            INDArray[] states = new INDArray[count];
            for (int i = 0; i < count; i++) {
                states[i] = batch[i].state;
            }
            INDArray predictions = batchPredict(states);
            
            for (int i = 0; i < count; i++) {
                batch[i].setPrediction(predictions.getRow(i, true));
            }
        }

        // Calculate and print average predictions per second
        long totalTimeElapsed = System.currentTimeMillis() - startTime;
        double averagePredictionsPerSecond = (totalPredictions * 1000.0) / totalTimeElapsed;
        logger.warn("Average predictions per second: " + String.format("%.2f", averagePredictionsPerSecond));
    }

    private INDArray batchPredict(INDArray[] states) {
        // Create an input batch with the correct shape on the GPU
        for (int i = 0; i < states.length; i++) {
            inputBatch.putRow(i, states[i]);
        }

        // Start timing the prediction
//        long predictionStartTime = System.currentTimeMillis();

        // Get predictions from the neural network
        INDArray predictions = RLTrainer.sharedModel.getNetwork().network.output(inputBatch);

        // End timing the prediction
//        long predictionEndTime = System.currentTimeMillis();
//        logger.warn("Prediction time: " + (predictionEndTime - predictionStartTime) + " milliseconds");

        return predictions;
    }

    private static class Request {
        private final INDArray state;
        private final CountDownLatch latch;
        private INDArray prediction;

        public Request(INDArray state) {
            this.state = state;
            this.latch = new CountDownLatch(1);
        }

        public INDArray getPrediction() throws InterruptedException {
            latch.await();
            return prediction;
        }

        public void setPrediction(INDArray prediction) {
            this.prediction = prediction;
            latch.countDown();
        }
    }
}
