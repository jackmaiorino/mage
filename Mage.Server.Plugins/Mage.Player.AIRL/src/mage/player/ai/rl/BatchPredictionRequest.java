package mage.player.ai.rl;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class BatchPredictionRequest {
    private static BatchPredictionRequest instance;

    public static synchronized BatchPredictionRequest getInstance(int batchSize, long timeout, TimeUnit timeUnit) {
        if (instance == null) {
            instance = new BatchPredictionRequest(batchSize, timeout, timeUnit);
        }
        return instance;
    }

    private final BlockingQueue<Request> requestQueue;
    private int batchSize;
    private final long timeout;
    private final TimeUnit timeUnit;

    public BatchPredictionRequest(int batchSize, long timeout, TimeUnit timeUnit) {
        this.batchSize = batchSize;
        this.timeout = timeout;
        this.timeUnit = timeUnit;
        this.requestQueue = new LinkedBlockingQueue<>();
        startBatchProcessor();
    }

    public synchronized void decrementBatchSize() {
        this.batchSize--;
    }

    public synchronized void incrementBatchSize() {
        this.batchSize++;
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
        batchProcessor.start();
    }

    private void processBatch() throws InterruptedException {
        Request[] batch = new Request[batchSize];
        int count = 0;
        long startTime = System.currentTimeMillis();

        while (count < batchSize && (System.currentTimeMillis() - startTime) < timeUnit.toMillis(timeout)) {
            Request request = requestQueue.poll(timeout, timeUnit);
            if (request != null) {
                batch[count++] = request;
            }
        }

        if (count > 0) {
            INDArray[] states = new INDArray[count];
            for (int i = 0; i < count; i++) {
                states[i] = batch[i].state;
            }
            INDArray predictions = batchPredict(states);
            

            for (int i = 0; i < count; i++) {
                batch[i].setPrediction(predictions.getRow(i));
            }
        }
    }

    private INDArray batchPredict(INDArray[] states) {
        // Create an input batch with the correct shape
        INDArray inputBatch = Nd4j.create(states.length, RLState.STATE_VECTOR_SIZE);
        for (int i = 0; i < states.length; i++) {
            inputBatch.putRow(i, states[i]);
        }

        // Get predictions from the neural network
        INDArray predictions = RLTrainer.sharedModel.getNetwork().network.output(inputBatch);
        return predictions.reshape(states.length, RLModel.OUTPUT_SIZE);
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
