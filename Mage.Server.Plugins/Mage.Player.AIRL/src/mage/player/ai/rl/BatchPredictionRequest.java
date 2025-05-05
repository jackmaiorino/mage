package mage.player.ai.rl;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * BatchPredictionRequest for the new
 * {@code <sequence, mask>} representation produced by
 * {@link com.mtg.rl.StateSequenceBuilder}.  It batches variable‑length
 * sequences by simple concatenation in the batch dimension; all sequences
 * are already padded to the same {@code maxLen} when they enter.
 */
public class BatchPredictionRequest {

    // ---------------------------------------------------------------------
    // Singleton boiler‑plate ------------------------------------------------
    // ---------------------------------------------------------------------
    private static BatchPredictionRequest instance;

    public static synchronized BatchPredictionRequest getInstance(
                                                                long timeout,
                                                                TimeUnit unit) {
        if (instance == null) {
            instance = new BatchPredictionRequest(timeout, unit);
        }
        return instance;
    }

    // ---------------------------------------------------------------------
    // Instance state -------------------------------------------------------
    // ---------------------------------------------------------------------
    private final BlockingQueue<Request> queue;
    private int                          activeGameRunners;
    private final long                   timeoutMs;
    private final Logger                 log = Logger.getLogger(BatchPredictionRequest.class);

    private long totalPredictions;
    private long startWall;

    private BatchPredictionRequest(long timeout, TimeUnit unit) {
        this.activeGameRunners = 0;
        this.timeoutMs        = unit.toMillis(timeout);
        this.queue            = new LinkedBlockingQueue<Request>();
        this.startWall        = System.currentTimeMillis();
        startBatchThread();
    }

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------
    public void incrementActiveGameRunners() { synchronized (this) { activeGameRunners++; } }
    public void decrementActiveGameRunners() { synchronized (this) { activeGameRunners--; } }

    /**
     * Enqueue a prediction request consisting of one padded sequence + mask.
     */
    public INDArray predict(INDArray sequence, INDArray mask) throws InterruptedException {
        Request req = new Request(sequence, mask);
        queue.put(req);
        return req.await();
    }

    // ---------------------------------------------------------------------
    // Internal batch loop --------------------------------------------------
    // ---------------------------------------------------------------------
    private void startBatchThread() {
        Thread t = new Thread(new Runnable() {
            public void run() {
                while (!Thread.currentThread().isInterrupted()) {
                    try { processBatch(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                }
            }
        }, "BatchPredictManager");
        t.setPriority(Thread.MAX_PRIORITY);
        t.start();
    }

    private void processBatch() throws InterruptedException {
        if (activeGameRunners == 0) {
            Thread.sleep(1000L);
            return;
        }

        int batchCap = Math.max(1, activeGameRunners / 2);

        Request[] batch = new Request[batchCap];
        int filled = 0;
        long startFill = System.currentTimeMillis();
        while (filled < batchCap && (System.currentTimeMillis() - startFill) < timeoutMs) {
            Request r = queue.poll(timeoutMs, TimeUnit.MILLISECONDS);
            if (r != null) { batch[filled++] = r; }
        }

        if (filled == 0) return;

        // Get shared model
        RLModel shared = RLTrainer.sharedModel;
        if (shared == null) {
            throw new IllegalStateException("SharedModel is null");
        }

        // Build batched input
        int D = Math.toIntExact(batch[0].sequence.size(1)); // sequence shape [1, D, L]
        int L = Math.toIntExact(batch[0].sequence.size(2));
        INDArray seqBatch = Nd4j.create(new int[]{filled, D, L}, 'c');
        INDArray maskBatch = Nd4j.create(new int[]{filled, L}, 'c');

        for (int i = 0; i < filled; i++) {
            seqBatch.putSlice(i, batch[i].sequence.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()));
            maskBatch.putRow(i, batch[i].mask);
        }

        // Do batch prediction using the specialized batch method
        try {
            System.out.println("Num gamerunners: " + activeGameRunners);
            INDArray preds = shared.getNetwork().batchCastLogits(seqBatch, maskBatch);
            
            // Apply masking based on number of available options
            for (int i = 0; i < filled; i++) {
                // Get the number of available options from the state
                int numOptions = 0;
                // Extract numOptions from the sequence (it's encoded in the first element of the options token)
                if (seqBatch.size(1) > 0) {  // Check if we have at least one feature
                    numOptions = (int)(seqBatch.getDouble(i, 0, 0) * RLModel.CAST_OPTIONS);
                }
                
                // Create mask for valid options
                INDArray optionMask = Nd4j.zeros(RLModel.CAST_OPTIONS);
                for (int j = 0; j < Math.min(numOptions, RLModel.CAST_OPTIONS); j++) {
                    optionMask.putScalar(j, 1.0);
                }
                
                // Apply mask to predictions for this entry
                INDArray maskedPredictions = preds.getRow(i).mul(optionMask);
                
                // Normalize the masked predictions
                double sum = maskedPredictions.sumNumber().doubleValue();
                if (sum > 0) {
                    maskedPredictions = maskedPredictions.div(sum);
                }
                
                // Store the masked and normalized predictions
                batch[i].fulfil(maskedPredictions);
            }
        } catch (Exception e) {
            throw new RuntimeException("Error during batch prediction", e);
        }
    }

    // ---------------------------------------------------------------------
    // Inner class ---------------------------------------------------------
    // ---------------------------------------------------------------------
    private static final class Request {
        private final INDArray sequence;
        private final INDArray mask;
        private final CountDownLatch latch = new CountDownLatch(1);
        private INDArray result;

        Request(INDArray sequence, INDArray mask) {
            this.sequence = sequence;
            this.mask     = mask;
        }

        INDArray await() throws InterruptedException {
            latch.await();
            return result;
        }

        void fulfil(INDArray prediction) {
            this.result = prediction;
            latch.countDown();
        }
    }
}
