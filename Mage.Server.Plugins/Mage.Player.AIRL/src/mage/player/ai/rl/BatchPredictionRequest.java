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

    public static synchronized BatchPredictionRequest getInstance(int activeGameRunners,
                                                                  long timeout,
                                                                  TimeUnit unit) {
        if (instance == null) {
            instance = new BatchPredictionRequest(activeGameRunners, timeout, unit);
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

    private BatchPredictionRequest(int activeGameRunners, long timeout, TimeUnit unit) {
        this.activeGameRunners = activeGameRunners;
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

        // ------------------------------------------------------------------
        // Build input tensors ---------------------------------------------
        // ------------------------------------------------------------------
        int D      = Math.toIntExact(batch[0].sequence.size(1)); // sequence shape [1, D, L]
        int L      = Math.toIntExact(batch[0].sequence.size(2));
        INDArray seqBatch  = Nd4j.create(new int[]{filled, D, L}, 'c');
        INDArray maskBatch = Nd4j.create(new int[]{filled, L},   'c');

        for (int i = 0; i < filled; i++) {
            seqBatch.putSlice(i, batch[i].sequence.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()));
            maskBatch.putRow(i, batch[i].mask);
        }

        // ------------------------------------------------------------------
        // Forward pass -----------------------------------------------------
        // ------------------------------------------------------------------
        INDArray preds;
        RLModel shared = RLTrainer.sharedModel;
        if (shared == null) {
            throw new IllegalStateException("SharedModel is null");
        } else {
            preds = shared.getNetwork().batchCastLogits(seqBatch, maskBatch);
        }

        // ------------------------------------------------------------------
        // Fulfil promises ---------------------------------------------------
        // ------------------------------------------------------------------
        totalPredictions += filled;
        for (int i = 0; i < filled; i++) {
            batch[i].fulfil(preds.getRow(i, true));
        }

        // ------------------------------------------------------------------
        // Throughput log ----------------------------------------------------
        // ------------------------------------------------------------------
        long elapsed = System.currentTimeMillis() - startWall;
        if (elapsed > 0) {
            double pps = (totalPredictions * 1000.0) / elapsed;
            log.warn(String.format("Average predictions/sec: %.2f", pps));
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
