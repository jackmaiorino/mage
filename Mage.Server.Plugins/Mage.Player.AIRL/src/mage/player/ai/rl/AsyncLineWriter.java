package mage.player.ai.rl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;

/**
 * Single-threaded async appender for low-value CSV/log writes in hot paths.
 */
final class AsyncLineWriter implements AutoCloseable {

    private static final class WriteTask {
        final Path path;
        final String header;
        final String line;

        WriteTask(Path path, String header, String line) {
            this.path = path;
            this.header = header;
            this.line = line;
        }
    }

    private final Logger logger;
    private final BlockingQueue<WriteTask> queue = new LinkedBlockingQueue<>();
    private final Thread thread;
    private volatile boolean running = true;

    AsyncLineWriter(String threadName, Logger logger) {
        this.logger = logger;
        this.thread = new Thread(this::runLoop, threadName);
        this.thread.setDaemon(true);
        this.thread.start();
    }

    void append(Path path, String headerIfMissing, String line) {
        if (!running || path == null || line == null) {
            return;
        }
        queue.offer(new WriteTask(path, headerIfMissing == null ? "" : headerIfMissing, line));
    }

    private void runLoop() {
        while (running || !queue.isEmpty()) {
            try {
                WriteTask task = queue.poll(250, TimeUnit.MILLISECONDS);
                if (task == null) {
                    continue;
                }
                if (task.path.getParent() != null) {
                    Files.createDirectories(task.path.getParent());
                }
                boolean writeHeader = !task.header.isEmpty() && !Files.exists(task.path);
                if (writeHeader) {
                    Files.write(
                            task.path,
                            task.header.getBytes(StandardCharsets.UTF_8),
                            StandardOpenOption.CREATE,
                            StandardOpenOption.APPEND
                    );
                }
                Files.write(
                        task.path,
                        task.line.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.APPEND
                );
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (IOException e) {
                if (logger != null) {
                    logger.warn("AsyncLineWriter failed to append", e);
                }
            }
        }
    }

    @Override
    public void close() {
        running = false;
        thread.interrupt();
        try {
            thread.join(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
