package mage.util;

import java.awt.*;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by IGOUDT on 5-9-2016.
 */
public final class RandomUtil {

    private static final CopyableRandom random = new CopyableRandom(); // thread safe with seed support
    private static final Random directTraceRandom = new DirectTraceRandom(random);
    private static final AtomicLong consumptionCount = new AtomicLong(0L);
    private static final AtomicLong directGetRandomAccessCount = new AtomicLong(0L);
    private static final AtomicLong directGetRandomConsumptionCount = new AtomicLong(0L);
    private static final AtomicLong directTraceSeq = new AtomicLong(0L);
    private static final AtomicLong wrapperTraceSeq = new AtomicLong(0L);
    private static final ThreadLocal<Random> isolatedRandom = new ThreadLocal<>();
    private static final ThreadLocal<String> wrapperTraceSourceDecisionOrdinal = new ThreadLocal<>();
    private static final ThreadLocal<String> wrapperTraceSourceName = new ThreadLocal<>();
    private static final Object DIRECT_TRACE_LOCK = new Object();
    private static final Object WRAPPER_TRACE_LOCK = new Object();
    private static final boolean DIRECT_TRACE_ENABLED = settingFlag("EVAL_RANDOM_UTIL_DIRECT_TRACE_JSON")
            || settingFlag("EVAL_RANDOM_UTIL_DIRECT_TRACE");
    private static final String DIRECT_TRACE_FILE = setting("EVAL_RANDOM_UTIL_DIRECT_TRACE_FILE");
    private static final boolean WRAPPER_TRACE_ENABLED = settingFlag("EVAL_RANDOM_UTIL_WRAPPER_TRACE_JSON")
            || settingFlag("EVAL_RANDOM_UTIL_WRAPPER_TRACE");
    private static final String WRAPPER_TRACE_FILE = setting("EVAL_RANDOM_UTIL_WRAPPER_TRACE_FILE");

    private RandomUtil() {
    }

    public static Random getRandom() {
        Random active = activeRandom();
        if (active == random && DIRECT_TRACE_ENABLED) {
            directGetRandomAccessCount.incrementAndGet();
            return directTraceRandom;
        }
        return active;
    }

    public static RandomIsolation isolateThreadLocalRandom(long seed) {
        Random previous = isolatedRandom.get();
        isolatedRandom.set(new Random(seed));
        return new RandomIsolation(previous);
    }

    public static WrapperTraceContext withWrapperTraceContext(String sourceDecisionOrdinal, String sourceName) {
        return new WrapperTraceContext(
                wrapperTraceSourceDecisionOrdinal.get(),
                wrapperTraceSourceName.get(),
                sourceDecisionOrdinal,
                sourceName);
    }

    // Counts calls through RandomUtil wrapper methods. Direct getRandom() consumers
    // are counted separately only when the validation-only direct trace flag is on.
    public static long getConsumptionCount() {
        return consumptionCount.get();
    }

    public static void advanceGlobalToConsumptionCount(long targetCount) {
        while (consumptionCount.get() < targetCount) {
            long before = consumptionCount.get();
            int result = random.nextInt();
            long after = consumptionCount.incrementAndGet();
            recordWrapperCall("advanceGlobalToConsumptionCount", "target=" + targetCount,
                    before, after, true, String.valueOf(result));
        }
    }

    public static long getDirectGetRandomAccessCount() {
        return directGetRandomAccessCount.get();
    }

    public static long getDirectGetRandomConsumptionCount() {
        return directGetRandomConsumptionCount.get();
    }

    private static Random activeRandom() {
        Random threadRandom = isolatedRandom.get();
        return threadRandom == null ? random : threadRandom;
    }

    public static int nextInt() {
        Random active = activeRandom();
        boolean counted = active == random;
        long before = consumptionCount.get();
        if (active == random) {
            consumptionCount.incrementAndGet();
        }
        int result = active.nextInt();
        recordWrapperCall("nextInt", "", before, consumptionCount.get(), counted, String.valueOf(result));
        return result;
    }

    public static int nextInt(int max) {
        Random active = activeRandom();
        boolean counted = active == random;
        long before = consumptionCount.get();
        if (active == random) {
            consumptionCount.incrementAndGet();
        }
        int result = active.nextInt(max);
        recordWrapperCall("nextInt", "bound=" + max, before, consumptionCount.get(), counted, String.valueOf(result));
        return result;
    }

    public static boolean nextBoolean() {
        Random active = activeRandom();
        boolean counted = active == random;
        long before = consumptionCount.get();
        if (active == random) {
            consumptionCount.incrementAndGet();
        }
        boolean result = active.nextBoolean();
        recordWrapperCall("nextBoolean", "", before, consumptionCount.get(), counted, String.valueOf(result));
        return result;
    }

    public static double nextDouble() {
        Random active = activeRandom();
        boolean counted = active == random;
        long before = consumptionCount.get();
        if (active == random) {
            consumptionCount.incrementAndGet();
        }
        double result = active.nextDouble();
        recordWrapperCall("nextDouble", "", before, consumptionCount.get(), counted, String.valueOf(result));
        return result;
    }

    public static Color nextColor() {
        return new Color(RandomUtil.nextInt(256), RandomUtil.nextInt(256), RandomUtil.nextInt(256));
    }

    public static void setSeed(long newSeed) {
        Random active = isolatedRandom.get();
        if (active != null) {
            active.setSeed(newSeed);
        } else {
            random.setSeed(newSeed);
            consumptionCount.set(0L);
            directGetRandomAccessCount.set(0L);
            directGetRandomConsumptionCount.set(0L);
            directTraceSeq.set(0L);
            wrapperTraceSeq.set(0L);
        }
    }

    public static State captureState() {
        return new State(
                random.captureState(),
                consumptionCount.get(),
                directGetRandomAccessCount.get(),
                directGetRandomConsumptionCount.get(),
                directTraceSeq.get(),
                wrapperTraceSeq.get(),
                System.getProperty("xmage.replay.random_util_seed", ""));
    }

    public static void restoreState(State state) {
        if (state == null) {
            return;
        }
        random.restoreState(state.randomState);
        consumptionCount.set(state.consumptionCount);
        directGetRandomAccessCount.set(state.directGetRandomAccessCount);
        directGetRandomConsumptionCount.set(state.directGetRandomConsumptionCount);
        directTraceSeq.set(state.directTraceSeq);
        wrapperTraceSeq.set(state.wrapperTraceSeq);
        if (state.replayRandomUtilSeed == null || state.replayRandomUtilSeed.isEmpty()) {
            System.clearProperty("xmage.replay.random_util_seed");
        } else {
            System.setProperty("xmage.replay.random_util_seed", state.replayRandomUtilSeed);
        }
    }

    public static <T> T randomFromCollection(Collection<T> collection) {
        if (collection.size() < 2) {
            return collection.stream().findFirst().orElse(null);
        }
        int rand = nextInt(collection.size());
        int count = 0;
        for (T current : collection) {
            if (count == rand) {
                return current;
            }
            count++;
        }
        return null;
    }

    private static boolean settingFlag(String key) {
        String value = setting(key);
        return "1".equals(value) || "true".equalsIgnoreCase(value) || "yes".equalsIgnoreCase(value);
    }

    private static String setting(String key) {
        String value = System.getenv(key);
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty(key);
        }
        return value == null ? "" : value.trim();
    }

    private static long recordDirectGetRandomConsumption(String method, String detail) {
        long count = directGetRandomConsumptionCount.incrementAndGet();
        if (!DIRECT_TRACE_ENABLED || DIRECT_TRACE_FILE.isEmpty()) {
            return count;
        }
        try {
            StringBuilder sb = new StringBuilder(512);
            sb.append('{');
            appendJsonString(sb, "event", "direct_get_random");
            sb.append(',');
            appendJsonNumber(sb, "seq", directTraceSeq.incrementAndGet());
            sb.append(',');
            appendJsonString(sb, "method", method);
            sb.append(',');
            appendJsonString(sb, "detail", detail == null ? "" : detail);
            sb.append(',');
            appendJsonNumber(sb, "wrapper_consumption_count", consumptionCount.get());
            sb.append(',');
            appendJsonNumber(sb, "direct_getrandom_access_count", directGetRandomAccessCount.get());
            sb.append(',');
            appendJsonNumber(sb, "direct_getrandom_consumption_count", count);
            sb.append(',');
            appendJsonString(sb, "scenario", System.getProperty("xmage.replay.scenario", ""));
            sb.append(',');
            appendJsonString(sb, "seed", System.getProperty("xmage.replay.seed", ""));
            sb.append(',');
            appendJsonString(sb, "random_util_seed", System.getProperty("xmage.replay.random_util_seed", ""));
            sb.append(',');
            appendJsonString(sb, "scope", System.getProperty("xmage.replay.scope", ""));
            sb.append(',');
            appendJsonString(sb, "caller", directTraceCaller());
            sb.append('}');
            Path path = Paths.get(DIRECT_TRACE_FILE);
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            byte[] bytes = ("RANDOM_UTIL_DIRECT_JSON: " + sb.toString() + System.lineSeparator())
                    .getBytes(StandardCharsets.UTF_8);
            synchronized (DIRECT_TRACE_LOCK) {
                Files.write(path, bytes, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                        StandardOpenOption.APPEND);
            }
        } catch (Throwable ignored) {
            // Diagnostics must never affect gameplay.
        }
        return count;
    }

    private static void recordWrapperCall(
            String method,
            String detail,
            long before,
            long after,
            boolean countedGlobal,
            String result
    ) {
        if (!WRAPPER_TRACE_ENABLED || WRAPPER_TRACE_FILE.isEmpty()) {
            return;
        }
        try {
            StringBuilder sb = new StringBuilder(768);
            sb.append('{');
            appendJsonString(sb, "event", "wrapper_random_util");
            sb.append(',');
            appendJsonNumber(sb, "seq", wrapperTraceSeq.incrementAndGet());
            sb.append(',');
            appendJsonString(sb, "method", method);
            sb.append(',');
            appendJsonString(sb, "detail", detail == null ? "" : detail);
            sb.append(',');
            appendJsonNumber(sb, "wrapper_counter_before", before);
            sb.append(',');
            appendJsonNumber(sb, "wrapper_counter_after", after);
            sb.append(',');
            appendJsonNumber(sb, "wrapper_counter_delta", Math.max(0L, after - before));
            sb.append(',');
            appendJsonBoolean(sb, "counted_global_stream", countedGlobal);
            sb.append(',');
            appendJsonString(sb, "result", result == null ? "" : result);
            sb.append(',');
            appendJsonString(sb, "scenario", System.getProperty("xmage.replay.scenario", ""));
            sb.append(',');
            appendJsonString(sb, "seed", System.getProperty("xmage.replay.seed", ""));
            sb.append(',');
            appendJsonString(sb, "random_util_seed", System.getProperty("xmage.replay.random_util_seed", ""));
            sb.append(',');
            appendJsonString(sb, "scope", System.getProperty("xmage.replay.scope", ""));
            sb.append(',');
            appendJsonString(sb, "mode", setting("MODE"));
            sb.append(',');
            appendJsonString(sb, "source_decision_ordinal", wrapperTraceSourceDecisionOrdinal());
            sb.append(',');
            appendJsonString(sb, "source_name", wrapperTraceSourceName());
            sb.append(',');
            appendJsonString(sb, "thread", Thread.currentThread().getName());
            sb.append(',');
            appendJsonString(sb, "caller", directTraceCaller());
            sb.append('}');
            Path path = Paths.get(WRAPPER_TRACE_FILE);
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            byte[] bytes = ("RANDOM_UTIL_WRAPPER_JSON: " + sb.toString() + System.lineSeparator())
                    .getBytes(StandardCharsets.UTF_8);
            synchronized (WRAPPER_TRACE_LOCK) {
                Files.write(path, bytes, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                        StandardOpenOption.APPEND);
            }
        } catch (Throwable ignored) {
            // Diagnostics must never affect gameplay.
        }
    }

    private static String wrapperTraceSourceDecisionOrdinal() {
        String value = wrapperTraceSourceDecisionOrdinal.get();
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty("xmage.replay.source_decision_ordinal", "");
        }
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty("xmage.replay.decision_ordinal", "");
        }
        return value == null ? "" : value.trim();
    }

    private static String wrapperTraceSourceName() {
        String value = wrapperTraceSourceName.get();
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty("xmage.replay.source_name", "");
        }
        return value == null ? "" : value.trim();
    }

    private static String directTraceCaller() {
        StackTraceElement[] trace = Thread.currentThread().getStackTrace();
        for (StackTraceElement element : trace) {
            if (element == null) {
                continue;
            }
            String cls = element.getClassName();
            if (cls == null
                    || cls.equals(Thread.class.getName())
                    || cls.equals(RandomUtil.class.getName())
                    || cls.startsWith(RandomUtil.class.getName() + "$")
                    || cls.equals("java.util.Collections")
                    || cls.startsWith("java.util.Random")) {
                continue;
            }
            return cls + "." + element.getMethodName() + ":" + element.getLineNumber();
        }
        return "";
    }

    private static void appendJsonNumber(StringBuilder sb, String key, long value) {
        appendJsonString(sb, key, null);
        sb.append(value);
    }

    private static void appendJsonBoolean(StringBuilder sb, String key, boolean value) {
        appendJsonString(sb, key, null);
        sb.append(value);
    }

    private static void appendJsonString(StringBuilder sb, String key, String value) {
        sb.append('"').append(escapeJson(key)).append('"').append(':');
        if (value == null) {
            return;
        }
        sb.append('"').append(escapeJson(value)).append('"');
    }

    private static String escapeJson(String value) {
        StringBuilder out = new StringBuilder(value == null ? 0 : value.length() + 16);
        if (value == null) {
            return "";
        }
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                    break;
            }
        }
        return out.toString();
    }

    public static final class State implements Serializable {
        private static final long serialVersionUID = 1L;

        private final CopyableRandom.RandomState randomState;
        private final long consumptionCount;
        private final long directGetRandomAccessCount;
        private final long directGetRandomConsumptionCount;
        private final long directTraceSeq;
        private final long wrapperTraceSeq;
        private final String replayRandomUtilSeed;

        private State(
                CopyableRandom.RandomState randomState,
                long consumptionCount,
                long directGetRandomAccessCount,
                long directGetRandomConsumptionCount,
                long directTraceSeq,
                long wrapperTraceSeq,
                String replayRandomUtilSeed
        ) {
            this.randomState = randomState;
            this.consumptionCount = consumptionCount;
            this.directGetRandomAccessCount = directGetRandomAccessCount;
            this.directGetRandomConsumptionCount = directGetRandomConsumptionCount;
            this.directTraceSeq = directTraceSeq;
            this.wrapperTraceSeq = wrapperTraceSeq;
            this.replayRandomUtilSeed = replayRandomUtilSeed == null ? "" : replayRandomUtilSeed;
        }

        public long getConsumptionCount() {
            return consumptionCount;
        }

        public long getDirectGetRandomAccessCount() {
            return directGetRandomAccessCount;
        }

        public long getDirectGetRandomConsumptionCount() {
            return directGetRandomConsumptionCount;
        }

        public String fingerprint() {
            long hash = 1469598103934665603L;
            hash = fnv64(hash, randomState.seed);
            hash = fnv64(hash, randomState.haveNextNextGaussian ? 1L : 0L);
            hash = fnv64(hash, Double.doubleToLongBits(randomState.nextNextGaussian));
            hash = fnv64(hash, consumptionCount);
            hash = fnv64(hash, directGetRandomAccessCount);
            hash = fnv64(hash, directGetRandomConsumptionCount);
            hash = fnv64(hash, directTraceSeq);
            hash = fnv64(hash, wrapperTraceSeq);
            return Long.toHexString(hash);
        }

        private static long fnv64(long hash, long value) {
            hash ^= value;
            return hash * 1099511628211L;
        }
    }

    private static final class CopyableRandom extends Random {
        private static final long serialVersionUID = 1L;
        private static final long MULTIPLIER = 0x5DEECE66DL;
        private static final long ADDEND = 0xBL;
        private static final long MASK = (1L << 48) - 1;

        private AtomicLong seed;
        private double nextNextGaussian;
        private boolean haveNextNextGaussian;

        private CopyableRandom() {
            this(System.currentTimeMillis() ^ System.nanoTime());
        }

        private CopyableRandom(long seed) {
            super(0L);
            setSeed(seed);
        }

        @Override
        public synchronized void setSeed(long seed) {
            seedRef().set(initialScramble(seed));
            haveNextNextGaussian = false;
            nextNextGaussian = 0.0d;
        }

        @Override
        protected int next(int bits) {
            AtomicLong seedRef = seedRef();
            long oldSeed;
            long nextSeed;
            do {
                oldSeed = seedRef.get();
                nextSeed = (oldSeed * MULTIPLIER + ADDEND) & MASK;
            } while (!seedRef.compareAndSet(oldSeed, nextSeed));
            return (int) (nextSeed >>> (48 - bits));
        }

        @Override
        public synchronized double nextGaussian() {
            if (haveNextNextGaussian) {
                haveNextNextGaussian = false;
                return nextNextGaussian;
            }
            double v1;
            double v2;
            double s;
            do {
                v1 = 2 * nextDouble() - 1;
                v2 = 2 * nextDouble() - 1;
                s = v1 * v1 + v2 * v2;
            } while (s >= 1 || s == 0);
            double multiplier = StrictMath.sqrt(-2 * StrictMath.log(s) / s);
            nextNextGaussian = v2 * multiplier;
            haveNextNextGaussian = true;
            return v1 * multiplier;
        }

        private synchronized RandomState captureState() {
            return new RandomState(seedRef().get(), haveNextNextGaussian, nextNextGaussian);
        }

        private synchronized void restoreState(RandomState state) {
            if (state == null) {
                return;
            }
            seedRef().set(state.seed & MASK);
            haveNextNextGaussian = state.haveNextNextGaussian;
            nextNextGaussian = state.nextNextGaussian;
        }

        private AtomicLong seedRef() {
            if (seed == null) {
                seed = new AtomicLong(initialScramble(0L));
            }
            return seed;
        }

        private static long initialScramble(long seed) {
            return (seed ^ MULTIPLIER) & MASK;
        }

        private static final class RandomState implements Serializable {
            private static final long serialVersionUID = 1L;

            private final long seed;
            private final boolean haveNextNextGaussian;
            private final double nextNextGaussian;

            private RandomState(long seed, boolean haveNextNextGaussian, double nextNextGaussian) {
                this.seed = seed;
                this.haveNextNextGaussian = haveNextNextGaussian;
                this.nextNextGaussian = nextNextGaussian;
            }
        }
    }

    private static final class DirectTraceRandom extends Random {
        private final Random delegate;

        private DirectTraceRandom(Random delegate) {
            super(0L);
            this.delegate = delegate;
        }

        @Override
        protected int next(int bits) {
            recordDirectGetRandomConsumption("next", "bits=" + bits);
            return delegate.nextInt() >>> (32 - bits);
        }

        @Override
        public int nextInt() {
            recordDirectGetRandomConsumption("nextInt", "");
            return delegate.nextInt();
        }

        @Override
        public int nextInt(int bound) {
            recordDirectGetRandomConsumption("nextInt", "bound=" + bound);
            return delegate.nextInt(bound);
        }

        @Override
        public long nextLong() {
            recordDirectGetRandomConsumption("nextLong", "");
            return delegate.nextLong();
        }

        @Override
        public boolean nextBoolean() {
            recordDirectGetRandomConsumption("nextBoolean", "");
            return delegate.nextBoolean();
        }

        @Override
        public float nextFloat() {
            recordDirectGetRandomConsumption("nextFloat", "");
            return delegate.nextFloat();
        }

        @Override
        public double nextDouble() {
            recordDirectGetRandomConsumption("nextDouble", "");
            return delegate.nextDouble();
        }

        @Override
        public synchronized double nextGaussian() {
            recordDirectGetRandomConsumption("nextGaussian", "");
            return delegate.nextGaussian();
        }

        @Override
        public void nextBytes(byte[] bytes) {
            recordDirectGetRandomConsumption("nextBytes", bytes == null ? "bytes=0" : "bytes=" + bytes.length);
            delegate.nextBytes(bytes);
        }

        @Override
        public synchronized void setSeed(long seed) {
            if (delegate != null) {
                delegate.setSeed(seed);
            }
        }
    }

    public static final class RandomIsolation implements AutoCloseable {
        private final Random previous;
        private boolean closed;

        private RandomIsolation(Random previous) {
            this.previous = previous;
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            if (previous == null) {
                isolatedRandom.remove();
            } else {
                isolatedRandom.set(previous);
            }
        }
    }

    public static final class WrapperTraceContext implements AutoCloseable {
        private final String previousSourceDecisionOrdinal;
        private final String previousSourceName;
        private boolean closed;

        private WrapperTraceContext(
                String previousSourceDecisionOrdinal,
                String previousSourceName,
                String sourceDecisionOrdinal,
                String sourceName
        ) {
            this.previousSourceDecisionOrdinal = previousSourceDecisionOrdinal;
            this.previousSourceName = previousSourceName;
            setOrRemove(wrapperTraceSourceDecisionOrdinal, sourceDecisionOrdinal);
            setOrRemove(wrapperTraceSourceName, sourceName);
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            setOrRemove(wrapperTraceSourceDecisionOrdinal, previousSourceDecisionOrdinal);
            setOrRemove(wrapperTraceSourceName, previousSourceName);
        }

        private static void setOrRemove(ThreadLocal<String> target, String value) {
            if (value == null || value.trim().isEmpty()) {
                target.remove();
            } else {
                target.set(value.trim());
            }
        }
    }
}
