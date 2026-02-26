package mage.player.ai.rl;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Singleton that loads pre-computed text embeddings from card_embeddings.json
 * and provides fast lookup by card name.
 *
 * Lookup order:
 * 1) Profile-local embeddings (rl/profiles/<profile>/models/card_embeddings.json)
 * 2) Global shared lookup (rl/models/card_embeddings.global.json)
 * 3) Deterministic fallback embedding (auto-persisted into global lookup)
 *
 * This ensures training/inference never crashes due to missing embeddings.
 * Profile-local values take precedence over global values.
 *
 * The profile-local file is generated offline by generate_card_embeddings.py and lives
 * alongside the profile model artifacts:
 *   rl/profiles/<profile>/models/card_embeddings.json
 */
public final class CardTextEmbeddings {

    private static final Logger logger = Logger.getLogger(CardTextEmbeddings.class.getName());
    private static final int EMBED_DIM = StateSequenceBuilder.TEXT_EMBED_DIM;
    private static final String DEFAULT_GLOBAL_LOOKUP_PATH =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/card_embeddings.global.json";

    private static volatile CardTextEmbeddings INSTANCE;

    private final ConcurrentHashMap<String, float[]> embeddings;
    private final Path globalLookupPath;
    private final Object persistLock = new Object();

    private CardTextEmbeddings(Map<String, float[]> embeddings, Path globalLookupPath) {
        this.embeddings = new ConcurrentHashMap<>(embeddings);
        this.globalLookupPath = globalLookupPath;
    }

    public static CardTextEmbeddings getInstance() {
        if (INSTANCE == null) {
            synchronized (CardTextEmbeddings.class) {
                if (INSTANCE == null) {
                    INSTANCE = load();
                }
            }
        }
        return INSTANCE;
    }

    /** Force reload (useful if the file is regenerated mid-run). */
    public static void reset() {
        synchronized (CardTextEmbeddings.class) {
            INSTANCE = null;
        }
    }

    /**
     * Returns the 32-dim embedding for the given card name.
     * Missing entries are synthesized deterministically and persisted to global lookup.
     */
    public float[] getEmbedding(String cardName) {
        String key = normalizeCardName(cardName);
        float[] v = embeddings.get(key);
        if (v != null) {
            return v;
        }
        return loadOrCreateAndPersist(key);
    }

    public int size() {
        return embeddings.size();
    }

    // -----------------------------------------------------------------------

    private static CardTextEmbeddings load() {
        String profilePath = RLLogPaths.MODELS_BASE_DIR + "/card_embeddings.json";
        Path globalPath = Paths.get(EnvConfig.str("RL_GLOBAL_EMBEDDINGS_LOOKUP_PATH", DEFAULT_GLOBAL_LOOKUP_PATH));

        Map<String, float[]> profileMap = loadEmbeddingFile(Paths.get(profilePath), "profile");
        Map<String, float[]> globalMap = loadEmbeddingFile(globalPath, "global");

        // Lookup precedence: global first, then profile override.
        Map<String, float[]> merged = new HashMap<>(globalMap);
        merged.putAll(profileMap);

        if (merged.isEmpty()) {
            logger.warning("CardTextEmbeddings: no embedding files found/populated. "
                    + "Missing cards will be generated deterministically.");
        } else {
            logger.info("CardTextEmbeddings: loaded profile=" + profileMap.size()
                    + ", global=" + globalMap.size() + ", merged=" + merged.size());
        }
        return new CardTextEmbeddings(merged, globalPath);
    }

    private static Map<String, float[]> loadEmbeddingFile(Path path, String label) {
        File file = path.toFile();
        if (!file.exists()) {
            logger.info("CardTextEmbeddings: " + label + " embedding file missing at " + path);
            return new HashMap<>();
        }
        try {
            String json = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
            Map<String, float[]> map = parseJson(json);
            logger.info("CardTextEmbeddings: loaded " + map.size() + " " + label + " embeddings from " + path);
            return map;
        } catch (Exception e) {
            logger.warning("CardTextEmbeddings: failed to load " + label + " embedding file at "
                    + path + " (" + e.getMessage() + ")");
            return new HashMap<>();
        }
    }

    private static void writeEmbeddingFile(Path path, Map<String, float[]> map) {
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            String json = toJson(map);
            Files.write(path, json.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        } catch (IOException e) {
            logger.warning("CardTextEmbeddings: failed to write embeddings to " + path + " (" + e.getMessage() + ")");
        }
    }

    private float[] loadOrCreateAndPersist(String cardName) {
        synchronized (persistLock) {
            float[] existing = embeddings.get(cardName);
            if (existing != null) {
                return existing;
            }

            Map<String, float[]> latestGlobal = loadEmbeddingFile(globalLookupPath, "global");
            float[] global = latestGlobal.get(cardName);
            if (global != null) {
                embeddings.put(cardName, global);
                return global;
            }

            float[] generated = generateDeterministicEmbedding(cardName);
            embeddings.put(cardName, generated);
            latestGlobal.put(cardName, generated);
            writeEmbeddingFile(globalLookupPath, latestGlobal);
            logger.info("CardTextEmbeddings: auto-added missing embedding for '" + cardName
                    + "' to global lookup " + globalLookupPath);
            return generated;
        }
    }

    private static String normalizeCardName(String cardName) {
        if (cardName == null) {
            return "<unknown-card>";
        }
        String key = cardName.trim();
        return key.isEmpty() ? "<empty-card-name>" : key;
    }

    private static float[] generateDeterministicEmbedding(String cardName) {
        float[] vec = new float[EMBED_DIM];
        long seed = 1469598103934665603L;
        for (int i = 0; i < cardName.length(); i++) {
            seed ^= cardName.charAt(i);
            seed *= 1099511628211L;
        }
        Random rng = new Random(seed);
        double normSq = 0.0;
        for (int i = 0; i < EMBED_DIM; i++) {
            float value = (float) rng.nextGaussian();
            vec[i] = value;
            normSq += (double) value * value;
        }
        double norm = Math.sqrt(normSq);
        if (norm > 0.0) {
            for (int i = 0; i < EMBED_DIM; i++) {
                vec[i] = (float) (vec[i] / norm);
            }
        }
        return vec;
    }

    private static String toJson(Map<String, float[]> map) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        List<String> keys = new ArrayList<>(map.keySet());
        Collections.sort(keys);
        for (int k = 0; k < keys.size(); k++) {
            String key = keys.get(k);
            float[] vec = map.get(key);
            if (vec == null) {
                vec = new float[EMBED_DIM];
            }
            sb.append("  \"").append(escapeJson(key)).append("\": [");
            for (int i = 0; i < EMBED_DIM; i++) {
                if (i > 0) sb.append(", ");
                float value = i < vec.length ? vec[i] : 0.0f;
                sb.append(String.format(Locale.ROOT, "%.6f", value));
            }
            sb.append("]");
            if (k < keys.size() - 1) {
                sb.append(",");
            }
            sb.append("\n");
        }
        sb.append("}\n");
        return sb.toString();
    }

    private static String escapeJson(String s) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' || c == '"') {
                out.append('\\').append(c);
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * Minimal hand-rolled JSON parser for the specific format:
     *   { "Card Name": [f0, f1, ...], ... }
     *
     * Avoids pulling in a JSON library dependency.
     */
    static Map<String, float[]> parseJson(String json) {
        Map<String, float[]> result = new HashMap<>();
        int i = 0;
        int len = json.length();

        // skip to opening brace
        while (i < len && json.charAt(i) != '{') i++;
        i++; // skip '{'

        while (i < len) {
            // skip whitespace / commas
            while (i < len && (json.charAt(i) == ',' || Character.isWhitespace(json.charAt(i)))) i++;
            if (i >= len || json.charAt(i) == '}') break;

            // parse key string
            if (json.charAt(i) != '"') {
                i++;
                continue;
            }
            i++; // skip opening quote
            StringBuilder key = new StringBuilder();
            while (i < len && json.charAt(i) != '"') {
                if (json.charAt(i) == '\\' && i + 1 < len) {
                    i++; // skip escape char
                    key.append(json.charAt(i));
                } else {
                    key.append(json.charAt(i));
                }
                i++;
            }
            i++; // skip closing quote

            // skip colon
            while (i < len && json.charAt(i) != '[') i++;
            i++; // skip '['

            // parse float array
            float[] vec = new float[EMBED_DIM];
            int valueCount = 0;
            while (i < len && json.charAt(i) != ']') {
                while (i < len && (json.charAt(i) == ',' || Character.isWhitespace(json.charAt(i)))) i++;
                if (i >= len || json.charAt(i) == ']') break;
                int start = i;
                while (i < len && json.charAt(i) != ',' && json.charAt(i) != ']') i++;
                String raw = json.substring(start, i).trim();
                try {
                    float parsed = Float.parseFloat(raw);
                    if (valueCount < EMBED_DIM) {
                        vec[valueCount] = parsed;
                    }
                    valueCount++;
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Invalid float value in card_embeddings.json for card '" + key
                                    + "' at index " + valueCount + ": '" + raw + "'",
                            e
                    );
                }
            }
            i++; // skip ']'

            if (valueCount != EMBED_DIM) {
                throw new IllegalArgumentException(
                        "Embedding dimension mismatch for card '" + key
                                + "': expected " + EMBED_DIM + ", got " + valueCount);
            }
            result.put(key.toString(), vec);
        }
        return result;
    }
}
