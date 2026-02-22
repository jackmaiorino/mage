package mage.player.ai.rl;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Singleton that loads pre-computed text embeddings from card_embeddings.json
 * and provides fast lookup by card name.
 *
 * The JSON file is generated offline by generate_card_embeddings.py and lives
 * alongside the profile's model artifacts:
 *   rl/profiles/<profile>/models/card_embeddings.json
 *
 * Falls back to a zero vector silently if the file is missing or a card name
 * is not found, so training is unaffected when no embeddings are available.
 */
public final class CardTextEmbeddings {

    private static final Logger logger = Logger.getLogger(CardTextEmbeddings.class.getName());
    private static final int EMBED_DIM = StateSequenceBuilder.TEXT_EMBED_DIM;
    private static final float[] ZERO = new float[EMBED_DIM];

    private static volatile CardTextEmbeddings INSTANCE;

    private final Map<String, float[]> embeddings;

    private CardTextEmbeddings(Map<String, float[]> embeddings) {
        this.embeddings = embeddings;
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
     * Returns the 32-dim embedding for the given card name, or a zero vector
     * if not found.
     */
    public float[] getEmbedding(String cardName) {
        if (cardName == null) return ZERO;
        float[] v = embeddings.get(cardName);
        return v != null ? v : ZERO;
    }

    public int size() {
        return embeddings.size();
    }

    // -----------------------------------------------------------------------

    private static CardTextEmbeddings load() {
        String path = RLLogPaths.MODELS_BASE_DIR + "/card_embeddings.json";
        File file = new File(path);
        if (!file.exists()) {
            logger.info("CardTextEmbeddings: no embedding file found at " + path + " — using zero vectors");
            return new CardTextEmbeddings(Collections.emptyMap());
        }

        try {
            String json = new String(Files.readAllBytes(file.toPath()), java.nio.charset.StandardCharsets.UTF_8);
            Map<String, float[]> map = parseJson(json);
            logger.info("CardTextEmbeddings: loaded " + map.size() + " card embeddings from " + path);
            return new CardTextEmbeddings(map);
        } catch (IOException e) {
            logger.warning("CardTextEmbeddings: failed to read " + path + ": " + e.getMessage());
            return new CardTextEmbeddings(Collections.emptyMap());
        } catch (Exception e) {
            logger.warning("CardTextEmbeddings: failed to parse " + path + ": " + e.getMessage());
            return new CardTextEmbeddings(Collections.emptyMap());
        }
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
            if (json.charAt(i) != '"') { i++; continue; }
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
            int idx = 0;
            while (i < len && json.charAt(i) != ']') {
                while (i < len && (json.charAt(i) == ',' || Character.isWhitespace(json.charAt(i)))) i++;
                if (i >= len || json.charAt(i) == ']') break;
                int start = i;
                while (i < len && json.charAt(i) != ',' && json.charAt(i) != ']') i++;
                if (idx < EMBED_DIM) {
                    try {
                        vec[idx++] = Float.parseFloat(json.substring(start, i).trim());
                    } catch (NumberFormatException ignored) {}
                }
            }
            i++; // skip ']'

            result.put(key.toString(), vec);
        }
        return result;
    }
}
