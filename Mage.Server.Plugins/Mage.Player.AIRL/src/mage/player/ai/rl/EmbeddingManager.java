package mage.player.ai.rl;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.log4j.Logger;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.CreateEmbeddingResponse;
import com.openai.models.EmbeddingCreateParams;
import com.openai.models.EmbeddingModel;

//TODO: This may be oversynchronized slowing training, investigating.
public class EmbeddingManager {

    private static final Logger logger = Logger.getLogger(EmbeddingManager.class);
    private static final String MAPPING_FILE = System.getenv().getOrDefault(
            "EMBEDDING_MAPPING_PATH",
            "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/mapping.json"
    );
    private static volatile Map<String, float[]> embeddings;
    private static volatile OpenAIClient openAIClient;
    public static final int EMBEDDING_SIZE = 100;

    // Track unsaved embeddings to batch saves
    private static final AtomicInteger unsavedCount = new AtomicInteger(0);
    private static final int SAVE_BATCH_SIZE = 10; // Save every 10 new embeddings
    private static final Object saveLock = new Object();

    public static OpenAIClient getOpenAIClient() {
        if (openAIClient == null) {
            synchronized (EmbeddingManager.class) {
                if (openAIClient == null) {
                    String apiKey = System.getenv("OPENAI_API_KEY");
                    if (apiKey == null) {
                        throw new IllegalStateException("Environment variable OPENAI_API_KEY is not set.");
                    }
                    System.out.println("Using API Key: " + apiKey); // Avoid logging sensitive info in production!
                    // TODO: Why doesn't FromEnv work here anymore?!
                    openAIClient = OpenAIOkHttpClient.builder()
                            .apiKey(apiKey)
                            .build();
                }
            }
        }
        return openAIClient;
    }

    public static Map<String, float[]> getEmbeddings() {
        if (embeddings == null) {
            synchronized (EmbeddingManager.class) {
                if (embeddings == null) {
                    embeddings = new ConcurrentHashMap<>(loadEmbeddings());
                }
            }
        }
        return embeddings;
    }

    private static Map<String, float[]> loadEmbeddings() {
        try (FileReader reader = new FileReader(MAPPING_FILE)) {
            Type type = new TypeToken<HashMap<String, float[]>>() {
            }.getType();
            Map<String, float[]> loadedEmbeddings = new Gson().fromJson(reader, type);
            if (loadedEmbeddings == null) {
                logger.error("Loaded embeddings are null. Check the file content and format.");
                return new HashMap<>();
            }
            logger.info("Successfully loaded embeddings from " + MAPPING_FILE);
            return loadedEmbeddings;
        } catch (IOException e) {
            logger.error("Error loading embeddings from " + MAPPING_FILE, e);
            return new HashMap<>();
        }
    }

    public static float[] getEmbedding(String text) {
        if (text == null || text.isEmpty()) {
            return new float[EMBEDDING_SIZE];
        }

        // Log the raw text being sent for embedding
        //logger.info("Getting embedding for text: " + text);
        Map<String, float[]> embeddingMap = getEmbeddings();

        float[] cachedEmbedding = embeddingMap.get(text);
        if (cachedEmbedding != null) {
            // Validate cached embedding
            for (int i = 0; i < cachedEmbedding.length; i++) {
                if (Float.isNaN(cachedEmbedding[i]) || Float.isInfinite(cachedEmbedding[i])) {
                    logger.warn("Invalid value in cached embedding for text: " + text);
                    cachedEmbedding[i] = 0.0f;
                }
            }
            return cachedEmbedding;
        } else {
            // Tokenize only if we need to query OpenAI
            String tokenizedText = tokenizeCardText(text);
            float[] embedding = queryOpenAIForEmbedding(tokenizedText);

            // Validate embedding before caching
            for (int i = 0; i < embedding.length; i++) {
                if (Float.isNaN(embedding[i]) || Float.isInfinite(embedding[i])) {
                    logger.warn("Invalid value in new embedding for text: " + text);
                    embedding[i] = 0.0f;
                }
            }

            embeddingMap.put(text, embedding);

            // Batch saves instead of saving after every embedding
            int currentUnsaved = unsavedCount.incrementAndGet();
            if (currentUnsaved >= SAVE_BATCH_SIZE) {
                // Use separate lock for saving to avoid blocking embedding access
                if (unsavedCount.compareAndSet(currentUnsaved, 0)) {
                    new Thread(() -> saveEmbeddingsAsync()).start();
                }
            }

            // Log the first few values of the embedding
            StringBuilder sb = new StringBuilder("First 5 embedding values: ");
            for (int i = 0; i < Math.min(5, embedding.length); i++) {
                sb.append(embedding[i]).append(" ");
            }
            logger.info(sb.toString());

            return embedding;
        }
    }

    public static String tokenizeCardText(String cardText) {
        //TODO: Further tokenization of common effects and phrases could be beneficial
        // Replace mana costs with structured tokens
        // TODO: distinguish between cost and effect?
        cardText = cardText.replaceAll("\\{(\\d+)\\}", "GENERIC_MANA_$1");
        cardText = cardText.replaceAll("\\{W\\}", "WHITE_MANA");
        cardText = cardText.replaceAll("\\{U\\}", "BLUE_MANA");
        cardText = cardText.replaceAll("\\{B\\}", "BLACK_MANA");
        cardText = cardText.replaceAll("\\{R\\}", "RED_MANA");
        cardText = cardText.replaceAll("\\{G\\}", "GREEN_MANA");
        cardText = cardText.replaceAll("\\{C\\}", "COLORLESS_MANA");
        cardText = cardText.replaceAll("\\{T\\}", "COST_TAP");
        cardText = cardText.replaceAll("\\{S\\}", "COST_SACRIFICE");
        cardText = cardText.replaceAll("\\{X\\}", "COST_X_MANA");

        // Optionally, add delimiters for different sections
        cardText = cardText.replaceAll("\\. ", ". | ");

        return cardText;
    }

    private static float[] queryOpenAIForEmbedding(String text) {
        try {
            EmbeddingCreateParams params = new EmbeddingCreateParams.Builder()
                    .model(EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
                    .dimensions(EMBEDDING_SIZE)
                    .input(EmbeddingCreateParams.Input.ofString(text))
                    .build();

            CreateEmbeddingResponse response = getOpenAIClient().embeddings().create(params);

            if (response == null || response.data() == null || response.data().isEmpty()) {
                throw new IllegalStateException("Received null or empty embedding response for text: " + text);
            }

            List<Double> embedding = response.data().get(0).embedding();
            if (embedding == null || embedding.isEmpty()) {
                throw new IllegalStateException("Received null or empty embedding list for text: " + text);
            }

            float[] result = new float[embedding.size()];
            for (int i = 0; i < embedding.size(); i++) {
                Double value = embedding.get(i);
                if (value == null || Double.isNaN(value) || Double.isInfinite(value)) {
                    throw new IllegalStateException("Invalid embedding value at index " + i + " for text: " + text);
                }
                result[i] = value.floatValue();
            }
            return result;

        } catch (Exception e) {
            logger.error("Error querying OpenAI API for embedding: " + e.getMessage());
            throw new RuntimeException("Failed to get embedding from OpenAI API", e);
        }
    }

    // Asynchronous save to avoid blocking
    private static void saveEmbeddingsAsync() {
        synchronized (saveLock) {
            try (FileWriter writer = new FileWriter(MAPPING_FILE)) {
                new Gson().toJson(getEmbeddings(), writer);
                logger.info("Saved embeddings to " + MAPPING_FILE);
            } catch (IOException e) {
                logger.error("Error saving embeddings", e);
            }
        }
    }

    // Keep original method for compatibility, but make it less frequent
    public static void saveEmbeddings() {
        unsavedCount.set(SAVE_BATCH_SIZE); // Force a save
        saveEmbeddingsAsync();
    }
}
