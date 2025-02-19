package mage.player.ai.rl;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.openai.models.EmbeddingModel;
import org.apache.log4j.Logger;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.CreateEmbeddingResponse;
import com.openai.models.EmbeddingCreateParams;

//TODO: This may be oversynchronized slowing training, investigating.
public class EmbeddingManager {
    private static final Logger logger = Logger.getLogger(EmbeddingManager.class);
    private static final String MAPPING_FILE = "../Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/Storage/mapping.json";
    private static Map<String, float[]> embeddings;
    private static OpenAIClient openAIClient;
    public static final int EMBEDDING_SIZE = 100;

    public static synchronized OpenAIClient getOpenAIClient() {
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
        return openAIClient;
    }

    public static synchronized Map<String, float[]> getEmbeddings() {
        if (embeddings == null) {
            embeddings = loadEmbeddings();
        }
        return embeddings;
    }

    private static synchronized Map<String, float[]> loadEmbeddings() {
        try (FileReader reader = new FileReader(MAPPING_FILE)) {
            Type type = new TypeToken<HashMap<String, float[]>>() {}.getType();
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
        embeddings = getEmbeddings();
        if (embeddings.containsKey(text)) {
            return embeddings.get(text);
        } else {
            // Tokenize only if we need to query OpenAI
            String tokenizedText = tokenizeCardText(text);
            float[] embedding = queryOpenAIForEmbedding(tokenizedText);
            embeddings.put(text, embedding);
            saveEmbeddings();
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

    private static synchronized float[] queryOpenAIForEmbedding(String text) {
        try {
            EmbeddingCreateParams params = new EmbeddingCreateParams.Builder()
                .model(EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
                .dimensions(EMBEDDING_SIZE)
                .input(EmbeddingCreateParams.Input.ofString(text))
                .build();

            CreateEmbeddingResponse response = getOpenAIClient().embeddings().create(params);
            
            // TODO: We should probably change our implementation to use List<Double> instead of float[]
            // This is a temporary fix to get the embedding
            List<Double> embedding = response.data().get(0).embedding();
            float[] result = new float[embedding.size()];
            for (int i = 0; i < embedding.size(); i++) {
                result[i] = embedding.get(i).floatValue();
            }
            return result;

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Error querying OpenAI API for embedding", e);
        }
    }

    public static synchronized void saveEmbeddings() {
        try (FileWriter writer = new FileWriter(MAPPING_FILE)) {
            new Gson().toJson(getEmbeddings(), writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
} 