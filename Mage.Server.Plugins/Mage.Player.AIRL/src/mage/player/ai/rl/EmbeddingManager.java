package mage.player.ai.rl;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.CreateEmbeddingResponse;
import com.openai.models.EmbeddingCreateParams;

public class EmbeddingManager {
    private static final Logger logger = Logger.getLogger(EmbeddingManager.class);
    private static final String MAPPING_FILE = "mapping.json";
    private static Map<String, float[]> embeddings;
    private static OpenAIClient openAIClient;
    public static final int EMBEDDING_SIZE = 100;

    public static synchronized OpenAIClient getOpenAIClient() {
        if (openAIClient == null) {
            openAIClient = OpenAIOkHttpClient.fromEnv();
        }
        return openAIClient;
    }

    public static synchronized Map<String, float[]> getEmbeddings() {
        if (embeddings == null) {
            embeddings = loadEmbeddings();
        }
        return embeddings;
    }

    private static Map<String, float[]> loadEmbeddings() {
        try (FileReader reader = new FileReader(MAPPING_FILE)) {
            Type type = new TypeToken<HashMap<String, float[]>>() {}.getType();
            return new Gson().fromJson(reader, type);
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
            float[] embedding = queryOpenAIForEmbedding(text);
            embeddings.put(text, embedding);
            saveEmbeddings();
            return embedding;
        }
    }

    private static float[] queryOpenAIForEmbedding(String text) {
        try {
            EmbeddingCreateParams params = new EmbeddingCreateParams.Builder()
                .model("text-embedding-3-small")
                .dimensions(EMBEDDING_SIZE)
                .input(text)
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

    private static void saveEmbeddings() {
        try (FileWriter writer = new FileWriter(MAPPING_FILE)) {
            new Gson().toJson(getEmbeddings(), writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
} 