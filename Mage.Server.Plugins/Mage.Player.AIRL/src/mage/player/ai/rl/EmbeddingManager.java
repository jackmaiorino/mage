package mage.player.ai.rl;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.embeddings.EmbeddingRequest;
import com.openai.embeddings.EmbeddingResponse;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixImpl;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

public class EmbeddingManager {
    private static final String MAPPING_FILE = "mapping.json";
    private static final Map<String, float[]> embeddings = loadEmbeddings();
    private static final OpenAIClient openAIClient = OpenAIOkHttpClient.fromEnv();
    public static final int EMBEDDING_SIZE = 1536;
    public static final int REDUCED_EMBEDDING_SIZE = 100; // Reduced size

    private static Map<String, float[]> loadEmbeddings() {
        try (FileReader reader = new FileReader(MAPPING_FILE)) {
            Type type = new TypeToken<HashMap<String, float[]>>() {}.getType();
            return new Gson().fromJson(reader, type);
        } catch (IOException e) {
            e.printStackTrace();
            return new HashMap<>();
        }
    }

    private static void saveEmbeddings() {
        try (FileWriter writer = new FileWriter(MAPPING_FILE)) {
            new Gson().toJson(embeddings, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static float[] getEmbedding(String text) {
        if (embeddings.containsKey(text)) {
            return embeddings.get(text);
        } else {
            float[] embedding = queryOpenAIForEmbedding(text);
            float[] reducedEmbedding = reduceEmbedding(embedding);
            embeddings.put(text, reducedEmbedding);
            saveEmbeddings();
            return reducedEmbedding;
        }
    }

    private static float[] queryOpenAIForEmbedding(String text) {
        try {
            EmbeddingResponse response = openAIClient.createEmbedding(
                EmbeddingRequest.builder()
                    .model("text-embedding-ada-002")
                    .input(text)
                    .build()
            );

            if (response != null && !response.getData().isEmpty()) {
                return response.getData().get(0).getEmbedding();
            } else {
                throw new RuntimeException("Failed to get embedding from OpenAI API");
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Error querying OpenAI API for embedding", e);
        }
    }

    private static float[] reduceEmbedding(float[] embedding) {
        double[] doubleEmbedding = new double[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            doubleEmbedding[i] = embedding[i];
        }
        RealMatrix matrix = new Array2DRowRealMatrix(new double[][]{doubleEmbedding});
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        RealMatrix reducedMatrix = svd.getU().getSubMatrix(0, embedding.length - 1, 0, REDUCED_EMBEDDING_SIZE - 1);
        double[] doubleResult = reducedMatrix.getColumn(0);
        float[] result = new float[doubleResult.length];
        for (int i = 0; i < doubleResult.length; i++) {
            result[i] = (float) doubleResult[i];
        }
        return result;
    }
} 