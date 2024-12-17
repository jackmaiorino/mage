package mage.player.ai.rl;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
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
    public static final int EMBEDDING_SIZE = 1536;
    public static final int REDUCED_EMBEDDING_SIZE = 100;

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
            float[] reducedEmbedding = reduceEmbedding(embedding);
            embeddings.put(text, reducedEmbedding);
            saveEmbeddings();
            return reducedEmbedding;
        }
    }

    private static float[] queryOpenAIForEmbedding(String text) {
        try {
            EmbeddingCreateParams params = new EmbeddingCreateParams.Builder()
                .model("text-embedding-ada-002")
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

    // I don't really understand this.Need to research more.
    private static float[] reduceEmbedding(float[] embedding) {
        double[] doubleEmbedding = new double[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            doubleEmbedding[i] = embedding[i];
        }
        RealMatrix matrix = new Array2DRowRealMatrix(new double[][]{doubleEmbedding});
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);

        // Get actual dimensions of the U matrix
        int uRows = svd.getU().getRowDimension();
        int uCols = svd.getU().getColumnDimension();

        // Ensure indices are within bounds
        int numRows = Math.min(embedding.length, uRows);
        int numCols = Math.min(REDUCED_EMBEDDING_SIZE, uCols);

        RealMatrix reducedMatrix = svd.getU().getSubMatrix(0, numRows - 1, 0, numCols - 1);
        double[] doubleResult = reducedMatrix.getColumn(0);
        float[] result = new float[doubleResult.length];
        for (int i = 0; i < doubleResult.length; i++) {
            result[i] = (float) doubleResult[i];
        }
        return result;
    }

    private static void saveEmbeddings() {
        try (FileWriter writer = new FileWriter(MAPPING_FILE)) {
            new Gson().toJson(getEmbeddings(), writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
} 