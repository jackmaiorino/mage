package mage.player.ai.rl;

import java.util.*;

public class ReplayBuffer {
    private final int capacity;
    private final Deque<Experience> buffer;
    private int updateCounter;
    private final Random random;
    
    public ReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayDeque<>(capacity);
        this.random = new Random();
        this.updateCounter = 0;
    }
    
    public void add(Experience experience) {
        if (buffer.size() >= capacity) {
            buffer.removeFirst();
        }
        buffer.addLast(experience);
        updateCounter++;
    }
    
    public List<Experience> sample(int batchSize) {
        batchSize = Math.min(batchSize, buffer.size());
        List<Experience> batch = new ArrayList<>(batchSize);
        List<Experience> temp = new ArrayList<>(buffer);
        
        for (int i = 0; i < batchSize; i++) {
            int index = random.nextInt(temp.size());
            batch.add(temp.get(index));
        }
        
        return batch;
    }
    
    public int size() {
        return buffer.size();
    }
    
    public int getUpdateCounter() {
        return updateCounter;
    }
} 