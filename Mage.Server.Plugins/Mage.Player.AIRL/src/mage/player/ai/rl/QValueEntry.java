package mage.player.ai.rl;

public class QValueEntry {
    private final float qValue;
    private final int index;

    public QValueEntry(float qValue,int index) {
        this.qValue = qValue;
        this.index = index;
    }

    public float getQValue() {
        return qValue;
    }

    public int getIndex() {
        return index;
    }
} 