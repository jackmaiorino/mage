package mage.player.ai.rl;

public class QValueEntry {
    private final float qValue;
    private final int xIndex;
    private final int yIndex;

    public QValueEntry(float qValue, int xIndex, int yIndex) {
        this.qValue = qValue;
        this.xIndex = xIndex;
        this.yIndex = yIndex;
    }

    public float getQValue() {
        return qValue;
    }

    public int getXIndex() {
        return xIndex;
    }

    public int getYIndex() {
        return yIndex;
    }
} 