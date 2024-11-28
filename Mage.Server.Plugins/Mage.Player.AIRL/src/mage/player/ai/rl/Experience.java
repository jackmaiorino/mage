package mage.player.ai.rl;

public class Experience {
    public final RLState state;
    public final RLAction action;
    public final double reward;
    public final RLState nextState;
    
    public Experience(RLState state, RLAction action, double reward, RLState nextState) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
    }
} 