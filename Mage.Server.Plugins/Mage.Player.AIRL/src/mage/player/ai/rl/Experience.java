package mage.player.ai.rl;

public class Experience {
    public final RLState state;
        
    //This class used to have a lot more before I realized we could trim it down
    public Experience(RLState state) {
        this.state = state;
    }
} 