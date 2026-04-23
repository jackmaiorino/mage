package org.mage.test.player;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import mage.abilities.Ability;
import mage.cards.Cards;
import mage.choices.Choice;
import mage.constants.Outcome;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.rl.PythonMLBatchManager;
import mage.player.ai.rl.PythonModel;
import mage.player.ai.rl.StateSequenceBuilder;
import mage.target.Target;
import mage.target.TargetCard;

/**
 * RL AI: helper class for tests.
 */
public final class TestComputerPlayerRL extends ComputerPlayerRL {

    private TestPlayer testPlayerLink;

    public TestComputerPlayerRL(String name, RangeOfInfluence range, int skill) {
        super(name, range, NoOpPythonModel.INSTANCE, false, false, "test");
    }

    public void setTestPlayerLink(TestPlayer testPlayerLink) {
        this.testPlayerLink = testPlayerLink;
    }

    @Override
    public boolean choose(Outcome outcome, Target target, Ability source, Game game) {
        if (testPlayerLink.canChooseByComputer()) {
            return super.choose(outcome, target, source, game);
        } else {
            return testPlayerLink.choose(outcome, target, source, game);
        }
    }

    @Override
    public boolean choose(Outcome outcome, Choice choice, Game game) {
        if (testPlayerLink.canChooseByComputer()) {
            return super.choose(outcome, choice, game);
        } else {
            return testPlayerLink.choose(outcome, choice, game);
        }
    }

    @Override
    public boolean choose(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        if (testPlayerLink.canChooseByComputer()) {
            return super.choose(outcome, cards, target, source, game);
        } else {
            return testPlayerLink.choose(outcome, cards, target, source, game);
        }
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Target target, Ability source, Game game) {
        if (testPlayerLink.canChooseByComputer()) {
            return super.chooseTarget(outcome, target, source, game);
        } else {
            return testPlayerLink.chooseTarget(outcome, target, source, game);
        }
    }

    @Override
    public boolean chooseTarget(Outcome outcome, Cards cards, TargetCard target, Ability source, Game game) {
        if (testPlayerLink.canChooseByComputer()) {
            return super.chooseTarget(outcome, cards, target, source, game);
        } else {
            return testPlayerLink.chooseTarget(outcome, cards, target, source, game);
        }
    }

    @Override
    public boolean flipCoinResult(Game game) {
        return testPlayerLink.flipCoinResult(game);
    }

    @Override
    public int rollDieResult(int sides, Game game) {
        return testPlayerLink.rollDieResult(sides, game);
    }

    @Override
    public boolean isComputer() {
        if (testPlayerLink.canChooseByComputer()) {
            return super.isComputer();
        } else {
            return testPlayerLink.isComputer();
        }
    }

    private enum NoOpPythonModel implements PythonModel {
        INSTANCE;

        @Override
        public PythonMLBatchManager.PredictionResult scoreCandidates(
                StateSequenceBuilder.SequenceOutput state,
                int[] candidateActionIds,
                float[][] candidateFeatures,
                int[] candidateMask,
                String policyKey,
                String headId,
                int pickIndex,
                int minTargets,
                int maxTargets
        ) {
            // Random uniform policy + zero value. Lets benchmark tests play
            // full games without a real model; previously threw to catch
            // accidental inference in mana-shape tests, but we need scoring
            // to work for MCTS profiling harnesses.
            int n = candidateActionIds.length;
            float[] policy = new float[n];
            ThreadLocalRandom rng = ThreadLocalRandom.current();
            for (int i = 0; i < n; i++) policy[i] = rng.nextFloat();
            return new PythonMLBatchManager.PredictionResult(policy, 0.0f);
        }

        @Override
        public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        }

        @Override
        public void saveModel(String path) {
        }

        @Override
        public String getDeviceInfo() {
            return "test";
        }

        @Override
        public Map<String, Integer> getMainModelTrainingStats() {
            return Collections.emptyMap();
        }

        @Override
        public Map<String, Integer> getHealthStats() {
            return Collections.emptyMap();
        }

        @Override
        public void resetHealthStats() {
        }

        @Override
        public void recordGameResult(float lastValuePrediction, boolean won) {
        }

        @Override
        public Map<String, Object> getValueHeadMetrics() {
            return Collections.emptyMap();
        }

        @Override
        public void shutdown() {
        }
    }
}
