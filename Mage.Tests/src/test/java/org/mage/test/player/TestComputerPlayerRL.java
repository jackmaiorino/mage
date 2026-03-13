package org.mage.test.player;

import java.util.Collections;
import java.util.List;
import java.util.Map;

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
            throw new UnsupportedOperationException("TestComputerPlayerRL does not support policy scoring");
        }

        @Override
        public void enqueueTraining(List<StateSequenceBuilder.TrainingData> trainingData, List<Double> rewards) {
        }

        @Override
        public float predictMulligan(float[] features) {
            return 0.0f;
        }

        @Override
        public float[] predictMulliganScores(float[] features) {
            return new float[]{0.0f, 0.0f};
        }

        @Override
        public void trainMulligan(byte[] features, byte[] decisions, byte[] outcomes, byte[] gameLengths, byte[] earlyLandScores, byte[] overrides, int batchSize) {
        }

        @Override
        public void saveMulliganModel() {
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
        public Map<String, Integer> getMulliganModelTrainingStats() {
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
