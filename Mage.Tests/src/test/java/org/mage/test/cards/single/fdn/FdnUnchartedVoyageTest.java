package org.mage.test.cards.single.fdn;

import mage.constants.PhaseStep;
import mage.constants.Zone;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.player.TestPlayer;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Reference cases for owner placement followed by the caster's surveil. */
public class FdnUnchartedVoyageTest extends CardTestPlayerBase {
    private static final String VOYAGE = "Uncharted Voyage";
    private static final String ELF = "Elvish Mystic";

    private void setup(int islands) {
        skipInitShuffling();
        addCard(Zone.BATTLEFIELD, playerA, "Island", islands);
        addCard(Zone.HAND, playerA, VOYAGE);
        addCard(Zone.LIBRARY, playerA, "Forest", 2);
        addCard(Zone.LIBRARY, playerB, "Mountain", 2);
    }

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
    }

    private void assertTop(TestPlayer player, String name) {
        Assert.assertEquals(name, currentGame.getPlayer(player.getId())
                .getLibrary().getFromTop(currentGame).getName());
    }

    @Test
    public void opposingOwnerChoosesTopAndCasterKeepsTheirOwnTopCard() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerB, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerB, true);
        addTarget(playerA, TestPlayer.TARGET_SKIP);
        finish();
        assertTop(playerB, ELF);
        assertTop(playerA, "Forest");
        assertLibraryCount(playerB, ELF, 1);
        assertLibraryCount(playerA, "Forest", 2);
        assertGraveyardCount(playerA, VOYAGE, 1);
        assertTappedCount("Island", true, 4);
    }

    @Test
    public void opposingOwnerChoosesBottomAndCasterSurveilsIntoGraveyard() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerB, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerB, false);
        addTarget(playerA, "Forest");
        finish();
        assertTop(playerB, "Mountain");
        assertLibraryCount(playerB, ELF, 1);
        assertGraveyardCount(playerA, "Forest", 1);
        assertLibraryCount(playerA, "Forest", 1);
    }

    @Test
    public void ownTopTargetIsTheCardSurveiledAndCanBeKept() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerA, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerA, true);
        addTarget(playerA, TestPlayer.TARGET_SKIP);
        finish();
        assertTop(playerA, ELF);
        assertLibraryCount(playerA, "Forest", 2);
        assertGraveyardCount(playerA, ELF, 0);
    }

    @Test
    public void ownTopTargetCanBeSurveiledIntoGraveyard() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerA, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerA, true);
        addTarget(playerA, ELF);
        finish();
        assertTop(playerA, "Forest");
        assertLibraryCount(playerA, ELF, 0);
        assertGraveyardCount(playerA, ELF, 1);
    }

    @Test
    public void stolenCreatureGoesIntoOwnersLibrary() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerA, "Mountain", 3);
        addCard(Zone.BATTLEFIELD, playerB, ELF);
        addCard(Zone.HAND, playerA, "Act of Treason");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Act of Treason", ELF, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerB, true);
        addTarget(playerA, TestPlayer.TARGET_SKIP);
        finish();
        assertTop(playerB, ELF);
        assertLibraryCount(playerA, ELF, 0);
        assertLibraryCount(playerB, ELF, 1);
    }

    @Test
    public void blinkResponseInvalidatesOnlyTargetAndSkipsSurveil() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerB, ELF);
        addCard(Zone.BATTLEFIELD, playerB, "Plains", 2);
        addCard(Zone.HAND, playerB, "Momentary Blink");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerB, "Momentary Blink", ELF);
        finish();
        assertPermanentCount(playerB, ELF, 1);
        assertLibraryCount(playerA, "Forest", 2);
        assertGraveyardCount(playerA, VOYAGE, 1);
    }

    @Test
    public void ownTopTokenDoesNotHideTheCardSurveiled() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 2);
        addCard(Zone.HAND, playerA, "Raise the Alarm");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Raise the Alarm", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, "Soldier Token");
        setChoice(playerA, true);
        addTarget(playerA, "Forest");
        finish();
        assertPermanentCount(playerA, "Soldier Token", 1);
        assertLibraryCount(playerA, "Soldier Token", 0);
        assertGraveyardCount(playerA, "Soldier Token", 0);
        assertGraveyardCount(playerA, "Forest", 1);
    }

    @Test
    public void opposingTokenDisappearsAndCasterCanKeepTheirTopCard() {
        setup(4);
        addCard(Zone.BATTLEFIELD, playerB, "Plains", 2);
        addCard(Zone.HAND, playerB, "Raise the Alarm");
        castSpell(1, PhaseStep.UPKEEP, playerB, "Raise the Alarm", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, "Soldier Token");
        setChoice(playerB, false);
        addTarget(playerA, TestPlayer.TARGET_SKIP);
        finish();
        assertPermanentCount(playerB, "Soldier Token", 1);
        assertLibraryCount(playerB, "Soldier Token", 0);
        assertGraveyardCount(playerB, "Soldier Token", 0);
        assertLibraryCount(playerA, "Forest", 2);
    }

    @Test
    public void emptyCasterLibrarySkipsSurveilWithoutDrawing() {
        skipInitShuffling();
        removeAllCardsFromLibrary(playerA);
        addCard(Zone.BATTLEFIELD, playerA, "Island", 4);
        addCard(Zone.HAND, playerA, VOYAGE);
        addCard(Zone.LIBRARY, playerB, "Mountain", 2);
        addCard(Zone.BATTLEFIELD, playerB, ELF);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, ELF);
        setChoice(playerB, false);
        finish();
        assertLibraryCount(playerA, 0);
        assertLife(playerA, 20);
        assertLibraryCount(playerB, ELF, 1);
        assertGraveyardCount(playerA, VOYAGE, 1);
    }

    @Test
    public void wardTwoIsPaidBeforeOwnerPlacement() {
        setup(6);
        addCard(Zone.BATTLEFIELD, playerB, "Cackling Prowler");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, "Cackling Prowler");
        setChoice(playerA, true);
        setChoice(playerB, true);
        addTarget(playerA, TestPlayer.TARGET_SKIP);
        finish();
        assertLibraryCount(playerB, "Cackling Prowler", 1);
        assertTappedCount("Island", true, 6);
    }

    @Test
    public void decliningWardSkipsPlacementAndSurveil() {
        setup(6);
        addCard(Zone.BATTLEFIELD, playerB, "Cackling Prowler");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, VOYAGE, "Cackling Prowler");
        setChoice(playerA, false);
        finish();
        assertPermanentCount(playerB, "Cackling Prowler", 1);
        assertLibraryCount(playerA, "Forest", 2);
        assertGraveyardCount(playerA, VOYAGE, 1);
        assertTappedCount("Island", true, 4);
    }
}
