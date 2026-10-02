package org.mage.test.cards.single.fdn;

import mage.ObjectColor;
import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.IndestructibleAbility;
import mage.abilities.keyword.ReachAbility;
import mage.abilities.keyword.TrampleAbility;
import mage.constants.CardType;
import mage.constants.PhaseStep;
import mage.constants.SubType;
import mage.constants.SuperType;
import mage.constants.Zone;
import org.junit.Assert;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Strict reference cases for the kernel's Witness characteristic layers. */
public class FdnWitnessProtectionTest extends CardTestPlayerBase {
    private static final String WITNESS = "Witness Protection";
    private static final String BUSINESS = "Legitimate Businessperson";
    private static final String SPINNER = "Snarespinner";
    private static final String ARMOR = "Celestial Armor";
    private static final String DWYNEN = "Dwynen, Gilt-Leaf Daen";

    private void setup(String creature) {
        addCard(Zone.BATTLEFIELD, playerA, "Island", 3);
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 10);
        addCard(Zone.BATTLEFIELD, playerA, creature);
        addCard(Zone.HAND, playerA, WITNESS);
    }

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
    }

    @Test
    public void nameColorsSubtypeBaseAndAbilityRemovalAreExact() {
        setup(SPINNER);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER);
        finish();
        assertPermanentCount(playerA, BUSINESS, 1);
        assertAttachedTo(playerA, WITNESS, BUSINESS, true);
        assertPowerToughness(playerA, BUSINESS, 1, 1);
        assertColor(playerA, BUSINESS, ObjectColor.GREEN, true);
        assertColor(playerA, BUSINESS, ObjectColor.WHITE, true);
        assertColor(playerA, BUSINESS, ObjectColor.BLUE, false);
        assertSubtype(BUSINESS, SubType.CITIZEN);
        Assert.assertFalse(getPermanent(BUSINESS, playerA).hasSubtype(SubType.SPIDER, currentGame));
        assertAbility(playerA, BUSINESS, ReachAbility.getInstance(), false);
    }

    @Test
    public void creatureOnlyOverrideRemovesArtifactType() {
        setup("Memnite");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Memnite");
        finish();
        assertType(BUSINESS, CardType.CREATURE, true);
        assertType(BUSINESS, CardType.ARTIFACT, false);
    }

    @Test
    public void legendarySupertypeSurvives() {
        setup(DWYNEN);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, DWYNEN);
        finish();
        Assert.assertTrue(getPermanent(BUSINESS, playerA).getSuperType().contains(SuperType.LEGENDARY));
        assertPowerToughness(playerA, BUSINESS, 1, 1);
    }

    @Test
    public void olderArmorLosesFlyingAndProtectionButRetainsPowerBonus() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, ARMOR);
        addTarget(playerA, SPINNER);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER);
        finish();
        assertPowerToughness(playerA, BUSINESS, 3, 1);
        assertAbility(playerA, BUSINESS, FlyingAbility.getInstance(), false);
        assertAbility(playerA, BUSINESS, HexproofAbility.getInstance(), false);
        assertAbility(playerA, BUSINESS, IndestructibleAbility.getInstance(), false);
        assertAbility(playerA, BUSINESS, ReachAbility.getInstance(), false);
    }

    @Test
    public void laterArmorGrantsFlyingAndProtection() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, ARMOR);
        addTarget(playerA, BUSINESS);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        finish();
        assertPowerToughness(playerA, BUSINESS, 3, 1);
        assertAbility(playerA, BUSINESS, FlyingAbility.getInstance(), true);
        assertAbility(playerA, BUSINESS, HexproofAbility.getInstance(), true);
        assertAbility(playerA, BUSINESS, IndestructibleAbility.getInstance(), true);
        assertAbility(playerA, BUSINESS, ReachAbility.getInstance(), false);
    }

    @Test
    public void olderFlightKeepsCounterButLosesFlying() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, "Fleeting Flight");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Fleeting Flight", SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER);
        finish();
        assertPowerToughness(playerA, BUSINESS, 2, 2);
        assertAbility(playerA, BUSINESS, FlyingAbility.getInstance(), false);
    }

    @Test
    public void laterFlightGrantsFlyingAndCounter() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, "Fleeting Flight");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Fleeting Flight", BUSINESS);
        finish();
        assertPowerToughness(playerA, BUSINESS, 2, 2);
        assertAbility(playerA, BUSINESS, FlyingAbility.getInstance(), true);
        assertAbility(playerA, BUSINESS, ReachAbility.getInstance(), false);
    }

    @Test
    public void removingLordAbilitiesStopsBoostingOtherElves() {
        setup(DWYNEN);
        addCard(Zone.BATTLEFIELD, playerA, "Llanowar Elves");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, DWYNEN);
        finish();
        assertPowerToughness(playerA, "Llanowar Elves", 1, 1);
        assertPowerToughness(playerA, BUSINESS, 1, 1);
    }

    @Test
    public void removingColonyAbilitiesStopsItsTrampleGrant() {
        setup("Gnarlid Colony");
        addCard(Zone.BATTLEFIELD, playerA, SPINNER);
        addCard(Zone.HAND, playerA, "Fleeting Flight");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Fleeting Flight", SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Gnarlid Colony");
        finish();
        assertAbility(playerA, SPINNER, TrampleAbility.getInstance(), false);
        assertAbility(playerA, SPINNER, FlyingAbility.getInstance(), true);
    }

    @Test
    public void transformedHeirDoesNotCreateAKnightWhenItDies() {
        setup("Guarded Heir");
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Guarded Heir", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", BUSINESS);
        finish();
        assertGraveyardCount(playerA, "Guarded Heir", 1);
        assertGraveyardCount(playerA, WITNESS, 1);
        assertPermanentCount(playerA, "Knight Token", 0);
    }

    @Test
    public void removingWitnessBeforeHeirDiesRestoresItsDeathTrigger() {
        setup("Guarded Heir");
        addCard(Zone.BATTLEFIELD, playerA, "Swamp", 2);
        addCard(Zone.HAND, playerA, "Disenchant");
        addCard(Zone.HAND, playerA, "Doom Blade");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Guarded Heir", true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Disenchant", WITNESS, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Doom Blade", "Guarded Heir");
        finish();
        assertGraveyardCount(playerA, "Guarded Heir", 1);
        assertPermanentCount(playerA, "Knight Token", 1);
    }

    @Test
    public void auraRemovalRestoresPrintedCharacteristics() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, "Disenchant");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Disenchant", WITNESS);
        finish();
        assertPowerToughness(playerA, SPINNER, 1, 3);
        assertAbility(playerA, SPINNER, ReachAbility.getInstance(), true);
        assertPermanentCount(playerA, BUSINESS, 0);
    }

    @Test
    public void bouncingTheHostRemovesTheAuraAndRestoresTheHandCardName() {
        setup(SPINNER);
        addCard(Zone.HAND, playerA, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, SPINNER, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Unsummon", BUSINESS);
        finish();
        assertHandCount(playerA, SPINNER, 1);
        assertGraveyardCount(playerA, WITNESS, 1);
    }

    @Test
    public void manaCreatureLosesItsPrintedActivatedAbility() {
        setup("Llanowar Elves");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Llanowar Elves");
        finish();
        Assert.assertTrue(getPermanent(BUSINESS, playerA).getAbilities().isEmpty());
    }

    @Test
    public void sailorLosesItsPrintedDrawActivationAndKeywords() {
        setup("Spectral Sailor");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Spectral Sailor");
        finish();
        Assert.assertTrue(getPermanent(BUSINESS, playerA).getAbilities().isEmpty());
    }

    @Test
    public void transformedKomaDoesNotTriggerWardForTheOpponent() {
        setup("Koma, World-Eater");
        addCard(Zone.BATTLEFIELD, playerB, "Mountain");
        addCard(Zone.HAND, playerB, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, WITNESS, "Koma, World-Eater", true);
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerB, "Lightning Bolt", BUSINESS);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.BEGIN_COMBAT);
        execute();
        assertGraveyardCount(playerA, "Koma, World-Eater", 1);
        assertGraveyardCount(playerB, "Lightning Bolt", 1);
    }
}
