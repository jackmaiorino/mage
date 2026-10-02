package org.mage.test.cards.single.fdn;

import mage.abilities.keyword.FlyingAbility;
import mage.abilities.keyword.HexproofAbility;
import mage.abilities.keyword.IndestructibleAbility;
import mage.constants.PhaseStep;
import mage.constants.Zone;
import org.junit.Test;
import org.mage.test.serverside.base.CardTestPlayerBase;

/** Strict reference cases separating entry protection from attachment bonuses. */
public class FdnCelestialArmorTest extends CardTestPlayerBase {
    private static final String ARMOR = "Celestial Armor";
    private static final String BEAR = "Grizzly Bears";

    private void setup() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 7);
        addCard(Zone.BATTLEFIELD, playerA, BEAR);
        addCard(Zone.HAND, playerA, ARMOR);
        addTarget(playerA, BEAR);
    }

    private void finish() {
        setStrictChooseMode(true);
        setStopAt(1, PhaseStep.BEGIN_COMBAT);
        execute();
    }

    private void protection(boolean expected) {
        assertAbility(playerA, BEAR, HexproofAbility.getInstance(), expected);
        assertAbility(playerA, BEAR, IndestructibleAbility.getInstance(), expected);
    }

    @Test
    public void entryAttachesAndGrantsFlyingPowerAndTemporaryProtection() {
        setup();
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        finish();
        assertAttachedTo(playerA, ARMOR, BEAR, true);
        assertPowerToughness(playerA, BEAR, 4, 2);
        assertAbility(playerA, BEAR, FlyingAbility.getInstance(), true);
        protection(true);
    }

    @Test
    public void flashAllowsEntryDuringTheOpponentsTurn() {
        setup();
        castSpell(2, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.BEGIN_COMBAT);
        execute();
        assertAttachedTo(playerA, ARMOR, BEAR, true);
        assertPowerToughness(playerA, BEAR, 4, 2);
        protection(true);
    }

    @Test
    public void noCreatureStillAllowsArmorToResolve() {
        addCard(Zone.BATTLEFIELD, playerA, "Plains", 3);
        addCard(Zone.HAND, playerA, ARMOR);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        finish();
        assertPermanentCount(playerA, ARMOR, 1);
        assertTappedCount("Plains", true, 3);
    }

    @Test
    public void cleanupExpiresProtectionButPreservesTheEquipmentBonus() {
        setup();
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        setStrictChooseMode(true);
        setStopAt(2, PhaseStep.UPKEEP);
        execute();
        assertAttachedTo(playerA, ARMOR, BEAR, true);
        assertPowerToughness(playerA, BEAR, 4, 2);
        assertAbility(playerA, BEAR, FlyingAbility.getInstance(), true);
        protection(false);
    }

    @Test
    public void equipMovesTheBonusAndLeavesProtectionOnTheEntryTarget() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Elvish Mystic");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR, true);
        activateAbility(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Equip {3}{W}", "Elvish Mystic");
        finish();
        assertAttachedTo(playerA, ARMOR, "Elvish Mystic", true);
        assertPowerToughness(playerA, BEAR, 2, 2);
        assertPowerToughness(playerA, "Elvish Mystic", 3, 1);
        assertAbility(playerA, BEAR, FlyingAbility.getInstance(), false);
        assertAbility(playerA, "Elvish Mystic", FlyingAbility.getInstance(), true);
        assertAbility(playerA, "Elvish Mystic", HexproofAbility.getInstance(), false);
        assertAbility(playerA, "Elvish Mystic", IndestructibleAbility.getInstance(), false);
        protection(true);
    }

    @Test
    public void removingAttachedArmorLeavesTemporaryProtection() {
        setup();
        addCard(Zone.HAND, playerA, "Disenchant");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Disenchant", ARMOR);
        finish();
        assertGraveyardCount(playerA, ARMOR, 1);
        assertPowerToughness(playerA, BEAR, 2, 2);
        assertAbility(playerA, BEAR, FlyingAbility.getInstance(), false);
        protection(true);
    }

    @Test
    public void sourceRemovedBeforeEntryTriggerStillGrantsProtection() {
        setup();
        addCard(Zone.HAND, playerA, "Disenchant");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        waitStackResolved(1, PhaseStep.PRECOMBAT_MAIN, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Disenchant", ARMOR);
        finish();
        assertGraveyardCount(playerA, ARMOR, 1);
        assertPowerToughness(playerA, BEAR, 2, 2);
        assertAbility(playerA, BEAR, FlyingAbility.getInstance(), false);
        protection(true);
    }

    @Test
    public void targetRemovedBeforeEntryTriggerLeavesArmorUnattached() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Island");
        addCard(Zone.HAND, playerA, "Unsummon");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR);
        waitStackResolved(1, PhaseStep.PRECOMBAT_MAIN, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Unsummon", BEAR);
        finish();
        assertHandCount(playerA, BEAR, 1);
        assertPermanentCount(playerA, ARMOR, 1);
    }

    @Test
    public void entryIndestructiblePreventsLethalDamage() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Mountain");
        addCard(Zone.HAND, playerA, "Lightning Bolt");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Lightning Bolt", BEAR);
        finish();
        assertPowerToughness(playerA, BEAR, 4, 2);
        protection(true);
    }

    @Test
    public void entryIndestructiblePreventsDestroyButNotZeroToughness() {
        setup();
        addCard(Zone.BATTLEFIELD, playerA, "Swamp", 2);
        addCard(Zone.HAND, playerA, "Cast Down");
        addCard(Zone.HAND, playerA, "Disfigure");
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, ARMOR, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Cast Down", BEAR, true);
        castSpell(1, PhaseStep.PRECOMBAT_MAIN, playerA, "Disfigure", BEAR);
        finish();
        assertGraveyardCount(playerA, BEAR, 1);
        assertPermanentCount(playerA, ARMOR, 1);
    }
}
