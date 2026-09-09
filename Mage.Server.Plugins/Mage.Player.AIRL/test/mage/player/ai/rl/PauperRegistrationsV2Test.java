package mage.player.ai.rl;

import mage.cards.decks.DeckCardInfo;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.CardLookup;
import mage.cards.decks.importer.DekDeckImporter;
import mage.cards.repository.CardCriteria;
import mage.cards.repository.CardInfo;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * JUnit coverage for the Pauper deck roster used by DeterminizationSampler.
 * <p>
 * Written for task-4-brief.md (2026-09-09 pauper-meta-wave1): the nine V2
 * deck revisions are not yet wired into pauperRegistrationsV2() (that
 * roster append is Task 12), so this class checks two separate things:
 * <ol>
 *     <li>{@link DeterminizationSampler#pauperRegistrationsV2()} loads the
 *     nine historical archetypes (proving every historical .dek path is
 *     still wired and present), and each of the underlying .dek files has
 *     60 mainboard cards.</li>
 *     <li>Each of the nine new V2 {@code .dek} files (not yet in any
 *     roster) imports cleanly through the same importer class the
 *     sampler's {@code loadDeck(String)} uses for {@code .dek} files
 *     ({@link DekDeckImporter}, dispatched to by {@code DeckImporter}
 *     for the .dek extension), with 60 mainboard and 15 sideboard
 *     cards.</li>
 * </ol>
 * <p>
 * Card-name resolution is stubbed with an always-matching {@link CardLookup}
 * rather than the real {@code CardRepository}: this module (AIRL) does not
 * depend on Mage.Sets, and a bare offline `mvn -pl
 * Mage.Server.Plugins/Mage.Player.AIRL test` run has no populated, schema-
 * matched card database to resolve real set/collector-number data against
 * (verified: CardRepository starts from an empty table here even when a
 * fully populated cards.h2.mv.db from another checkout is dropped in,
 * because CardRepository.CARD_DB_VERSION mismatches drop and recreate the
 * table). This is the same technique the project's own import test uses for
 * this exact importer class -- see
 * Mage/src/test/java/mage/cards/decks/importer/DekDeckImportTest.java and
 * its FakeCardLookup. Both checks below stop at DekDeckImporter's parsed
 * {@link DeckCardLists} (Quantity/Sideboard/Name attribute extraction) --
 * exactly what DekDeckImportTest checks -- rather than continuing on to
 * mage.cards.decks.Deck#load, which re-resolves every card a second time
 * against the live CardRepository by set code and collector number
 * (independent of the CardLookup used during import) and so cannot be
 * faked the same way; that step needs a real, version-matched card
 * database and is out of scope for a deck-file-writer task.
 * <p>
 * Run with:
 * {@code mvn -o -q -pl Mage.Server.Plugins/Mage.Player.AIRL -Dtest=PauperRegistrationsV2Test -DfailIfNoTests=false test}
 */
public class PauperRegistrationsV2Test {

    private static final String PAUPER_BASE =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper";

    private static final String[] HISTORICAL_ARCHETYPES = {
            "Wildfire", "Rally", "Affinity", "Elves", "SpyCombo",
            "Burn", "Terror", "CawGates", "Faeries",
    };

    // Mirrors DeterminizationSampler.pauperRegistrationsV2()'s own
    // hardcoded archetype -> file map (that method exposes archetype names
    // via getArchetypes() but not the underlying paths).
    private static final Map<String, String> HISTORICAL_FILES = new LinkedHashMap<>();

    static {
        HISTORICAL_FILES.put("Wildfire", PAUPER_BASE + "/Deck - Jund Wildfire.dek");
        HISTORICAL_FILES.put("Rally", PAUPER_BASE + "/Deck - Mono Red Rally.dek");
        HISTORICAL_FILES.put("Affinity", PAUPER_BASE + "/Deck - Grixis Affinity.dek");
        HISTORICAL_FILES.put("Elves", PAUPER_BASE + "/Deck - Elves.dek");
        HISTORICAL_FILES.put("SpyCombo", PAUPER_BASE + "/Deck - Spy Combo.dek");
        HISTORICAL_FILES.put("Burn", PAUPER_BASE + "/Deck - Mono-Red Burn.dek");
        HISTORICAL_FILES.put("Terror", PAUPER_BASE + "/Deck - Mono-Blue Terror.dek");
        HISTORICAL_FILES.put("CawGates", PAUPER_BASE + "/Deck - Caw-Gates.dek");
        HISTORICAL_FILES.put("Faeries", PAUPER_BASE + "/Deck - Mono-Blue Faeries.dek");
    }

    private static final Map<String, String> V2_FILES = new LinkedHashMap<>();

    static {
        V2_FILES.put("MadnessBurnV2", PAUPER_BASE + "/Deck - Madness Burn V2.dek");
        V2_FILES.put("MonoBlueDelverV2", PAUPER_BASE + "/Deck - Mono-Blue Delver V2.dek");
        V2_FILES.put("GrixisAffinityV2", PAUPER_BASE + "/Deck - Grixis Affinity V2.dek");
        V2_FILES.put("RedDeckWinsV2", PAUPER_BASE + "/Deck - Red Deck Wins V2.dek");
        V2_FILES.put("JundWildfireV2", PAUPER_BASE + "/Deck - Jund Wildfire V2.dek");
        V2_FILES.put("ElvesV2", PAUPER_BASE + "/Deck - Elves V2.dek");
        V2_FILES.put("MonoBlueTerrorV2", PAUPER_BASE + "/Deck - Mono-Blue Terror V2.dek");
        V2_FILES.put("SpyComboV2", PAUPER_BASE + "/Deck - Spy Combo V2.dek");
        V2_FILES.put("DimirTerrorV2", PAUPER_BASE + "/Deck - Dimir Terror V2.dek");
    }

    /** Matches any card name; see the class Javadoc for why this is needed here. */
    private static final class AlwaysMatchesCardLookup extends CardLookup {
        @Override
        public CardInfo lookupCardInfo(String name) {
            return new CardInfo() {{
                this.name = name;
            }};
        }

        @Override
        public List<CardInfo> lookupCardInfo(CardCriteria criteria) {
            return Collections.singletonList(lookupCardInfo(criteria.getName()));
        }
    }

    private static DeckCardLists importDek(String path) {
        DekDeckImporter importer = new DekDeckImporter() {
            @Override
            public CardLookup getCardLookup() {
                return new AlwaysMatchesCardLookup();
            }
        };
        StringBuilder errors = new StringBuilder();
        DeckCardLists lists = importer.importDeck(path, errors, false);
        Assert.assertEquals(path + " import warnings", "", errors.toString());
        return lists;
    }

    private static int totalAmount(List<DeckCardInfo> entries) {
        return entries.stream().mapToInt(DeckCardInfo::getAmount).sum();
    }

    @Test
    public void pauperRegistrationsV2HistoricalArchetypesHave60MainboardCards() {
        DeterminizationSampler sampler = DeterminizationSampler.pauperRegistrationsV2();
        Assert.assertNotNull("pauperRegistrationsV2() must load", sampler);
        Assert.assertEquals(
                "pauperRegistrationsV2() archetype roster",
                Arrays.asList(HISTORICAL_ARCHETYPES),
                sampler.getArchetypes()
        );

        for (String archetype : HISTORICAL_ARCHETYPES) {
            String path = HISTORICAL_FILES.get(archetype);
            DeckCardLists lists = importDek(path);
            int mainboard = totalAmount(lists.getCards());
            Assert.assertEquals(archetype + " (" + path + ") mainboard count", 60, mainboard);
        }
    }

    @Test
    public void nineV2DekFilesImportSixtyMainboardFifteenSideboard() {
        for (Map.Entry<String, String> entry : V2_FILES.entrySet()) {
            String label = entry.getKey();
            String path = entry.getValue();

            DeckCardLists lists = importDek(path);
            int mainboard = totalAmount(lists.getCards());
            int sideboard = totalAmount(lists.getSideboard());
            Assert.assertEquals(label + " (" + path + ") mainboard count", 60, mainboard);
            Assert.assertEquals(label + " (" + path + ") sideboard count", 15, sideboard);
        }
    }
}
