package mage.player.ai.rl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.log4j.Logger;

import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.game.GameException;

/**
 * Caches fully materialized deck templates and returns deep copies on demand.
 */
final class DeckTemplateCache {

    private static final class Entry {
        final long lastModifiedMillis;
        final Deck template;

        Entry(long lastModifiedMillis, Deck template) {
            this.lastModifiedMillis = lastModifiedMillis;
            this.template = template;
        }
    }

    private final ConcurrentHashMap<String, Entry> cache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Object> locks = new ConcurrentHashMap<>();

    Deck load(String filePath, Logger logger) {
        Path path = Paths.get(filePath).toAbsolutePath().normalize();
        String key = path.toString();
        long mtime;
        try {
            mtime = Files.getLastModifiedTime(path).toMillis();
        } catch (IOException e) {
            if (logger != null) {
                logger.error("Error getting deck timestamp: " + filePath, e);
            }
            return null;
        }

        Entry cached = cache.get(key);
        if (cached != null && cached.lastModifiedMillis == mtime) {
            return cached.template.copy();
        }

        Object lock = locks.computeIfAbsent(key, ignored -> new Object());
        synchronized (lock) {
            cached = cache.get(key);
            if (cached != null && cached.lastModifiedMillis == mtime) {
                return cached.template.copy();
            }
            try {
                StringBuilder importWarnings = new StringBuilder();
                DeckCardLists deckCardLists = DeckImporter.importDeckFromFile(filePath, importWarnings, false);
                if (importWarnings.length() > 0 && logger != null) {
                    logger.warn("Deck import warnings for " + filePath + ":\n" + importWarnings);
                }
                Deck deck = Deck.load(deckCardLists, false, false, null);
                if (deck != null && logger != null) {
                    int mainCount = deck.getCards().size();
                    int sideCount = deck.getSideboard().size();
                    if (mainCount != 60) {
                        logger.warn("Deck mainboard size is " + mainCount + " (expected 60) for: " + filePath
                                + " (sideboard=" + sideCount + ")");
                    }
                }
                if (deck != null) {
                    cache.put(key, new Entry(mtime, deck));
                    return deck.copy();
                }
                return null;
            } catch (GameException e) {
                if (logger != null) {
                    logger.error("Error loading deck: " + filePath, e);
                }
                return null;
            }
        }
    }
}
