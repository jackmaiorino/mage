package mage.player.ai.rl;

import mage.cards.Card;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;
import mage.util.RandomUtil;

import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import mage.counters.Counter;
import mage.counters.Counters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

/**
 * Opt-in durable checkpoint capture for live accepted-policy evaluation games.
 */
public final class LiveCheckpointRecorder {

    private static final boolean ENABLED = EnvConfig.bool("EVAL_LIVE_CHECKPOINTS", false);
    private static final int MIN_CANDIDATES = Math.max(2, EnvConfig.i32("EVAL_LIVE_CHECKPOINT_MIN_CANDIDATES", 2));
    private static final int MAX_PER_GAME = Math.max(1, EnvConfig.i32("EVAL_LIVE_CHECKPOINT_MAX_PER_GAME", 96));
    private static final Set<String> ACTION_TYPES = parseActionTypes(EnvConfig.str(
            "EVAL_LIVE_CHECKPOINT_ACTION_TYPES",
            "ACTIVATE_ABILITY_OR_SPELL,SELECT_TARGETS,SELECT_CARD,DECLARE_ATTACKS,DECLARE_BLOCKS,CHOOSE_USE,CHOOSE_MODE,ANNOUNCE_X"));
    private static final Object MANIFEST_LOCK = new Object();
    private static final ConcurrentHashMap<String, AtomicInteger> GAME_COUNTS = new ConcurrentHashMap<>();

    private LiveCheckpointRecorder() {
    }

    public static void maybeCapture(
            Game game,
            Player player,
            StateSequenceBuilder.ActionType actionType,
            int ordinal,
            int decisionNumber,
            int recordId,
            List<String> candidateTexts,
            List<String> candidateObjectIds,
            List<Integer> selectedIndices,
            List<String> selectedTexts,
            List<String> selectedObjectIds,
            float selectedProb,
            float valueScore
    ) {
        if (!ENABLED || game == null || player == null || actionType == null) {
            return;
        }
        if (game.isSimulation() || candidateTexts == null || candidateTexts.size() < MIN_CANDIDATES) {
            return;
        }
        if (!ACTION_TYPES.isEmpty() && !ACTION_TYPES.contains(actionType.name())) {
            return;
        }

        String gameKey = safeGameKey(game);
        AtomicInteger counter = GAME_COUNTS.computeIfAbsent(gameKey, k -> new AtomicInteger());
        int captureIndex = counter.incrementAndGet();
        if (captureIndex > MAX_PER_GAME) {
            return;
        }

        Path root = resolveRoot();
        String fileName = String.format(Locale.US, "game_%s_ord%03d_D%03d_%s.ser.gz",
                sanitize(gameKey), ordinal, decisionNumber, sanitize(actionType.name()));
        Path snapshotPath = root.resolve(fileName);

        try {
            Files.createDirectories(root);
            RandomUtil.State randomState = RandomUtil.captureState();
            Game gameSnapshot = game.createSimulationForAI();
            Snapshot snapshot = new Snapshot(
                    gameSnapshot,
                    player.getId(),
                    player.getName(),
                    ordinal,
                    decisionNumber,
                    recordId,
                    actionType.name(),
                    candidateTexts,
                    candidateObjectIds,
                    selectedIndices,
                    selectedTexts,
                    selectedObjectIds,
                    selectedProb,
                    valueScore,
                    compactState(game, player),
                    randomState);
            try (ObjectOutputStream out = new ObjectOutputStream(
                    new GZIPOutputStream(Files.newOutputStream(snapshotPath,
                            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)))) {
                out.writeObject(snapshot);
            }
            appendManifest(root, snapshot, snapshotPath, "captured", "");
        } catch (Throwable t) {
            Snapshot failed = Snapshot.failed(
                    player.getId(),
                    player.getName(),
                    ordinal,
                    decisionNumber,
                    recordId,
                    actionType.name(),
                    candidateTexts,
                    candidateObjectIds,
                    selectedIndices,
                    selectedTexts,
                    selectedObjectIds,
                    selectedProb,
                    valueScore,
                    compactState(game, player),
                    RandomUtil.captureState());
            appendManifest(root, failed, snapshotPath, "error", errorSummary(t));
        }
    }

    public static final class Snapshot implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String schemaVersion;
        public final long createdAtMillis;
        public final Game gameSnapshot;
        public final UUID playerId;
        public final String playerName;
        public final int ordinal;
        public final int decisionNumber;
        public final int recordId;
        public final String actionType;
        public final List<String> candidateTexts;
        public final List<String> candidateObjectIds;
        public final List<Integer> selectedIndices;
        public final List<String> selectedTexts;
        public final List<String> selectedObjectIds;
        public final float selectedProb;
        public final float valueScore;
        public final String compactState;
        public final RandomUtil.State randomState;
        public final String candidateHash;
        public final String stateHash;
        public final String randomStateHash;

        private Snapshot(
                Game gameSnapshot,
                UUID playerId,
                String playerName,
                int ordinal,
                int decisionNumber,
                int recordId,
                String actionType,
                List<String> candidateTexts,
                List<String> candidateObjectIds,
                List<Integer> selectedIndices,
                List<String> selectedTexts,
                List<String> selectedObjectIds,
                float selectedProb,
                float valueScore,
                String compactState,
                RandomUtil.State randomState
        ) {
            this.schemaVersion = "live-checkpoint-v1";
            this.createdAtMillis = System.currentTimeMillis();
            this.gameSnapshot = gameSnapshot;
            this.playerId = playerId;
            this.playerName = playerName == null ? "" : playerName;
            this.ordinal = ordinal;
            this.decisionNumber = decisionNumber;
            this.recordId = recordId;
            this.actionType = actionType == null ? "" : actionType;
            this.candidateTexts = strings(candidateTexts);
            this.candidateObjectIds = strings(candidateObjectIds);
            this.selectedIndices = ints(selectedIndices);
            this.selectedTexts = strings(selectedTexts);
            this.selectedObjectIds = strings(selectedObjectIds);
            this.selectedProb = selectedProb;
            this.valueScore = valueScore;
            this.compactState = compactState == null ? "" : compactState;
            this.randomState = randomState;
            this.candidateHash = sha256(String.join("\n", this.candidateTexts));
            this.stateHash = sha256(this.compactState);
            this.randomStateHash = randomState == null ? "" : randomState.fingerprint();
        }

        private static Snapshot failed(
                UUID playerId,
                String playerName,
                int ordinal,
                int decisionNumber,
                int recordId,
                String actionType,
                List<String> candidateTexts,
                List<String> candidateObjectIds,
                List<Integer> selectedIndices,
                List<String> selectedTexts,
                List<String> selectedObjectIds,
                float selectedProb,
                float valueScore,
                String compactState,
                RandomUtil.State randomState
        ) {
            return new Snapshot(null, playerId, playerName, ordinal, decisionNumber, recordId, actionType,
                    candidateTexts, candidateObjectIds, selectedIndices, selectedTexts, selectedObjectIds,
                    selectedProb, valueScore, compactState, randomState);
        }
    }

    private static Path resolveRoot() {
        String explicit = EnvConfig.str("EVAL_LIVE_CHECKPOINT_DIR", "");
        if (!explicit.isEmpty()) {
            return Paths.get(explicit);
        }
        String gameLogDir = EnvConfig.str("GAME_LOG_DIR", "");
        if (!gameLogDir.isEmpty()) {
            return Paths.get(gameLogDir).resolve("live_checkpoints");
        }
        return Paths.get("local-training", "local_pbt", "live_checkpoints");
    }

    private static void appendManifest(Path root, Snapshot snapshot, Path snapshotPath, String status, String error) {
        try {
            Files.createDirectories(root);
            Path manifest = root.resolve("manifest.csv");
            String header = "status,snapshot_path,schema_version,player_id,player_name,ordinal,decision_number,"
                    + "record_id,"
                    + "action_type,candidate_count,selected_indices,selected_texts,selected_object_ids,"
                    + "selected_prob,value_score,candidate_hash,state_hash,rng_state_hash,error\n";
            String line = csv(status)
                    + "," + csv(snapshotPath == null ? "" : snapshotPath.toString())
                    + "," + csv(snapshot == null ? "" : snapshot.schemaVersion)
                    + "," + csv(snapshot == null || snapshot.playerId == null ? "" : snapshot.playerId.toString())
                    + "," + csv(snapshot == null ? "" : snapshot.playerName)
                    + "," + (snapshot == null ? -1 : snapshot.ordinal)
                    + "," + (snapshot == null ? -1 : snapshot.decisionNumber)
                    + "," + (snapshot == null ? -1 : snapshot.recordId)
                    + "," + csv(snapshot == null ? "" : snapshot.actionType)
                    + "," + (snapshot == null ? 0 : snapshot.candidateTexts.size())
                    + "," + csv(snapshot == null ? "" : joinInts(snapshot.selectedIndices))
                    + "," + csv(snapshot == null ? "" : String.join("|", snapshot.selectedTexts))
                    + "," + csv(snapshot == null ? "" : String.join("|", snapshot.selectedObjectIds))
                    + "," + String.format(Locale.US, "%.8f", snapshot == null ? 0.0f : snapshot.selectedProb)
                    + "," + String.format(Locale.US, "%.8f", snapshot == null ? 0.0f : snapshot.valueScore)
                    + "," + csv(snapshot == null ? "" : snapshot.candidateHash)
                    + "," + csv(snapshot == null ? "" : snapshot.stateHash)
                    + "," + csv(snapshot == null ? "" : snapshot.randomStateHash)
                    + "," + csv(error)
                    + "\n";
            synchronized (MANIFEST_LOCK) {
                boolean writeHeader = !Files.exists(manifest) || Files.size(manifest) == 0;
                if (writeHeader) {
                    Files.write(manifest, header.getBytes(StandardCharsets.UTF_8),
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                }
                Files.write(manifest, line.getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            }
        } catch (Throwable ignored) {
            // Diagnostic checkpoint capture must never affect game execution.
        }
    }

    public static String compactState(Game game, Player perspective) {
        StringBuilder sb = new StringBuilder(2048);
        try {
            sb.append("game=").append(game.getId())
                    .append(";turn=").append(game.getTurnNum())
                    .append(";phase=").append(game.getTurnPhaseType())
                    .append(";step=").append(game.getTurnStepType())
                    .append(";active=").append(game.getActivePlayerId())
                    .append(";priority=").append(game.getPriorityPlayerId())
                    .append(";choosing=").append(game.getState() == null ? "" : game.getState().getChoosingPlayerId())
                    .append(";perspective=").append(perspective == null ? "" : perspective.getId());
            List<Player> players = new ArrayList<>();
            for (UUID id : game.getPlayerList()) {
                Player p = game.getPlayer(id);
                if (p != null) {
                    players.add(p);
                }
            }
            players.sort(Comparator.comparing(p -> p.getId() == null ? "" : p.getId().toString()));
            for (Player p : players) {
                sb.append(";player=").append(p.getId())
                        .append(":").append(p.getName())
                        .append(":life=").append(p.getLife())
                        .append(":hand=").append(p.getHand() == null ? -1 : p.getHand().size())
                        .append(":library=").append(p.getLibrary() == null ? -1 : p.getLibrary().size())
                        .append(":graveyard=").append(p.getGraveyard() == null ? -1 : p.getGraveyard().size())
                        .append(":mana=").append(p.getManaPool() == null ? "" : p.getManaPool().toString())
                        .append(":hand_cards=").append(cardsText(p.getHand() == null ? Collections.emptyList() : p.getHand().getCards(game), game, 0, "hand"))
                        .append(":library_top=").append(cardsText(p.getLibrary() == null ? Collections.emptyList() : p.getLibrary().getCards(game), game, 12, "lib"))
                        .append(":graveyard_cards=").append(cardsText(p.getGraveyard() == null ? Collections.emptyList() : p.getGraveyard().getCards(game), game, 0, "gy"));
            }
            List<Permanent> permanents = new ArrayList<>(game.getBattlefield().getAllActivePermanents());
            sb.append(";battlefield=");
            appendCanonicalBattlefield(sb, permanents, game);
            sb.append(";stack=");
            appendCanonicalStack(sb, game);
        } catch (Throwable t) {
            sb.append(";compact_state_error=").append(errorSummary(t));
        }
        return sb.toString();
    }

    /**
     * Sol #93/#95 canonicalization. StackAbility/Spell ids are freshly minted via
     * UUID.randomUUID() (AbilityImpl#newId/newOriginalId, non-seeded, not derived
     * from RandomUtil) every time something is put on the stack -- confirmed via
     * field-by-field diff as the SOLE differing field behind three
     * checkpoint-reentry "state hash mismatch" gate failures whose action type,
     * candidate set, and RNG fingerprint all matched exactly. Comparing raw stack
     * object ids across independent JVM runs is meaningless; comparing STACK
     * POSITION (bottom-up via descendingIterator -- SpellStack#push is
     * ArrayDeque#addFirst, so descendingIterator visits tail-to-head, i.e.
     * bottom-to-top), source id, controller, rule text, targets, and modes is the
     * reproducible substitute. X value is not separately extracted: XMage's rule
     * text already substitutes an announced X value into getRule()'s output, and
     * there is no single stable X accessor on the Ability interface to read it
     * from directly.
     */
    private static void appendCanonicalStack(StringBuilder sb, Game game) {
        List<StackObject> bottomUp = new ArrayList<>();
        java.util.Iterator<StackObject> it = game.getStack().descendingIterator();
        while (it.hasNext()) {
            bottomUp.add(it.next());
        }
        for (int i = 0; i < bottomUp.size(); i++) {
            StackObject so = bottomUp.get(i);
            if (so == null) {
                continue;
            }
            sb.append("stack#").append(i)
                    .append(":").append(so.getName())
                    .append(":ctrl=").append(so.getControllerId())
                    .append(":src=").append(so.getSourceId())
                    .append(":rule=").append(stackObjectRule(so))
                    .append(":targets=").append(stackObjectTargets(so, game))
                    .append(":modes=").append(stackObjectModes(so))
                    .append("|");
        }
    }

    /**
     * Sol #100 canonicalization. Permanent ids -- including token permanents,
     * e.g. the Blood token from Voldaren Epicure's ETB -- are minted via
     * CardImpl's {@code this.objectId = UUID.randomUUID()} (not
     * RandomUtil-seeded), the identical non-seeded mechanism documented on
     * {@link #appendCanonicalStack} for StackAbility/Spell ids. Confirmed via
     * the 40-point targeting campaign as the sole differing content behind
     * 3/40 state-hash-only mismatches (all Voldaren Epicure/Blood Token
     * games; the mismatch lands on the walk step immediately after the ETB
     * trigger resolves and the token enters the battlefield). Fix mirrors the
     * stack scheme: render "bf#N", N = the permanent's position within ITS
     * CONTROLLER's own battlefield insertion order -- never the raw id.
     * <p>
     * {@code Battlefield#field} is a {@code LinkedHashMap}, so
     * {@code getAllActivePermanents()} already returns permanents in a real,
     * action/RNG-derived insertion order that is reproducible across
     * independent replays of identical history. The pre-existing
     * {@code permanents.sort(...getId()...)} call this replaces was itself
     * part of the bug: sorting by the permanent's own raw (sometimes fresh)
     * id threw away that reproducible order and substituted an
     * id-value-dependent one. A stable sort by controller id (Java's
     * {@code List#sort} is stable) groups permanents per controller while
     * preserving insertion order within each group, so numbering restarts at
     * 0 per controller and depends only on controller identity (stable:
     * player ids come from the deserialized ancestor snapshot, never
     * re-minted mid-game) -- never on the permanent's own id.
     * <p>
     * Damage and counters are appended alongside tap/zcc because they are
     * rules-relevant state that was previously omitted entirely (not a
     * raw-id leak, but a gap found during this canonicalization sweep).
     */
    private static void appendCanonicalBattlefield(StringBuilder sb, List<Permanent> permanents, Game game) {
        List<Permanent> ordered = new ArrayList<>(permanents);
        ordered.sort(Comparator.comparing(p -> p == null || p.getControllerId() == null ? "" : p.getControllerId().toString()));
        Map<String, Integer> perControllerIndex = new HashMap<>();
        for (Permanent p : ordered) {
            if (p == null) {
                continue;
            }
            String ctrlKey = p.getControllerId() == null ? "" : p.getControllerId().toString();
            int idx = perControllerIndex.merge(ctrlKey, 1, Integer::sum) - 1;
            sb.append("bf#").append(idx)
                    .append(":").append(p.getName())
                    .append(":ctrl=").append(p.getControllerId())
                    .append(":tap=").append(p.isTapped())
                    .append(":zcc=").append(p.getZoneChangeCounter(game))
                    .append(":dmg=").append(p.getDamage())
                    .append(":counters=").append(canonicalCounters(p.getCounters(game)))
                    .append("|");
        }
    }

    private static String canonicalCounters(Counters counters) {
        if (counters == null || counters.isEmpty()) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        for (Counter counter : counters.values()) {
            parts.add(counter.getName() + "=" + counter.getCount());
        }
        Collections.sort(parts);
        return String.join(",", parts);
    }

    private static String stackObjectRule(StackObject so) {
        try {
            mage.abilities.Ability ability = so.getStackAbility();
            return ability == null ? "" : String.valueOf(ability.getRule());
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static String stackObjectTargets(StackObject so, Game game) {
        try {
            mage.abilities.Ability ability = so.getStackAbility();
            if (ability == null || ability.getTargets() == null) {
                return "";
            }
            List<String> parts = new ArrayList<>();
            for (mage.target.Target target : ability.getTargets()) {
                if (target == null || target.getTargets() == null) {
                    continue;
                }
                for (UUID id : target.getTargets()) {
                    parts.add(canonicalStackObjectId(game, id));
                }
            }
            Collections.sort(parts);
            return String.join(",", parts);
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static String stackObjectModes(StackObject so) {
        try {
            mage.abilities.Ability ability = so.getStackAbility();
            if (ability == null || ability.getModes() == null) {
                return "";
            }
            List<UUID> selected = ability.getModes().getSelectedModes();
            if (selected == null) {
                return "";
            }
            // Canonicalize by ordinal position within the modes map (insertion
            // ordered), not the raw mode id, in case mode ids are also freshly
            // minted per activation rather than stable per card definition.
            List<UUID> allModeIds = new ArrayList<>(ability.getModes().keySet());
            List<String> positions = new ArrayList<>();
            for (UUID sel : selected) {
                int idx = allModeIds.indexOf(sel);
                positions.add(idx >= 0 ? ("mode#" + idx) : "mode#?");
            }
            Collections.sort(positions);
            return String.join(",", positions);
        } catch (Throwable ignored) {
            return "";
        }
    }

    /**
     * Sol #93/#95 canonicalization entry point, reused by ComputerPlayerRL's
     * replay/candidate object-id builders so a decision whose legal
     * candidates/targets include stack objects (e.g. Pyroblast targeting a
     * spell) records a reproducible "stack#N" label instead of the raw,
     * process-local UUID. Non-stack ids (permanents, cards, players) are stable
     * within one continuous resumed game and are returned unchanged.
     */
    public static String canonicalStackObjectId(Game game, UUID rawId) {
        if (rawId == null) {
            return "";
        }
        if (game != null) {
            try {
                int idx = 0;
                java.util.Iterator<StackObject> it = game.getStack().descendingIterator();
                while (it.hasNext()) {
                    StackObject so = it.next();
                    if (so != null && rawId.equals(so.getId())) {
                        return "stack#" + idx;
                    }
                    idx++;
                }
            } catch (Throwable ignored) {
                // fall through to raw id
            }
        }
        return rawId.toString();
    }

    /**
     * Sol #100 canonicalization sweep. Cards themselves can also carry a
     * freshly-minted id (same {@code UUID.randomUUID()} mechanism as stack
     * objects and battlefield permanents) if a token is ever transiently
     * represented as a Card in one of these zones before state-based-action
     * cleanup removes it. No sampled campaign mismatch has hit this path yet
     * (the confirmed 3/40 failures were all battlefield-only), but the risk
     * is identical in kind, so the same position-label scheme is applied
     * pre-emptively: "{zoneLabel}#N", N = the card's index within THIS
     * player's iteration of the zone. {@code CardsImpl} is a
     * {@code LinkedHashSet<UUID>}, and {@code getCards(Game)} streams it in
     * that same insertion order, so the position label is reproducible
     * across independent replays regardless of the card's own raw id.
     */
    private static String cardsText(Iterable<Card> cards, Game game, int limit, String zoneLabel) {
        List<String> values = new ArrayList<>();
        if (cards != null) {
            int idx = 0;
            for (Card card : cards) {
                if (card == null) {
                    continue;
                }
                values.add(zoneLabel + "#" + idx + ":" + zoneName(card, game) + ":" + card.getName());
                idx++;
                if (limit > 0 && values.size() >= limit) {
                    break;
                }
            }
        }
        return String.join("|", values);
    }

    private static String zoneName(Card card, Game game) {
        try {
            Zone zone = game == null || game.getState() == null || card == null ? null : game.getState().getZone(card.getId());
            return zone == null ? "" : zone.name();
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static Set<String> parseActionTypes(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return Collections.emptySet();
        }
        Set<String> out = new HashSet<>();
        for (String item : raw.split(",")) {
            String v = item.trim();
            if ("*".equals(v)) {
                return Collections.emptySet();
            }
            if (!v.isEmpty()) {
                out.add(v);
            }
        }
        return out;
    }

    private static String safeGameKey(Game game) {
        try {
            return game.getId() == null ? "unknown" : game.getId().toString();
        } catch (Throwable ignored) {
            return "unknown";
        }
    }

    private static String sanitize(String value) {
        String v = value == null ? "" : value;
        v = v.replaceAll("[^A-Za-z0-9_.-]+", "_");
        if (v.length() > 96) {
            v = v.substring(0, 96);
        }
        return v.isEmpty() ? "unknown" : v;
    }

    public static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest((value == null ? "" : value).getBytes(StandardCharsets.UTF_8));
            StringBuilder out = new StringBuilder(bytes.length * 2);
            for (byte b : bytes) {
                out.append(String.format(Locale.US, "%02x", b & 0xff));
            }
            return out.toString();
        } catch (Throwable ignored) {
            return "";
        }
    }

    private static String csv(String value) {
        String s = value == null ? "" : value;
        return "\"" + s.replace("\"", "\"\"").replace("\r", " ").replace("\n", " ") + "\"";
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        List<String> out = new ArrayList<>(values.size());
        for (Integer value : values) {
            out.add(String.valueOf(value == null ? -1 : value));
        }
        return String.join("|", out);
    }

    private static List<String> strings(List<String> values) {
        if (values == null || values.isEmpty()) {
            return Collections.emptyList();
        }
        return new ArrayList<>(values);
    }

    private static List<Integer> ints(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return Collections.emptyList();
        }
        return new ArrayList<>(values);
    }

    private static String errorSummary(Throwable t) {
        if (t == null) {
            return "";
        }
        String message = t.getMessage();
        return t.getClass().getSimpleName() + (message == null || message.isEmpty() ? "" : ": " + message);
    }
}
