package mage.player.ai.rl;

import mage.MageObject;
import mage.abilities.ActivatedAbility;
import mage.abilities.Ability;
import mage.abilities.costs.Cost;
import mage.abilities.costs.common.TapSourceCost;
import mage.abilities.costs.common.TapTargetCost;
import mage.abilities.mana.ManaAbility;
import mage.cards.Card;
import mage.constants.SubType;
import mage.constants.RangeOfInfluence;
import mage.constants.Zone;
import mage.game.Game;
import mage.game.combat.CombatGroup;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.player.ai.ComputerPlayer7;
import mage.players.Player;
import mage.target.Target;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Validation-only CP7 wrapper for replay parity diagnostics.
 */
public final class ReplayOpponentDecisionPlayer extends ComputerPlayer7 {

    private static final boolean ENABLED = EnvConfig.bool(
            "EVAL_OPPONENT_DECISION_JSON",
            EnvConfig.bool("EVAL_REPLAY_METADATA", false));
    private static final int MAX_SOURCE_TURN = EnvConfig.i32(
            "EVAL_OPPONENT_DECISION_MAX_SOURCE_TURN",
            EnvConfig.i32("EVAL_REPLAY_OPPONENT_DECISION_MAX_SOURCE_TURN", 99));
    private static final Path FILE_SINK = resolveFileSink();

    private final int scenario;
    private final long seed;
    private final GameLogger gameLogger;
    private int decisionIndex = 0;

    public static ComputerPlayer7 create(String name, RangeOfInfluence range, int skill,
                                         int scenario, long seed, GameLogger gameLogger) {
        if (!ENABLED) {
            return new ComputerPlayer7(name, range, skill);
        }
        return new ReplayOpponentDecisionPlayer(name, range, skill, scenario, seed, gameLogger);
    }

    private ReplayOpponentDecisionPlayer(String name, RangeOfInfluence range, int skill,
                                         int scenario, long seed, GameLogger gameLogger) {
        super(name, range, skill);
        this.scenario = scenario;
        this.seed = seed;
        this.gameLogger = gameLogger;
    }

    private ReplayOpponentDecisionPlayer(ReplayOpponentDecisionPlayer player) {
        super(player);
        this.scenario = player.scenario;
        this.seed = player.seed;
        this.gameLogger = player.gameLogger;
        this.decisionIndex = player.decisionIndex;
    }

    @Override
    public ReplayOpponentDecisionPlayer copy() {
        return new ReplayOpponentDecisionPlayer(this);
    }

    @Override
    protected void act(Game game) {
        List<Ability> planned = new ArrayList<>(actions);
        List<Ability> legal = legalPlayableActions(game);
        if (planned.isEmpty()) {
            logDecision(game, "PRIORITY", "Pass", 0, Collections.emptyList(),
                    null, Collections.emptyList(), legal, Collections.emptyList(), Collections.emptyList());
        } else {
            List<String> actionTexts = planned.stream()
                    .map(ability -> safeAbilityText(game, ability))
                    .collect(Collectors.toList());
            for (int i = 0; i < actionTexts.size(); i++) {
                logDecision(game, "PRIORITY", actionTexts.get(i), actionTexts.size(), actionTexts,
                        planned.get(i), planned, legal, Collections.emptyList(), Collections.emptyList());
            }
        }
        super.act(game);
    }

    @Override
    public void selectAttackers(Game game, UUID attackingPlayerId) {
        super.selectAttackers(game, attackingPlayerId);
        List<String> attackers = selectedCombatNames(game, true);
        List<UUID> attackerIds = selectedCombatIds(game, true);
        logDecision(game, "DECLARE_ATTACKERS",
                attackers.isEmpty() ? "Declare attackers: NONE" : "Declare attackers: " + String.join("|", attackers),
                attackers.size(), attackers,
                null, Collections.emptyList(), Collections.emptyList(),
                attackerIds, controlledCreatureIds(game, attackingPlayerId));
    }

    @Override
    public void selectBlockers(Ability source, Game game, UUID defendingPlayerId) {
        super.selectBlockers(source, game, defendingPlayerId);
        List<String> blockers = selectedCombatNames(game, false);
        List<UUID> blockerIds = selectedCombatIds(game, false);
        logDecision(game, "DECLARE_BLOCKERS",
                blockers.isEmpty() ? "Declare blockers: NONE" : "Declare blockers: " + String.join("|", blockers),
                blockers.size(), blockers,
                null, Collections.emptyList(), Collections.emptyList(),
                blockerIds, controlledCreatureIds(game, defendingPlayerId));
    }

    private void logDecision(Game game, String event, String chosenText, int candidateCount, List<String> candidates,
                             Ability chosenAbility, List<Ability> candidateAbilities, List<Ability> legalAbilities,
                             List<UUID> selectedObjectIds, List<UUID> candidateObjectIds) {
        if (!shouldLog(game)) {
            return;
        }
        decisionIndex++;
        List<Card> actorHand = actorHandCards(game);
        List<Card> actorLibrary = actorLibraryCards(game);
        List<String> candidateIds = idsFromAbilities(candidateAbilities);
        if (candidateIds.isEmpty()) {
            candidateIds = uuidStrings(candidateObjectIds);
        }
        List<String> chosenIds = uuidStrings(selectedObjectIds);
        if (chosenIds.isEmpty()) {
            chosenIds = chosenAbility == null ? Collections.emptyList() : singletonUuid(chosenAbility.getSourceId());
        }
        List<String> sourceIds = sourceObjectIds(chosenAbility);
        List<String> targetIds = targetObjectIds(chosenAbility);
        List<String> tapSourceIds = tapSourceObjectIds(chosenAbility);
        List<String> manaSourceIds = manaPaymentSourceObjectIds(chosenAbility, tapSourceIds);
        Player self = game == null ? null : game.getPlayer(getId());
        Player opponent = firstOpponent(game);
        List<Permanent> selfBattlefield = battlefieldPermanents(game, self == null ? null : self.getId());
        List<Permanent> opponentBattlefield = battlefieldPermanents(game, opponent == null ? null : opponent.getId());
        List<String> visibleSelfIds = permanentIds(selfBattlefield);
        List<String> visibleOpponentIds = permanentIds(opponentBattlefield);
        List<String> visibleAllIds = new ArrayList<>(visibleSelfIds);
        visibleAllIds.addAll(visibleOpponentIds);
        List<String> attachmentContext = attachmentContextRows(game);
        List<String> equipmentContext = equipmentContextRows(game);
        List<String> legalTexts = abilityTexts(game, legalAbilities);
        List<String> legalIds = idsFromAbilities(legalAbilities);
        String json = "{"
                + "\"scenario\":" + scenario
                + ",\"seed\":" + seed
                + ",\"decision_index\":" + decisionIndex
                + ",\"actor\":\"" + json(getName()) + "\""
                + ",\"event\":\"" + json(event) + "\""
                + ",\"turn\":" + safeTurn(game)
                + ",\"source_turn\":" + sourceTurn(game)
                + ",\"phase\":\"" + json(safePhase(game)) + "\""
                + ",\"chosen_action_text\":\"" + json(chosenText) + "\""
                + ",\"candidate_count\":" + candidateCount
                + ",\"candidate_texts\":" + jsonArray(candidates)
                + ",\"candidate_object_ids\":" + jsonArray(candidateIds)
                + ",\"chosen_object_ids\":" + jsonArray(chosenIds)
                + ",\"selected_object_ids\":" + jsonArray(chosenIds)
                + ",\"source_object_id\":\"" + json(sourceIds.isEmpty() ? "" : sourceIds.get(0)) + "\""
                + ",\"source_object_ids\":" + jsonArray(sourceIds)
                + ",\"target_object_id\":\"" + json(targetIds.isEmpty() ? "" : targetIds.get(0)) + "\""
                + ",\"target_object_ids\":" + jsonArray(targetIds)
                + ",\"tap_source_object_ids\":" + jsonArray(tapSourceIds)
                + ",\"mana_payment_source_object_ids\":" + jsonArray(manaSourceIds)
                + ",\"planned_candidate_texts\":" + jsonArray(candidates)
                + ",\"planned_candidate_object_ids\":" + jsonArray(candidateIds)
                + ",\"legal_candidate_texts\":" + jsonArray(legalTexts)
                + ",\"legal_candidate_object_ids\":" + jsonArray(legalIds)
                + ",\"visible_self_battlefield_names\":" + jsonArray(permanentNames(selfBattlefield))
                + ",\"visible_self_battlefield_object_ids\":" + jsonArray(visibleSelfIds)
                + ",\"visible_opponent_battlefield_names\":" + jsonArray(permanentNames(opponentBattlefield))
                + ",\"visible_opponent_battlefield_object_ids\":" + jsonArray(visibleOpponentIds)
                + ",\"visible_battlefield_object_ids\":" + jsonArray(visibleAllIds)
                + ",\"attachment_object_ids\":" + jsonArray(attachmentObjectIds(game))
                + ",\"attached_to_object_ids\":" + jsonArray(attachedToObjectIds(game))
                + ",\"attachment_context\":" + jsonArray(attachmentContext)
                + ",\"equipped_to_object_ids\":" + jsonArray(equippedToObjectIds(game))
                + ",\"equipment_context\":" + jsonArray(equipmentContext)
                + ",\"combat_attacker_object_ids\":" + jsonArray(combatIds(game, true))
                + ",\"combat_blocker_object_ids\":" + jsonArray(combatIds(game, false))
                + ",\"combat_trace\":\"" + json(combatTrace(game)) + "\""
                + ",\"visible_state\":\"" + json(visibleState(game)) + "\""
                + ",\"hidden_state_provenance\":\"live_player_zone_snapshot\""
                + ",\"actor_hand_size\":" + actorHand.size()
                + ",\"actor_hand\":" + jsonArray(cardNames(actorHand, 0))
                + ",\"actor_hand_object_ids\":" + jsonArray(cardIds(actorHand, 0))
                + ",\"actor_library_size\":" + actorLibrary.size()
                + ",\"actor_library\":" + jsonArray(cardNames(actorLibrary, 0))
                + ",\"actor_library_top\":" + jsonArray(cardNames(actorLibrary, 12))
                + ",\"actor_library_object_ids\":" + jsonArray(cardIds(actorLibrary, 0))
                + ",\"actor_library_top_object_ids\":" + jsonArray(cardIds(actorLibrary, 12))
                + "}";
        String line = "REPLAY_OPPONENT_DECISION_JSON: " + json;
        if (gameLogger != null && gameLogger.isEnabled()) {
            gameLogger.log(line);
        }
        appendFileSink(line);
    }

    private List<Ability> legalPlayableActions(Game game) {
        if (game == null) {
            return Collections.emptyList();
        }
        try {
            return new ArrayList<Ability>(getPlayableFast(game, true, Zone.ALL, false));
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
    }

    private boolean shouldLog(Game game) {
        if (game == null || game.isSimulation() || !ENABLED) {
            return false;
        }
        if (MAX_SOURCE_TURN > 0 && sourceTurn(game) > MAX_SOURCE_TURN) {
            return false;
        }
        return gameLogger != null && gameLogger.isEnabled() || FILE_SINK != null;
    }

    private List<Card> actorHandCards(Game game) {
        try {
            Player self = game == null ? null : game.getPlayer(getId());
            return self == null || self.getHand() == null
                    ? Collections.emptyList()
                    : new ArrayList<>(self.getHand().getCards(game));
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
    }

    private List<Card> actorLibraryCards(Game game) {
        try {
            Player self = game == null ? null : game.getPlayer(getId());
            return self == null || self.getLibrary() == null
                    ? Collections.emptyList()
                    : new ArrayList<>(self.getLibrary().getCards(game));
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
    }

    private String safeAbilityText(Game game, Ability ability) {
        try {
            return getAbilityAndSourceInfo(game, ability, true);
        } catch (Exception e) {
            return String.valueOf(ability);
        }
    }

    private List<String> abilityTexts(Game game, List<Ability> abilities) {
        if (abilities == null || abilities.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Ability ability : abilities) {
            out.add(safeAbilityText(game, ability));
        }
        return out;
    }

    private List<String> idsFromAbilities(List<Ability> abilities) {
        if (abilities == null || abilities.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Ability ability : abilities) {
            if (ability == null || ability.getSourceId() == null) {
                out.add("");
            } else {
                out.add(ability.getSourceId().toString());
            }
        }
        return out;
    }

    private List<String> sourceObjectIds(Ability ability) {
        return ability == null ? Collections.emptyList() : singletonUuid(ability.getSourceId());
    }

    private List<String> targetObjectIds(Ability ability) {
        if (ability == null || ability.getAllSelectedTargets() == null) {
            return Collections.emptyList();
        }
        Set<String> out = new LinkedHashSet<>();
        try {
            for (Target target : ability.getAllSelectedTargets()) {
                if (target == null || target.getTargets() == null) {
                    continue;
                }
                for (UUID id : target.getTargets()) {
                    if (id != null) {
                        out.add(id.toString());
                    }
                }
            }
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
        return new ArrayList<>(out);
    }

    private List<String> tapSourceObjectIds(Ability ability) {
        if (ability == null) {
            return Collections.emptyList();
        }
        Set<String> out = new LinkedHashSet<>();
        try {
            if (ability.getCosts() != null) {
                for (Cost cost : ability.getCosts()) {
                    collectTapCostIds(cost, ability, out);
                }
            }
            if (ability.getManaCostsToPay() != null) {
                for (Cost cost : ability.getManaCostsToPay()) {
                    collectTapCostIds(cost, ability, out);
                }
            }
        } catch (Exception ignored) {
            return Collections.emptyList();
        }
        return new ArrayList<>(out);
    }

    private void collectTapCostIds(Cost cost, Ability ability, Set<String> out) {
        if (cost == null || out == null) {
            return;
        }
        if (cost instanceof TapSourceCost && ability != null && ability.getSourceId() != null) {
            out.add(ability.getSourceId().toString());
        }
        if (cost instanceof TapTargetCost) {
            collectTargetIds(((TapTargetCost) cost).getTarget(), out);
        } else {
            try {
                if (cost.getTargets() != null) {
                    for (Target target : cost.getTargets()) {
                        collectTargetIds(target, out);
                    }
                }
            } catch (Exception ignored) {
            }
        }
    }

    private void collectTargetIds(Target target, Set<String> out) {
        if (target == null || target.getTargets() == null || out == null) {
            return;
        }
        for (UUID id : target.getTargets()) {
            if (id != null) {
                out.add(id.toString());
            }
        }
    }

    private List<String> manaPaymentSourceObjectIds(Ability ability, List<String> tapSourceIds) {
        if (!(ability instanceof ManaAbility)) {
            return Collections.emptyList();
        }
        Set<String> out = new LinkedHashSet<>();
        if (ability.getSourceId() != null) {
            out.add(ability.getSourceId().toString());
        }
        if (tapSourceIds != null) {
            out.addAll(tapSourceIds);
        }
        return new ArrayList<>(out);
    }

    private List<String> selectedCombatNames(Game game, boolean attackers) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getCombat() == null) {
            return new ArrayList<>();
        }
        for (CombatGroup group : game.getCombat().getGroups()) {
            List<UUID> ids = attackers ? group.getAttackers() : group.getBlockers();
            for (UUID id : ids) {
                Permanent permanent = game.getPermanent(id);
                if (permanent != null && getId().equals(permanent.getControllerId())) {
                    out.add(permanent.getName());
                }
            }
        }
        return new ArrayList<>(out);
    }

    private List<UUID> selectedCombatIds(Game game, boolean attackers) {
        Set<UUID> out = new LinkedHashSet<>();
        if (game == null || game.getCombat() == null) {
            return new ArrayList<>();
        }
        for (CombatGroup group : game.getCombat().getGroups()) {
            List<UUID> ids = attackers ? group.getAttackers() : group.getBlockers();
            for (UUID id : ids) {
                Permanent permanent = game.getPermanent(id);
                if (permanent != null && getId().equals(permanent.getControllerId())) {
                    out.add(permanent.getId());
                }
            }
        }
        return new ArrayList<>(out);
    }

    private List<UUID> controlledCreatureIds(Game game, UUID controllerId) {
        if (game == null || game.getBattlefield() == null || controllerId == null) {
            return Collections.emptyList();
        }
        List<UUID> out = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllActivePermanents(controllerId)) {
            if (permanent != null && permanent.isCreature(game)) {
                out.add(permanent.getId());
            }
        }
        return out;
    }

    private String combatTrace(Game game) {
        if (game == null || game.getCombat() == null || game.getBattlefield() == null) {
            return "";
        }
        try {
            return "turn=" + safeTurn(game)
                    + ";sourceTurn=" + sourceTurn(game)
                    + ";phase=" + safePhase(game)
                    + ";active=" + playerName(game, game.getActivePlayerId())
                    + ";priority=" + playerName(game, game.getPriorityPlayerId())
                    + ";attackers=" + combatSetText(game, true)
                    + ";blockers=" + combatSetText(game, false)
                    + ";groups=" + combatGroupsText(game)
                    + ";blockerLegality=" + blockerLegalityText(game);
        } catch (Exception e) {
            return "combat_trace_error=" + e.getClass().getSimpleName() + ":" + String.valueOf(e.getMessage());
        }
    }

    private String combatSetText(Game game, boolean attackers) {
        if (game == null || game.getCombat() == null) {
            return "";
        }
        List<String> names = new ArrayList<>();
        Set<UUID> ids = attackers ? game.getCombat().getAttackers() : game.getCombat().getBlockers();
        for (UUID id : ids) {
            names.add(permanentDetail(game, id));
        }
        return String.join("|", names);
    }

    private String combatGroupsText(Game game) {
        if (game == null || game.getCombat() == null) {
            return "";
        }
        List<String> groups = new ArrayList<>();
        for (CombatGroup group : game.getCombat().getGroups()) {
            List<String> attackers = new ArrayList<>();
            for (UUID attackerId : group.getAttackers()) {
                attackers.add(permanentDetail(game, attackerId));
            }
            List<String> blockers = new ArrayList<>();
            for (UUID blockerId : group.getBlockers()) {
                blockers.add(permanentDetail(game, blockerId));
            }
            groups.add("att=" + String.join("+", attackers)
                    + ">def=" + playerName(game, group.getDefenderId())
                    + "(" + String.valueOf(group.getDefenderId()) + ")"
                    + ">defendingPlayer=" + playerName(game, group.getDefendingPlayerId())
                    + "(" + String.valueOf(group.getDefendingPlayerId()) + ")"
                    + ">blk=" + String.join("+", blockers));
        }
        return String.join("|", groups);
    }

    private String blockerLegalityText(Game game) {
        if (game == null || game.getCombat() == null || game.getBattlefield() == null) {
            return "";
        }
        List<Permanent> candidates = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllActivePermanents(getId())) {
            if (permanent != null && permanent.isCreature(game)) {
                candidates.add(permanent);
            }
        }
        List<String> rows = new ArrayList<>();
        int groupIndex = 0;
        for (CombatGroup group : game.getCombat().getGroups()) {
            for (UUID attackerId : group.getAttackers()) {
                Permanent attacker = game.getPermanent(attackerId);
                for (Permanent blocker : candidates) {
                    boolean permanentCanBlock = false;
                    boolean groupCanBlock = false;
                    try {
                        permanentCanBlock = blocker.canBlock(attackerId, game);
                    } catch (Exception ignored) {
                    }
                    try {
                        groupCanBlock = group.canBlock(blocker, game);
                    } catch (Exception ignored) {
                    }
                    rows.add("group=" + groupIndex
                            + ">defender=" + playerName(game, group.getDefenderId())
                            + "(" + String.valueOf(group.getDefenderId()) + ")"
                            + ">defendingPlayer=" + playerName(game, group.getDefendingPlayerId())
                            + "(" + String.valueOf(group.getDefendingPlayerId()) + ")"
                            + ">attacker=" + permanentDetail(attacker, game)
                            + ">blocker=" + permanentDetail(blocker, game)
                            + ">Permanent.canBlock=" + permanentCanBlock
                            + ">CombatGroup.canBlock=" + groupCanBlock
                            + ">reason=" + blockerLegalityReason(group, blocker, game,
                            permanentCanBlock, groupCanBlock));
                }
            }
            groupIndex++;
        }
        return rows.isEmpty() ? "no_attackers_or_no_candidate_blockers" : String.join("|", rows);
    }

    private String blockerLegalityReason(
            CombatGroup group,
            Permanent blocker,
            Game game,
            boolean permanentCanBlock,
            boolean groupCanBlock
    ) {
        if (blocker == null) {
            return "missing_blocker";
        }
        if (group == null) {
            return "missing_combat_group";
        }
        if (group.getDefendingPlayerId() == null
                || !group.getDefendingPlayerId().equals(blocker.getControllerId())) {
            return "wrong_controller";
        }
        try {
            if (!blocker.isCreature(game)) {
                return "not_creature";
            }
        } catch (Exception ignored) {
            return "creature_check_error";
        }
        try {
            if (blocker.isTapped()) {
                return "tapped";
            }
        } catch (Exception ignored) {
        }
        try {
            if (blocker.isAttacking()) {
                return "attacking";
            }
        } catch (Exception ignored) {
        }
        try {
            if (blocker.getBlocking() > 0) {
                return "already_blocking";
            }
        } catch (Exception ignored) {
        }
        if (permanentCanBlock && groupCanBlock) {
            return "legal";
        }
        if (!permanentCanBlock) {
            return "permanent_canBlock_false";
        }
        if (!groupCanBlock) {
            return "combat_group_canBlock_false";
        }
        return "unknown";
    }

    private String permanentDetail(Game game, UUID permanentId) {
        Permanent permanent = game == null || permanentId == null ? null : game.getPermanent(permanentId);
        return permanent == null ? String.valueOf(permanentId) : permanentDetail(permanent, game);
    }

    private String permanentDetail(Permanent permanent, Game game) {
        if (permanent == null) {
            return "";
        }
        List<String> flags = new ArrayList<>();
        if (permanent.isTapped()) {
            flags.add("tapped");
        }
        if (permanent.hasSummoningSickness()) {
            flags.add("summoning_sick");
        }
        if (permanent.isAttacking()) {
            flags.add("attacking");
        }
        if (permanent.getBlocking() > 0) {
            flags.add("blocking");
        }
        return permanent.getName()
                + "{id=" + permanent.getId()
                + ",controller=" + playerName(game, permanent.getControllerId())
                + "(" + permanent.getControllerId() + ")"
                + ",status=" + (flags.isEmpty() ? "ready" : String.join("+", flags))
                + "}";
    }

    private String visibleState(Game game) {
        if (game == null) {
            return "";
        }
        Player self = game.getPlayer(getId());
        Player opponent = firstOpponent(game);
        List<String> parts = new ArrayList<>();
        parts.add("active=" + playerName(game, game.getActivePlayerId()));
        parts.add("priority=" + playerName(game, game.getPriorityPlayerId()));
        parts.add("stack=" + stackText(game));
        if (self != null) {
            parts.add("selfLife=" + self.getLife());
            parts.add("selfHand=" + self.getHand().size());
            parts.add("selfBattlefield=" + battlefieldText(game, self.getId()));
            parts.add("selfGraveyard=" + zoneNames(self.getGraveyard().getCards(game)));
            parts.add("selfExile=" + zoneNames(game.getExile().getCardsOwned(game, self.getId())));
        }
        if (opponent != null) {
            parts.add("opp=" + opponent.getName());
            parts.add("oppLife=" + opponent.getLife());
            parts.add("oppHand=" + opponent.getHand().size());
            parts.add("oppBattlefield=" + battlefieldText(game, opponent.getId()));
            parts.add("oppGraveyard=" + zoneNames(opponent.getGraveyard().getCards(game)));
        }
        return String.join(";", parts);
    }

    private Player firstOpponent(Game game) {
        for (UUID id : game.getOpponents(getId())) {
            Player player = game.getPlayer(id);
            if (player != null) {
                return player;
            }
        }
        return null;
    }

    private String battlefieldText(Game game, UUID controllerId) {
        return game.getBattlefield().getAllPermanents().stream()
                .filter(permanent -> controllerId.equals(permanent.getControllerId()))
                .map(permanent -> permanent.getName()
                        + (permanent.isTapped() ? "[tapped]" : "")
                        + (permanent.isAttacking() ? "[attacking]" : "")
                        + (permanent.getBlocking() > 0 ? "[blocking]" : ""))
                .collect(Collectors.joining("|"));
    }

    private List<Permanent> battlefieldPermanents(Game game, UUID controllerId) {
        if (game == null || game.getBattlefield() == null || controllerId == null) {
            return Collections.emptyList();
        }
        List<Permanent> out = new ArrayList<>();
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent != null && controllerId.equals(permanent.getControllerId())) {
                out.add(permanent);
            }
        }
        return out;
    }

    private List<String> permanentNames(List<Permanent> permanents) {
        if (permanents == null || permanents.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Permanent permanent : permanents) {
            if (permanent != null) {
                out.add(permanent.getName());
            }
        }
        return out;
    }

    private List<String> permanentIds(List<Permanent> permanents) {
        if (permanents == null || permanents.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Permanent permanent : permanents) {
            if (permanent != null && permanent.getId() != null) {
                out.add(permanent.getId().toString());
            }
        }
        return out;
    }

    private List<String> attachmentObjectIds(Game game) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getBattlefield() == null) {
            return new ArrayList<>();
        }
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent == null || permanent.getAttachments() == null) {
                continue;
            }
            for (UUID id : permanent.getAttachments()) {
                if (id != null) {
                    out.add(id.toString());
                }
            }
        }
        return new ArrayList<>(out);
    }

    private List<String> attachedToObjectIds(Game game) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getBattlefield() == null) {
            return new ArrayList<>();
        }
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent != null && permanent.getAttachedTo() != null) {
                out.add(permanent.getAttachedTo().toString());
            }
        }
        return new ArrayList<>(out);
    }

    private List<String> equippedToObjectIds(Game game) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getBattlefield() == null) {
            return new ArrayList<>();
        }
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent == null || permanent.getAttachedTo() == null) {
                continue;
            }
            try {
                if (permanent.hasSubtype(SubType.EQUIPMENT, game)) {
                    out.add(permanent.getAttachedTo().toString());
                }
            } catch (Exception ignored) {
            }
        }
        return new ArrayList<>(out);
    }

    private List<String> attachmentContextRows(Game game) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getBattlefield() == null) {
            return new ArrayList<>();
        }
        for (Permanent host : game.getBattlefield().getAllPermanents()) {
            if (host == null) {
                continue;
            }
            if (host.getAttachments() != null) {
                for (UUID attachmentId : host.getAttachments()) {
                    if (attachmentId != null) {
                        out.add(attachmentId + "->" + host.getId());
                    }
                }
            }
            if (host.getAttachedTo() != null) {
                out.add(host.getId() + "->" + host.getAttachedTo());
            }
        }
        return new ArrayList<>(out);
    }

    private List<String> equipmentContextRows(Game game) {
        Set<String> out = new LinkedHashSet<>();
        if (game == null || game.getBattlefield() == null) {
            return new ArrayList<>();
        }
        for (Permanent permanent : game.getBattlefield().getAllPermanents()) {
            if (permanent == null || permanent.getAttachedTo() == null) {
                continue;
            }
            try {
                if (permanent.hasSubtype(SubType.EQUIPMENT, game)) {
                    out.add(permanent.getId() + "->" + permanent.getAttachedTo());
                }
            } catch (Exception ignored) {
            }
        }
        return new ArrayList<>(out);
    }

    private List<String> combatIds(Game game, boolean attackers) {
        if (game == null || game.getCombat() == null) {
            return Collections.emptyList();
        }
        Set<UUID> ids = attackers ? game.getCombat().getAttackers() : game.getCombat().getBlockers();
        return uuidStrings(ids == null ? Collections.emptyList() : ids);
    }

    private String zoneNames(Collection<? extends MageObject> objects) {
        return objects.stream().map(MageObject::getName).collect(Collectors.joining("|"));
    }

    private String stackText(Game game) {
        if (game.getStack().isEmpty()) {
            return "";
        }
        StackObject top = game.getStack().getFirstOrNull();
        return top == null ? "" : top.toString();
    }

    private String playerName(Game game, UUID playerId) {
        Player player = game == null ? null : game.getPlayer(playerId);
        return player == null ? "" : player.getName();
    }

    private int safeTurn(Game game) {
        return game == null ? -1 : game.getTurnNum();
    }

    private int sourceTurn(Game game) {
        int raw = safeTurn(game);
        return raw < 0 ? -1 : (Math.max(1, raw) + 1) / 2;
    }

    private String safePhase(Game game) {
        return game == null || game.getTurnStepType() == null ? "" : String.valueOf(game.getTurnStepType());
    }

    private static String jsonArray(List<String> values) {
        if (values == null || values.isEmpty()) {
            return "[]";
        }
        return values.stream()
                .map(value -> "\"" + json(value) + "\"")
                .collect(Collectors.joining(",", "[", "]"));
    }

    private static List<String> cardNames(List<Card> cards, int limit) {
        if (cards == null || cards.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Card card : cards) {
            if (card != null && card.getName() != null && !card.getName().isEmpty()) {
                out.add(card.getName());
            }
            if (limit > 0 && out.size() >= limit) {
                break;
            }
        }
        return out;
    }

    private static List<String> cardIds(List<Card> cards, int limit) {
        if (cards == null || cards.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (Card card : cards) {
            if (card != null && card.getId() != null) {
                out.add(card.getId().toString());
            }
            if (limit > 0 && out.size() >= limit) {
                break;
            }
        }
        return out;
    }

    private static List<String> uuidStrings(Collection<UUID> ids) {
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (UUID id : ids) {
            if (id != null) {
                out.add(id.toString());
            }
        }
        return out;
    }

    private static List<String> singletonUuid(UUID id) {
        return id == null ? Collections.emptyList() : Collections.singletonList(id.toString());
    }

    private static String json(String text) {
        if (text == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder(text.length() + 8);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '\\':
                    sb.append("\\\\");
                    break;
                case '"':
                    sb.append("\\\"");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                    break;
            }
        }
        return sb.toString();
    }

    private static Path resolveFileSink() {
        String raw = EnvConfig.str("EVAL_OPPONENT_DECISION_FILE", "").trim();
        if (raw.isEmpty()) {
            return null;
        }
        return Paths.get(raw).toAbsolutePath().normalize();
    }

    private static synchronized void appendFileSink(String line) {
        if (FILE_SINK == null) {
            return;
        }
        try {
            Path parent = FILE_SINK.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(FILE_SINK, (line + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignored) {
        }
    }
}
