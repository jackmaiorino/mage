
package mage.abilities.effects.common.continuous;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import mage.abilities.Ability;
import mage.abilities.Mode;
import mage.abilities.effects.ContinuousEffectImpl;
import mage.constants.Duration;
import mage.constants.Layer;
import mage.constants.Outcome;
import mage.constants.SubLayer;
import mage.constants.SubType;
import mage.filter.FilterPermanent;
import mage.game.Game;
import mage.game.permanent.Permanent;

/**
 * @author magenoxx_at_googlemail.com
 */
public class ExchangeControlTargetEffect extends ContinuousEffectImpl {

    private String rule;
    private boolean withSource;
    private boolean withSecondTarget;
    private boolean destroyAttachedAuras;
    // LinkedHashMap: apply() iterates zoneChangeCounter.entrySet() to drive
    // sequential changeControllerId()/aura-destroy calls (deterministic
    // candidate order fix, see TargetCardInHand / CombatGroup).
    private Map<UUID, Integer> zoneChangeCounter = new LinkedHashMap<>();
    private Map<UUID, UUID> lockedControllers = new LinkedHashMap<>();  // Controllers for each permanent that is enforced by this effect

    public ExchangeControlTargetEffect(Duration duration, String rule) {
        this(duration, rule, false);
    }

    public ExchangeControlTargetEffect(Duration duration, String rule, boolean withSource) {
        this(duration, rule, withSource, false);
    }

    public ExchangeControlTargetEffect(Duration duration, String rule, boolean withSource, boolean withSecondTarget) {
        this(duration, rule, withSource, withSecondTarget, false);
    }

    public ExchangeControlTargetEffect(Duration duration, String rule, boolean withSource, boolean withSecondTarget, boolean destroyAttachedAuras) {
        super(duration, Layer.ControlChangingEffects_2, SubLayer.NA, Outcome.GainControl);
        this.withSource = withSource;
        this.withSecondTarget = withSecondTarget;
        this.destroyAttachedAuras = destroyAttachedAuras;
        this.rule = rule;
    }

    protected ExchangeControlTargetEffect(final ExchangeControlTargetEffect effect) {
        super(effect);
        this.rule = effect.rule;
        this.withSource = effect.withSource;
        this.withSecondTarget = effect.withSecondTarget;
        this.destroyAttachedAuras = effect.destroyAttachedAuras;
        this.lockedControllers = new LinkedHashMap<>(effect.lockedControllers);
        this.zoneChangeCounter = new LinkedHashMap<>(effect.zoneChangeCounter);
    }

    @Override
    public ExchangeControlTargetEffect copy() {
        return new ExchangeControlTargetEffect(this);
    }

    @Override
    public boolean isInactive(Ability source, Game game) {
        return isDiscarded();
    }

    @Override
    public void init(Ability source, Game game) {
        super.init(source, game);

        Permanent permanent1 = null;
        Permanent permanent2 = null;

        if (withSource) {
            permanent1 = game.getPermanent(getTargetPointer().getFirst(game, source));
            permanent2 = game.getPermanent(source.getSourceId());
        } else {
            for (UUID permanentId : getTargetPointer().getTargets(game, source)) {
                if (permanent1 == null) {
                    permanent1 = game.getPermanent(permanentId);
                } else if (permanent2 == null) {
                    permanent2 = game.getPermanent(permanentId);
                }
            }
            if (withSecondTarget) {
                UUID uuid = source.getTargets().get(1).getFirstTarget();
                permanent2 = game.getPermanent(uuid);
            } else if (permanent2 == null) {
                UUID uuid = source.getTargets().get(0).getFirstTarget();
                permanent2 = game.getPermanent(uuid);
            }
        }
        if (permanent1 != null && permanent2 != null) {
            // exchange works only for two different controllers
            if (permanent1.isControlledBy(permanent2.getControllerId())) {
                // discard effect if controller of both permanents is the same
                discard();
                return;
            }
            // Meant to be swapped since this enforced the
            this.lockedControllers.put(permanent1.getId(), permanent2.getControllerId());
            this.lockedControllers.put(permanent2.getId(), permanent1.getControllerId());

            this.zoneChangeCounter.put(permanent1.getId(), permanent1.getZoneChangeCounter(game));
            this.zoneChangeCounter.put(permanent2.getId(), permanent2.getZoneChangeCounter(game));
        } else {
            // discard if there are less than 2 permanents
            discard();
        }
    }

    @Override
    public boolean apply(Game game, Ability source) {
        Set<UUID> toDelete = new HashSet<>();
        for (Map.Entry<UUID, Integer> entry : zoneChangeCounter.entrySet()) {
            Permanent permanent = game.getPermanent(entry.getKey());
            if (permanent == null || permanent.getZoneChangeCounter(game) != entry.getValue()) {
                // Control effect cease if the same permanent is no longer on the battlefield
                toDelete.add(entry.getKey());
                continue;
            }
            permanent.changeControllerId(lockedControllers.get(permanent.getId()), game, source);
            permanent.getAbilities().setControllerId(lockedControllers.get(permanent.getId()));
            if (destroyAttachedAuras) {
                FilterPermanent filter = new FilterPermanent();
                filter.add(SubType.AURA.getPredicate());
                // LinkedHashSet: preserve getAttachments()' (already-ordered)
                // attach order for the sequential destroy() calls below.
                for (UUID attachmentId : new LinkedHashSet<>(permanent.getAttachments())) {
                    Permanent attachment = game.getPermanent(attachmentId);
                    if (filter.match(attachment, game)) {
                        attachment.destroy(source, game, false);
                    }
                }
            }
        }
        if (!toDelete.isEmpty()) {
            for (UUID uuid : toDelete) {
                zoneChangeCounter.remove(uuid);
                lockedControllers.remove(uuid);
            }
            if (zoneChangeCounter.isEmpty()) {
                discard();
                return false;
            }
        }
        return true;
    }

    @Override
    public String getText(Mode mode) {
        return this.rule;
    }
}
