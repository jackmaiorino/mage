package mage.watchers;

import mage.constants.WatcherScope;
import mage.game.Game;
import mage.game.events.GameEvent;
import mage.util.CardUtil;
import org.apache.log4j.Logger;

import java.io.Serializable;
import java.lang.reflect.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * watches for certain game events to occur and flags condition
 *
 * @author BetaSteward_at_googlemail.com
 */
public abstract class Watcher implements Serializable {

    private static final Logger logger = Logger.getLogger(Watcher.class);

    /** Per-class copy metadata cache: the constructor we reflectively invoke
     *  and the list of non-static fields whose values get copied across.
     *  Reflection lookup is 10-100x slower than direct call, and copy() is
     *  hit once per watcher per game clone — which during MCTS happens
     *  thousands of times per second. Cache once, pay the reflection cost
     *  exactly once per class per JVM. */
    private static final ConcurrentHashMap<Class<? extends Watcher>, CopyMeta> COPY_META_CACHE =
            new ConcurrentHashMap<>();

    private static final class CopyMeta {
        final Constructor<? extends Watcher> constructor;
        final Object[] argTemplate;
        final Field[] fields;

        CopyMeta(Constructor<? extends Watcher> constructor, Object[] argTemplate, Field[] fields) {
            this.constructor = constructor;
            this.argTemplate = argTemplate;
            this.fields = fields;
        }
    }

    @SuppressWarnings("unchecked")
    private static CopyMeta getCopyMeta(Class<? extends Watcher> cls) {
        CopyMeta cached = COPY_META_CACHE.get(cls);
        if (cached != null) return cached;

        Constructor<?>[] constructors = cls.getDeclaredConstructors();
        if (constructors.length > 1) {
            logger.error(cls.getSimpleName() + " has multiple constructors");
            return null;
        }
        Constructor<? extends Watcher> ctor = (Constructor<? extends Watcher>) constructors[0];
        ctor.setAccessible(true);

        Class<?>[] paramTypes = ctor.getParameterTypes();
        Object[] argTemplate = new Object[paramTypes.length];
        for (int i = 0; i < paramTypes.length; i++) {
            if (paramTypes[i].isPrimitive()
                    && paramTypes[i].getSimpleName().equalsIgnoreCase("boolean")) {
                argTemplate[i] = false;
            } else {
                argTemplate[i] = null;
            }
        }

        List<Field> collected = new ArrayList<>();
        for (Field f : cls.getDeclaredFields()) {
            if (!Modifier.isStatic(f.getModifiers())) {
                f.setAccessible(true);
                collected.add(f);
            }
        }
        Class<?> superCls = cls.getSuperclass();
        if (superCls != null) {
            for (Field f : superCls.getDeclaredFields()) {
                if (!Modifier.isStatic(f.getModifiers())) {
                    f.setAccessible(true);
                    collected.add(f);
                }
            }
        }

        CopyMeta meta = new CopyMeta(ctor, argTemplate, collected.toArray(new Field[0]));
        COPY_META_CACHE.putIfAbsent(cls, meta);
        return meta;
    }

    protected UUID controllerId;
    protected UUID sourceId;
    protected boolean condition;
    protected final WatcherScope scope;

    public Watcher(WatcherScope scope) {
        this.scope = scope;
    }

    protected Watcher(final Watcher watcher) {
        this.condition = watcher.condition;
        this.controllerId = watcher.controllerId;
        this.sourceId = watcher.sourceId;
        this.scope = watcher.scope;
    }

    public UUID getControllerId() {
        return controllerId;
    }

    public void setControllerId(UUID controllerId) {
        this.controllerId = controllerId;
    }

    public UUID getSourceId() {
        return sourceId;
    }

    public void setSourceId(UUID sourceId) {
        this.sourceId = sourceId;
    }

    public String getKey() {
        switch (scope) {
            case GAME:
                return getBasicKey();
            case PLAYER:
                return controllerId + getBasicKey();
            case CARD:
                return sourceId + getBasicKey();
            default:
                throw new IllegalArgumentException("Unknown watcher scope: " + this.getClass().getSimpleName() + " - " + scope);
        }
    }

    public boolean conditionMet() {
        return condition;
    }

    public void reset() {
        condition = false;
    }

    protected String getBasicKey() {
        return getClass().getSimpleName();
    }

    public abstract void watch(GameEvent event, Game game);

    @SuppressWarnings("unchecked")
    public <T extends Watcher> T copy() {
        CopyMeta meta = getCopyMeta((Class<? extends Watcher>) this.getClass());
        if (meta == null) return null;
        try {
            T watcher = (T) meta.constructor.newInstance(meta.argTemplate);
            for (Field field : meta.fields) {
                field.set(watcher, CardUtil.deepCopyObject(field.get(this)));
            }
            return watcher;
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            logger.error("Can't copy watcher: " + e.getMessage(), e);
        }
        return null;
    }

    public WatcherScope getScope() {
        return scope;
    }
}
