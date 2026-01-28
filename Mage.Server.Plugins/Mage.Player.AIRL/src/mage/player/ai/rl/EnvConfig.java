package mage.player.ai.rl;

/**
 * Minimal environment-variable parsing helpers.
 *
 * Keep behavior consistent across training components and avoid duplicated
 * ad-hoc parsing.
 */
public final class EnvConfig {

    private EnvConfig() {
    }

    public static String str(String key, String def) {
        String v = System.getenv(key);
        if (v == null) {
            return def;
        }
        v = v.trim();
        return v.isEmpty() ? def : v;
    }

    public static int i32(String key, int def) {
        String v = str(key, null);
        if (v == null) {
            return def;
        }
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException ignored) {
            return def;
        }
    }

    public static long i64(String key, long def) {
        String v = str(key, null);
        if (v == null) {
            return def;
        }
        try {
            return Long.parseLong(v);
        } catch (NumberFormatException ignored) {
            return def;
        }
    }

    public static double f64(String key, double def) {
        String v = str(key, null);
        if (v == null) {
            return def;
        }
        try {
            return Double.parseDouble(v);
        } catch (NumberFormatException ignored) {
            return def;
        }
    }

    public static boolean bool(String key, boolean def) {
        String v = str(key, null);
        if (v == null) {
            return def;
        }
        if ("1".equals(v)) {
            return true;
        }
        if ("0".equals(v)) {
            return false;
        }
        if ("true".equalsIgnoreCase(v)) {
            return true;
        }
        if ("false".equalsIgnoreCase(v)) {
            return false;
        }
        return def;
    }
}

