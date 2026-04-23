package mage.player.ai.rl;

import java.lang.reflect.Method;
import java.util.Map;

/**
 * Smoke test for multi-profile shared-GPU routing. Run with:
 *   $env:PY_SERVICE_MODE='shared_gpu'; $env:MODEL_PROFILE='Pauper-Spy-Combo-Value';
 *   $env:TRAIN_PROFILES_LIST='Pauper-Spy-Combo-Value,Pauper-Wildfire-Value,Pauper-Rally-Anchor-Value,Pauper-Affinity-Anchor-Value';
 *   $env:GPU_SERVICE_ENDPOINT='localhost:1';
 *   mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java \
 *     -Dexec.mainClass=mage.player.ai.rl.SharedGpuProfileRoutingTest
 */
public final class SharedGpuProfileRoutingTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        requireEnv("PY_SERVICE_MODE", "shared_gpu");
        requireEnv("MODEL_PROFILE", "Pauper-Spy-Combo-Value");
        requireEnv("TRAIN_PROFILES_LIST",
                "Pauper-Spy-Combo-Value,Pauper-Wildfire-Value,Pauper-Rally-Anchor-Value,Pauper-Affinity-Anchor-Value");
        requireEnv("GPU_SERVICE_ENDPOINT", "localhost:1");

        String artifactsRoot = "build/test-profile-routing";
        registerProfile("Pauper-Spy-Combo-Value", artifactsRoot);
        registerProfile("Pauper-Wildfire-Value", artifactsRoot);
        registerProfile("Pauper-Rally-Anchor-Value", artifactsRoot);
        registerProfile("Pauper-Affinity-Anchor-Value", artifactsRoot);

        SharedGpuPythonModel model = SharedGpuPythonModel.getInstance();
        Method route = SharedGpuPythonModel.class.getDeclaredMethod("routeProfileForPolicy", String.class);
        route.setAccessible(true);
        Method headers = SharedGpuPythonModel.class.getDeclaredMethod("buildRegisterHeaders", String.class);
        headers.setAccessible(true);

        testRouteCurrentProfile(model, route);
        testRouteExplicitProfile(model, route);
        testRegisterHeadersUseRequestedProfile(model, headers);

        System.out.println();
        System.out.println("Total: " + (passed + failed) + "  Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) System.exit(1);
    }

    private static void testRouteCurrentProfile(SharedGpuPythonModel model, Method route) throws Exception {
        ProfileContext.setCurrent(ProfileContext.byName("Pauper-Wildfire-Value"));
        String routed = (String) route.invoke(model, "train");
        if (!"Pauper-Wildfire-Value".equals(routed)) {
            fail("route-current-profile", "expected Wildfire, got " + routed);
            return;
        }
        routed = (String) route.invoke(model, "unknown-policy");
        if (!"Pauper-Wildfire-Value".equals(routed)) {
            fail("route-unknown-to-current", "expected Wildfire, got " + routed);
            return;
        }
        pass("route-current-profile");
    }

    private static void testRouteExplicitProfile(SharedGpuPythonModel model, Method route) throws Exception {
        ProfileContext.setCurrent(ProfileContext.byName("Pauper-Wildfire-Value"));
        String routed = (String) route.invoke(model, "Pauper-Rally-Anchor-Value");
        if (!"Pauper-Rally-Anchor-Value".equals(routed)) {
            fail("route-bare-profile", "expected Rally, got " + routed);
            return;
        }
        routed = (String) route.invoke(model, "profile:Pauper-Affinity-Anchor-Value");
        if (!"Pauper-Affinity-Anchor-Value".equals(routed)) {
            fail("route-prefixed-profile", "expected Affinity, got " + routed);
            return;
        }
        pass("route-explicit-profile");
    }

    @SuppressWarnings("unchecked")
    private static void testRegisterHeadersUseRequestedProfile(SharedGpuPythonModel model, Method headers) throws Exception {
        ProfileContext.setCurrent(ProfileContext.byName("Pauper-Spy-Combo-Value"));
        Map<String, String> routed = (Map<String, String>) headers.invoke(model, "Pauper-Wildfire-Value");
        if (!"Pauper-Wildfire-Value".equals(routed.get("profile_id"))) {
            fail("headers-profile-id", "got " + routed.get("profile_id"));
            return;
        }
        if (!"Pauper-Wildfire-Value".equals(routed.get("env.MODEL_PROFILE"))) {
            fail("headers-model-profile", "got " + routed.get("env.MODEL_PROFILE"));
            return;
        }
        String modelPath = routed.get("env.MODEL_PATH");
        String expected = "profiles/Pauper-Wildfire-Value/models/model.pt";
        if (modelPath == null || !modelPath.replace('\\', '/').contains(expected)) {
            fail("headers-model-path", "expected path containing " + expected + ", got " + modelPath);
            return;
        }
        pass("register-headers-use-requested-profile");
    }

    private static void registerProfile(String name, String artifactsRoot) {
        ProfileContext.register(new ProfileContext(name, new ProfilePaths(name, artifactsRoot)));
    }

    private static void requireEnv(String key, String expected) {
        String actual = System.getenv(key);
        if (!expected.equals(actual)) {
            System.err.println("FATAL: set " + key + "=" + expected + " before running this test");
            System.exit(2);
        }
    }

    private static void pass(String name) {
        System.out.println("  PASS  " + name);
        passed++;
    }

    private static void fail(String name, String why) {
        System.err.println("  FAIL  " + name + " : " + why);
        failed++;
    }
}
