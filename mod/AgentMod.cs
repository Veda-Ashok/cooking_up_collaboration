using BepInEx;
using HarmonyLib;
using UnityEngine;
// using YourGameNamespace;

namespace AgentMod
{
    [BepInPlugin("com.yourname.agentmod", "Agent Mod", "1.0.0")]
    public class AgentModPlugin : BaseUnityPlugin
    {
        private void Awake()
        {
            Logger.LogInfo("AgentMod: Awake - applying Harmony patches.");
            var harmony = new Harmony("com.yourname.agentmod");
            harmony.PatchAll();
        }
    }

    [HarmonyPatch(typeof(PlayerControls))]
    public static class PlayerControlsPatches
    {
        // Patch the PlayerControls.Update() method
        [HarmonyPostfix]
        [HarmonyPatch("Update")]
        public static void UpdatePostfix(PlayerControls __instance)
        {
            if (__instance == null)
                return;

            Transform t = __instance.transform; // equivalent to base.gameObject.transform
            if (t == null)
                return;

            Vector3 pos = t.position;
            // Debug.Log($"[AgentMod] PlayerControls '{__instance.name}' at {pos}");
        }
    }

    [HarmonyPatch(typeof(IngredientContainer))]
    public static class IngredientContainerPatches
    {
        // simple global throttle so we don't log every frame
        private static float _logCooldown = 0.25f;
        private static float _timeSinceLastLog;

        [HarmonyPostfix]
        [HarmonyPatch("Update")]
        public static void UpdatePostfix(IngredientContainer __instance)
        {
            if (__instance == null)
                return;

            // Throttle logs to ~4 times per second globally
            _timeSinceLastLog += Time.deltaTime;
            if (_timeSinceLastLog < _logCooldown)
                return;

            _timeSinceLastLog = 0f;

            Transform t = __instance.transform; // same as base.gameObject.transform
            if (t == null)
                return;

            Vector3 pos = t.position;
            Debug.Log($"[AgentMod] IngredientContainer '{__instance.name}' at {pos}");
        }
    }
}