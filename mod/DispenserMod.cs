using BepInEx;
using UnityEngine;
// using YourGameNamespace; // add if dnSpy shows a namespace for PlayerControls

namespace DispenserMod
{
    [BepInPlugin("com.yourname.dispensermod", "Dispenser Mod", "1.0.0")]
    public class DispenserModPlugin : BaseUnityPlugin
    {
        private PickupItemSpawner[] _players = new PickupItemSpawner[0];
        private float _scanTimer;
        private float _logTimer;

        private void Start()
        {
            Logger.LogInfo("DispenserMod: Start - trying to find PickupItemSpawner instances...");
            RefreshPlayers();
        }

        private void Update()
        {
            // Periodically re-scan in case players spawn after Start
            _scanTimer += Time.deltaTime;
            if (_scanTimer >= 2f)
            {
                _scanTimer = 0f;
                RefreshPlayers();
            }

            if (_players == null || _players.Length == 0)
                return;

            _logTimer += Time.deltaTime;
            if (_logTimer >= 0.25f) // log 4x/sec, adjust as you like
            {
                _logTimer = 0f;

                for (int i = 0; i < _players.Length; i++)
                {
                    var pc = _players[i];
                    if (pc == null)
                        continue;

                    // base.gameObject.transform in dnSpy == pc.transform or pc.gameObject.transform
                    Transform t = pc.transform;
                    if (t == null)
                        continue;

                    Vector3 pos = t.position;
                    Logger.LogInfo($"PickupItemSpawner[{i}] ({pc.name}) pos: {pos}");
                }
            }
        }

        private void RefreshPlayers()
        {
            var found = Object.FindObjectsOfType<PickupItemSpawner>();
            _players = found ?? new PickupItemSpawner[0];
            Logger.LogInfo($"DispenserMod: Found {_players.Length} PickupItemSpawner instances.");
        }
    }
}