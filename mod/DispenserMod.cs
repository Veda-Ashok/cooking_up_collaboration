using BepInEx;
using System;
using System.Collections.Generic;
using UnityEngine;
// using YourGameNamespace; // add if dnSpy shows a namespace for PlayerControls

namespace DispenserMod
{
    [BepInPlugin("com.yourname.dispensermod", "Dispenser Mod", "1.0.0")]
    public class DispenserModPlugin : BaseUnityPlugin
    {
        private Dictionary<Type, MonoBehaviour[]> _trackedObjects = new Dictionary<Type, MonoBehaviour[]>();
        private float _scanTimer;
        private float _logTimer;

        private void Start()
        {
            Logger.LogInfo("DispenserMod: Start - trying to find PickupItemSpawner instances...");
            RefreshAllObjects();
        }

        private void Update()
        {
            // Periodically re-scan to update positions
            _scanTimer += Time.deltaTime;
            if (_scanTimer >= 5f)
            {
                _scanTimer = 0f;
                RefreshAllObjects();
            }

            _logTimer += Time.deltaTime;
            if (_logTimer >= 10f) // log 1x/sec, adjust as you like
            {
                _logTimer = 0f;
                LogAllPositions();
            }
        }

        private void LogPositions<T>(T[] objects) where T : MonoBehaviour
        {
            if (objects == null) return;

            if (objects.Length == 0)            {
                Logger.LogInfo($"No {typeof(T).Name} instances found.");
                return;
            }

            for (int i = 0; i < objects.Length; i++)
            {
                var obj = objects[i];
                if (obj == null) continue;

                Vector3 pos = obj.transform.position;
                Logger.LogInfo($"{typeof(T).Name}[{i}] ({obj.name}) pos: {pos}");
            }
        }

        private void RefreshAllObjects()
        {
            RefreshObjects<PickupItemSpawner>(); //Dispenser Crate
            RefreshObjects<PlayerControls>(); //Players
            RefreshObjects<IngredientContainer>(); //Pots and Plates
            RefreshObjects<CookingStation>(); //CookingStation
            //RefreshObjects<CookableIngredient>(); //don't know what it is
            RefreshObjects<PlateReturnStation>(); //Dirty plates station and sink clean plates station
            RefreshObjects<PlateStation>(); //Delivery station
            //RefreshObjects<PreparationContainer>(); //don't know what it is
            RefreshObjects<Workstation>(); //Cutting board
            //Missing sink, onions and order list
        }

        private void LogAllPositions()
        {
            LogPositions((PickupItemSpawner[])_trackedObjects[typeof(PickupItemSpawner)]); //Dispenser Crate
            LogPositions((PlayerControls[])_trackedObjects[typeof(PlayerControls)]); //Players
            LogPositions((IngredientContainer[])_trackedObjects[typeof(IngredientContainer)]); //Pots and Plates
            LogPositions((CookingStation[])_trackedObjects[typeof(CookingStation)]); //CookingStation
            LogPositions((PlateReturnStation[])_trackedObjects[typeof(PlateReturnStation)]);
            LogPositions((PlateStation[])_trackedObjects[typeof(PlateStation)]);
            LogPositions((Workstation[])_trackedObjects[typeof(Workstation)]);
        }

        private void RefreshObjects<T>() where T : MonoBehaviour
{
            // Find all instances of type T
            T[] found = UnityEngine.Object.FindObjectsOfType<T>();
            
            // Store them in the dictionary using the Type as the key
            _trackedObjects[typeof(T)] = found;
            
            Logger.LogInfo($"DispenserMod: Found {found.Length} {typeof(T).Name} instances.");
        }
    }
}