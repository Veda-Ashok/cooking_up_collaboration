using BepInEx;
using System;
using System.Collections.Generic;
using UnityEngine;
// using YourGameNamespace; // add if dnSpy shows a namespace for PlayerControls

namespace ExtractionMod
{
    [BepInPlugin("com.yourname.extractionmod", "Extraction Mod", "1.0.0")]
    public class ExtractionModPlugin : BaseUnityPlugin
    {
        private Dictionary<Type, MonoBehaviour[]> _trackedObjects = new Dictionary<Type, MonoBehaviour[]>();
        private Dictionary<string, string> _objectTypeMapping = new Dictionary<string, string>
        {
            { "CookingStation", "P" },
            { "Workstation", "W" },
            { "PlateStation", "D" },
            { "PlateReturnStation", "R" },
            { "PickupItemSpawner", "C" },
            { "IngredientContainer", "I" },
            { "PlayerControls", "@" }
        };
        private float _scanTimer;
        private float _logTimer;

        private string[,] _layoutGrid;
        private List<Player> _players;
        private List<Element> _objects;
        private List<Order> _orders;
        private int _gameloop;

        private void Start()
        {
            
            _gameloop = 0;
            Logger.LogInfo("ExtractionMod: Started and scanning for objects...");
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

                //TODO: Determine the number of columns and rows based on the game board
                //TODO: Automate getting these bounds from the game board
                float max_x = 24f;
                float min_x = 9f;
                float max_z = 11f;
                float min_z = 1f;
                int n = 9; // rows
                int m = 12; // columns
                ComputeLayout(n, m, max_x, min_x, max_z, min_z);
                ComputeState(n, m, max_x, min_x, max_z, min_z);
                float time_left = 120f; //TODO: Get actual time left in the game
                float time_elapsed = Time.time; //TODO: Get actual time elapsed in the game
                _gameloop++;

                Logger.LogInfo($"ComputeState finished. Players: {_players.Count}, Objects: {_objects.Count}");
                string temp =JsonGenerator.GenerateGameJson(_players, _objects, _orders, _layoutGrid, "layout1", time_left, time_elapsed, _gameloop);
                Logger.LogInfo($"state: {temp}");

            }
        }

        private void ComputeState(int n, int m, float max_x, float min_x, float max_z, float min_z)
        {
            List<Player> players = new List<Player>();
            if (_trackedObjects.ContainsKey(typeof(PlayerControls)))
            {
                foreach (var obj in _trackedObjects[typeof(PlayerControls)])
                {
                    Vector3 pos = obj.transform.position;
                    int x = ComputePosition(pos.x, max_x, min_x, m);
                    int z = ComputePosition(pos.z, max_z, min_z, n);
                    string symbol = _objectTypeMapping.ContainsKey("PlayerControls") ? _objectTypeMapping["PlayerControls"] : "?";
                    Logger.LogInfo($"{symbol} ({obj.name}) pos: {pos} mapped to grid ({z},{x})");
                    Player p = new Player 
                    {
                        position = new int[] { x, z },
                        orientation = new int[] { 0, 1 }, //TODO: Get actual orientation from the player object
                        held_object = null //TODO: Get actual held object from the player object 
                    };
                    players.Add(p);
                }
            }

            List<Element> objects = new List<Element>(); //TODO: Add ingredients and other interactable objects in the game to this list
            
            if (_trackedObjects.ContainsKey(typeof(IngredientContainer)))
            {
                foreach (var obj in _trackedObjects[typeof(IngredientContainer)])
                {
                    Vector3 pos = obj.transform.position;
                    int x = ComputePosition(pos.x, max_x, min_x, m);
                    int z = ComputePosition(pos.z, max_z, min_z, n);
                    string symbol = _objectTypeMapping.ContainsKey("IngredientContainer") ? _objectTypeMapping["IngredientContainer"] : "?";
                    Logger.LogInfo($"{symbol} ({obj.name}) pos: {pos} mapped to grid ({z},{x})");
                    Element e = new Element 
                    {
                        name = obj.name,
                        position = new int[] { x, z }
                    };
                    objects.Add(e);
                }
            }

            _objects = objects;
            _players = players;
            Order o = new Order 
            {
                ingredients = new string[] { "Onion", "Onion", "Onion"}
            };
            _orders = [o]; //TODO: Fill this with the orders in the game
        }

        private void ComputeLayout(int n, int m, float max_x, float min_x, float max_z, float min_z)
        {

            _layoutGrid = new string[n,m];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    _layoutGrid[i,j] = "";
                }
            }

            List<Type> staticElements = new List<Type>
            {
                typeof(CookingStation),
                typeof(Workstation),
                typeof(PlateStation),
                typeof(PlateReturnStation),
                typeof(PickupItemSpawner),
            };
            foreach (var type in staticElements)
            {
                if (_trackedObjects.ContainsKey(type))
                {
                    foreach (var obj in _trackedObjects[type])
                    {
                        Vector3 pos = obj.transform.position;
                        int x = ComputePosition(pos.x, max_x, min_x, m);
                        int z = ComputePosition(pos.z, max_z, min_z, n);
                        string symbol = _objectTypeMapping.ContainsKey(type.Name) ? _objectTypeMapping[type.Name] : "?";
                        Logger.LogInfo($"{symbol} ({obj.name}) pos: {pos} mapped to grid ({z},{x})");
                        _layoutGrid[z,x] = symbol;
                    }
                }
            }
        }

        private int ComputePosition(float coord, float max, float min, int n)
        {
            float cellSize = (max - min) / n;
            int index = Mathf.Clamp((int)((coord - min) / cellSize), 0, n - 1);
            return index;
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

        private void RefreshObjects<T>() where T : MonoBehaviour
{
            // Find all instances of type T
            T[] found = UnityEngine.Object.FindObjectsOfType<T>();
            
            // Store them in the dictionary using the Type as the key
            _trackedObjects[typeof(T)] = found;
            
            Logger.LogInfo($"ExtractionMod: Found {found.Length} {typeof(T).Name} instances.");
        }

        private void RefreshAllObjects()
        {
            RefreshObjects<PickupItemSpawner>(); //Dispenser Crate
            RefreshObjects<IngredientContainer>(); //Pots and Plates
            RefreshObjects<CookingStation>(); //Stove
            RefreshObjects<PlateReturnStation>(); //Dirty plates station and sink clean plates station
            RefreshObjects<PlateStation>(); //Delivery station
            RefreshObjects<Workstation>(); //Cutting board
            RefreshObjects<PlayerControls>(); //Players
            //Missing sink, onions and order list
        }

        private void LogAllPositions()
        {
            // Iterate through all arrays(types) currently stored in the dictionary
            foreach (var trackedArray in _trackedObjects.Values)
            {
                //LogAllPlayerData((PlayerControls[])_trackedObjects[typeof(PlayerControls)]); 
                LogPositions(trackedArray);
            }
        }
    }
}