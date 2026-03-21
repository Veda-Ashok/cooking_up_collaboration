using BepInEx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
// using YourGameNamespace; // add if dnSpy shows a namespace for PlayerControls

namespace ExtractionMod
{
    [BepInPlugin("com.yourname.extractionmod", "Extraction Mod", "1.0.0")]
    public class ExtractionModPlugin : BaseUnityPlugin
    {
        private Dictionary<Type, MonoBehaviour[]> _trackedObjects = [];
        private Dictionary<string, string> _objectTypeMapping = new()
        {
            { "CookingStation", "P" },
            { "Workstation", "W" },
            { "PlateStation", "D" },
            { "PlateReturnStation", "R" },
            { "PickupItemSpawner", "C" },
            { "IngredientContainer", "I" },
            { "PlayerControls", "@" }
        };
        private Dictionary<int, string[]> _recipeIngredientMapping = new()
        {
            { 11772, new string[] { "Onion", "Onion", "Onion" } },
            { 12064, new string[] { "Tomato", "Tomato", "Tomato" } },
            { 18460, new string[] { "Mushroom", "Mushroom", "Mushroom" } },
            { 22012, new string[] { "Bun", "Meat" } },
            { 21644, new string[] { "Bun", "Meat", "Lettuce" } },
            { 9512, new string[] { "Bun", "Meat", "Lettuce", "Tomato" } },
        };
        private HashSet<int> _unmappedRecipeIds = [];
        private float _scanTimer;
        private float _logTimer;

        private string[,] _layoutGrid;
        private List<Player> _players;
        private List<Element> _objects;
        private List<Order> _orders;
        private List<LayoutObject> _layoutObjects;
        private List<float[]> _playerStartPositions;
        private List<IOrderController> _orderControllers;
        private int _gameloop;

        private void Start()
        {

            _gameloop = 0;
            _layoutObjects = [];
            _playerStartPositions = [];
            _orderControllers = [];
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

                LogOrderDebugInfo();
                Logger.LogInfo($"ComputeState finished. Players: {_players.Count}, Objects: {_objects.Count}");
                string temp = JsonGenerator.GenerateGameJson(_players, _objects, _orders, _layoutGrid, "layout1", time_left, time_elapsed, _gameloop);
                JsonGenerator.GenerateLevelJson(_layoutObjects, _playerStartPositions);
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
                    Player p = new()
                    {
                        position = [x, z],
                        orientation = [0, 1], //TODO: Get actual orientation from the player object
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
                    Element e = new()
                    {
                        name = obj.name,
                        position = new int[] { x, z }
                    };
                    objects.Add(e);
                }
            }

            GameObject[] ingredients = GetAllIngredients();
            for (int i = 0; i < ingredients.Length; i++)
            {
                GameObject ingredient = ingredients[i];
                if (ingredient == null)
                {
                    continue;
                }

                Vector3 pos = ingredient.transform.position;
                int x = ComputePosition(pos.x, max_x, min_x, m);
                int z = ComputePosition(pos.z, max_z, min_z, n);
                Logger.LogInfo($"G ({ingredient.name}) tag: {ingredient.tag} pos: {pos} mapped to grid ({z},{x})");
                objects.Add(new Element
                {
                    name = ingredient.name,
                    position = [x, z]
                });
            }

            _objects = objects;
            _players = players;
            _orders = BuildOrdersFromControllers();
            _layoutObjects = BuildLayoutObjects();
            _playerStartPositions = BuildPlayerStartPositions();
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

            List<Type> staticElements =
            [
                typeof(CookingStation),
                typeof(Workstation),
                typeof(PlateStation),
                typeof(PlateReturnStation),
                typeof(PickupItemSpawner),
            ];
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
            T[] found = FindObjectsOfType<T>();

            // Store them in the dictionary using the Type as the key
            _trackedObjects[typeof(T)] = found;

            Logger.LogInfo($"ExtractionMod: Found {found.Length} {typeof(T).Name} instances.");
        }

        private void RefreshAndLogObjects<T>() where T : MonoBehaviour
        {
            RefreshObjects<T>();

            if (_trackedObjects.TryGetValue(typeof(T), out MonoBehaviour[] trackedObjects))
            {
                LogPositions((T[])trackedObjects);
            }
        }

        private void RefreshOrderControllers()
        {
            CampaignFlowController[] flows = FindObjectsOfType<CampaignFlowController>();
            List<IOrderController> found = [];

            for (int i = 0; i < flows.Length; i++)
            {
                CampaignFlowController flow = flows[i];
                if (flow == null)
                {
                    continue;
                }

                IOrderController orderController = flow.GetOrderController();
                if (orderController == null)
                {
                    Logger.LogInfo($"ExtractionMod: CampaignFlowController '{flow.name}' returned no order controller.");
                    continue;
                }

                found.Add(orderController);
                Logger.LogInfo(
                    $"ExtractionMod: CampaignFlowController '{flow.name}' has order controller type '{orderController.GetType().FullName}'.");
            }

            _orderControllers = found;
            Logger.LogInfo($"ExtractionMod: Found {_orderControllers.Count} order controller references via {flows.Length} CampaignFlowController instances.");
        }

        private void RefreshAllObjects()
        {
            RefreshObjects<IngredientContainer>(); // Pots and Plates
            RefreshObjects<PlateReturnStation>(); // Dirty plates station and sink clean plates station
            RefreshObjects<PlayerControls>(); // Players
            RefreshObjects<PickupItemSpawner>(); // Dispenser crates
            RefreshObjects<AttachStation>(); // TableTops DryingPart (clean plates), PlateStation, and chopping boards.
            RefreshObjects<Interactable>(); // Fire extinguisher, WashPart (sink), and chopping boards.
            RefreshObjects<RubbishBin>(); // Rubbish bins, they're British!
            RefreshOrderControllers();
            LogPickupItemSpawnerDetails();
            Logger.LogInfo($"ExtractionMod: Found {GetAllIngredients().Length} tagged ingredient GameObjects.");

            // RefreshObjects<PlateStation>(); // Delivery station. Included in AttachStation.
            // RefreshObjects<Workstation>(); // Cutting board. Included in AttachStation.
            // RefreshObjects<PlacementContainer>(); // The same as IngredientContainer.
            // RefreshObjects<CookingStation>(); // Stove. Included in AttachStation.
            // RefreshObjects<PickupItemSpawner>(); // Dispenser crates. Included in AttacheStation.
        }

        private GameObject[] GetAllIngredients()
        {
            GameObject[] preIngredients = FindGameObjectsWithTagSafe("Pre-Ingredient");
            GameObject[] ingredients = FindGameObjectsWithTagSafe("Ingredient");
            return preIngredients.Union(ingredients).ToArray();
        }

        private GameObject[] FindGameObjectsWithTagSafe(string tagName)
        {
            try
            {
                return GameObject.FindGameObjectsWithTag(tagName);
            }
            catch (UnityException)
            {
                Logger.LogInfo($"ExtractionMod: Tag '{tagName}' not found in current scene.");
                return [];
            }
        }

        private List<Order> BuildOrdersFromControllers()
        {
            List<Order> orders = [];
            if (_orderControllers == null || _orderControllers.Count == 0)
            {
                return orders;
            }

            foreach (IOrderController orderController in _orderControllers)
            {
                if (orderController == null)
                {
                    continue;
                }

                orders.AddRange(ExtractOrdersFromController(orderController));
            }

            return orders;
        }

        private List<LayoutObject> BuildLayoutObjects()
        {
            List<LayoutObject> layoutObjects = [];
            AppendLayoutObjects<AttachStation>(layoutObjects);
            AppendLayoutObjects<Interactable>(layoutObjects);
            AppendLayoutObjects<IngredientContainer>(layoutObjects);
            AppendLayoutObjects<PlateReturnStation>(layoutObjects);
            AppendLayoutObjects<RubbishBin>(layoutObjects);

            Logger.LogInfo($"ExtractionMod: Built {layoutObjects.Count} layout objects for export.");
            return layoutObjects;
        }

        private List<float[]> BuildPlayerStartPositions()
        {
            List<float[]> playerStartPositions = [];
            if (!_trackedObjects.TryGetValue(typeof(PlayerControls), out MonoBehaviour[] trackedObjects))
            {
                return playerStartPositions;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                PlayerControls player = trackedObjects[i] as PlayerControls;
                if (player == null)
                {
                    continue;
                }

                Vector3 pos = player.transform.position;
                playerStartPositions.Add([pos.x, pos.z]);
            }

            return playerStartPositions;
        }

        private void AppendLayoutObjects<T>(List<LayoutObject> layoutObjects)
            where T : MonoBehaviour
        {
            if (!_trackedObjects.TryGetValue(typeof(T), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                T obj = trackedObjects[i] as T;
                if (obj == null)
                {
                    continue;
                }

                Vector3 pos = obj.transform.position;
                layoutObjects.Add(new LayoutObject
                {
                    type = ClassifyLayoutObject(obj),
                    ingredient = GetLayoutObjectIngredient(obj),
                    position = [pos.x, pos.z]
                });
            }
        }

        private string ClassifyLayoutObject(MonoBehaviour obj)
        {
            if (obj == null)
            {
                return "unknown";
            }

            string name = obj.name ?? string.Empty;

            if (obj is AttachStation)
            {
                if (name.StartsWith("TableTop_", StringComparison.OrdinalIgnoreCase))
                {
                    return "tabletop";
                }

                if (name.IndexOf("ChoppingBoard", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "chopping_board";
                }

                if (name.IndexOf("CookingStation", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "stove";
                }

                if (name.IndexOf("DispenserCrate", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "dispenser";
                }

                if (name.IndexOf("PlateStation", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "delivery_station";
                }

                if (name.IndexOf("PlateReturn", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "plate_return";
                }

                if (name.IndexOf("DryingPart", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "drying_rack";
                }

                return "attach_station";
            }

            if (obj is Interactable)
            {
                if (name.IndexOf("FireExtinguisher", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "fire_extinguisher";
                }

                if (name.IndexOf("WashingPart", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "sink";
                }

                if (name.IndexOf("ChoppingBoard", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "chopping_board";
                }

                return "interactable";
            }

            if (obj is IngredientContainer)
            {
                if (name.IndexOf("Pot", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "pot";
                }

                if (name.IndexOf("Plate", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    return "plate";
                }

                return "ingredient_container";
            }

            if (obj is PlateReturnStation)
            {
                return "plate_return_station";
            }

            if (obj is RubbishBin)
            {
                return "rubbish_bin";
            }

            return obj.GetType().Name;
        }

        private string GetLayoutObjectIngredient(MonoBehaviour obj)
        {
            if (obj == null)
            {
                return null;
            }

            if (obj is AttachStation && obj.name.IndexOf("DispenserCrate", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                PickupItemSpawner spawner = FindMatchingPickupItemSpawner(obj.transform.position);
                string ingredientName = InferIngredientName(spawner);
                if (!string.IsNullOrEmpty(ingredientName))
                {
                    return ingredientName.ToLowerInvariant();
                }
            }

            return null;
        }

        private PickupItemSpawner FindMatchingPickupItemSpawner(Vector3 position)
        {
            if (!_trackedObjects.TryGetValue(typeof(PickupItemSpawner), out MonoBehaviour[] trackedObjects))
            {
                return null;
            }

            PickupItemSpawner bestMatch = null;
            float bestDistance = 0.75f;

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                PickupItemSpawner spawner = trackedObjects[i] as PickupItemSpawner;
                if (spawner == null)
                {
                    continue;
                }

                float distance = Vector3.Distance(spawner.transform.position, position);
                if (distance <= bestDistance)
                {
                    bestDistance = distance;
                    bestMatch = spawner;
                }
            }

            return bestMatch;
        }

        private void LogPickupItemSpawnerDetails()
        {
            if (!_trackedObjects.TryGetValue(typeof(PickupItemSpawner), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                PickupItemSpawner spawner = trackedObjects[i] as PickupItemSpawner;
                if (spawner == null)
                {
                    continue;
                }

                GameObject itemPrefab = spawner.GetItemPrefab();
                string inferredIngredient = InferIngredientName(spawner);

                Logger.LogInfo(
                    $"ExtractionMod: PickupItemSpawner '{spawner.name}' pos: {spawner.transform.position} " +
                    $"itemPrefab='{(itemPrefab != null ? itemPrefab.name : "<null>")}' " +
                    $"ingredient='{(inferredIngredient ?? "<unknown>")}'");
            }
        }

        private string InferIngredientName(PickupItemSpawner spawner)
        {
            if (spawner == null)
            {
                return null;
            }

            GameObject itemPrefab = spawner.GetItemPrefab();
            return itemPrefab != null ? itemPrefab.name : null;
        }

        private void LogOrderDebugInfo()
        {
            if (_orderControllers == null || _orderControllers.Count == 0)
            {
                Logger.LogInfo("ExtractionMod: No order controllers available for debug logging.");
                return;
            }

            for (int controllerIndex = 0; controllerIndex < _orderControllers.Count; controllerIndex++)
            {
                IOrderController orderController = _orderControllers[controllerIndex];
                if (orderController == null)
                {
                    continue;
                }

                FieldInfo activeRecipesField = orderController.GetType().GetField("m_activeRecipes", BindingFlags.Instance | BindingFlags.NonPublic);
                IEnumerable activeRecipes = activeRecipesField != null ? activeRecipesField.GetValue(orderController) as IEnumerable : null;
                if (activeRecipes == null)
                {
                    Logger.LogInfo($"ExtractionMod: Order controller '{orderController.GetType().FullName}' has no readable m_activeRecipes.");
                    continue;
                }

                int recipeIndex = 0;
                foreach (object activeRecipe in activeRecipes)
                {
                    if (activeRecipe == null)
                    {
                        continue;
                    }

                    RecipeList.Entry entry = GetFieldValue<RecipeList.Entry>(activeRecipe, "RecipeListEntry");
                    OrderDefinitionNode orderNode = entry != null ? entry.m_order : null;
                    string[] extractedIngredients = entry != null ? ExtractIngredients(entry) : [];

                    Logger.LogInfo(
                        $"OrderDebug controller={controllerIndex} type='{orderController.GetType().FullName}' recipe={recipeIndex} " +
                        $"recipeId={(orderNode != null ? orderNode.m_uID.ToString() : "<null>")} " +
                        $"plating='{(orderNode != null && orderNode.m_platingPrefab != null ? orderNode.m_platingPrefab.name : "<null>")}' " +
                        $"ingredients=[{string.Join(", ", extractedIngredients)}]");
                    recipeIndex++;
                }

                if (recipeIndex == 0)
                {
                    Logger.LogInfo($"ExtractionMod: Order controller '{orderController.GetType().FullName}' currently has no active recipes.");
                }
            }
        }

        private List<Order> ExtractOrdersFromController(IOrderController orderController)
        {
            List<Order> orders = [];
            FieldInfo activeRecipesField = orderController.GetType().GetField("m_activeRecipes", BindingFlags.Instance | BindingFlags.NonPublic);
            if (activeRecipesField == null)
            {
                Logger.LogInfo($"ExtractionMod: '{orderController.GetType().FullName}' does not expose m_activeRecipes.");
                return orders;
            }

            IEnumerable activeRecipes = activeRecipesField.GetValue(orderController) as IEnumerable;
            if (activeRecipes == null)
            {
                return orders;
            }

            foreach (object activeRecipe in activeRecipes)
            {
                if (activeRecipe == null)
                {
                    continue;
                }

                RecipeList.Entry entry = GetFieldValue<RecipeList.Entry>(activeRecipe, "RecipeListEntry");
                if (entry == null)
                {
                    continue;
                }

                string[] ingredients = ExtractIngredients(entry);
                if (ingredients.Length == 0)
                {
                    ingredients = [$"recipe_{entry.m_order.m_uID}"];
                }

                orders.Add(new Order
                {
                    ingredients = ingredients
                });
            }

            return orders;
        }

        private string[] ExtractIngredients(RecipeList.Entry entry)
        {
            if (entry == null || entry.m_order == null)
            {
                return [];
            }

            string[] mappedIngredients;
            if (_recipeIngredientMapping.TryGetValue(entry.m_order.m_uID, out mappedIngredients))
            {
                return mappedIngredients;
            }

            if (_unmappedRecipeIds.Add(entry.m_order.m_uID))
            {
                Logger.LogInfo($"ExtractionMod: Unmapped recipeId {entry.m_order.m_uID} for order type '{entry.m_order.GetType().FullName}'.");
            }

            return [];
        }

        private T GetFieldValue<T>(object instance, string fieldName) where T : class
        {
            if (instance == null)
            {
                return null;
            }

            FieldInfo field = instance.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (field == null)
            {
                return null;
            }

            return field.GetValue(instance) as T;
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
