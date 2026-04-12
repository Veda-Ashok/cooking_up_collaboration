using BepInEx;
using BepInEx.Configuration;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;
using System.Threading;
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
        private float _publishTimer;
        private float _logTimer;
        private HashSet<string> _reflectionWarnings = [];

        private string[,] _layoutGrid;
        private List<PlayerStateDto> _players;
        private List<ObjectStateDto> _objects;
        private List<Order> _orders;
        private List<LayoutObject> _layoutObjects;
        private List<float[]> _playerStartPositions;
        private List<IOrderController> _orderControllers;
        private int _gameloop;
        private string _debugLogPath;
        private string _lastLevelJson;
        private string _queuedStateJson;
        private string _queuedLevelJson;
        private string _lastStreamErrorMessage;
        private float _lastStreamErrorLogTime;
        private bool _streamWorkerActive;
        private readonly object _streamLock = new object();
        private ConfigEntry<bool> _writeJsonFilesConfig;
        private ConfigEntry<float> _publishIntervalSecondsConfig;
        private ConfigEntry<float> _statusLogIntervalSecondsConfig;
        private ConfigEntry<bool> _enableHttpStreamConfig;
        private ConfigEntry<string> _httpStreamBaseUrlConfig;
        private ConfigEntry<int> _httpStreamTimeoutMillisecondsConfig;

        private sealed class PlayerCandidate
        {
            public PlayerControls Source;
            public int PlayerIndex;
            public PlayerStateDto State;
        }

        private sealed class ObjectCandidate
        {
            public int SourceKey;
            public UnityEngine.Object Source;
            public ObjectStateDto State;
            public bool IsHolder;
            public bool AllowsMultipleChildren;
            public List<ObjectCandidate> Children = [];
        }

        private void Awake()
        {
            _debugLogPath = Path.Combine(Paths.GameRootPath, "extraction_mod_debug.log");
            _writeJsonFilesConfig = Config.Bind("Output", "WriteJsonFiles", true, "Write state.json and level.json to disk.");
            _publishIntervalSecondsConfig = Config.Bind("Output", "PublishIntervalSeconds", 0.1f, "Seconds between state publishes.");
            _statusLogIntervalSecondsConfig = Config.Bind("Diagnostics", "StatusLogIntervalSeconds", 5.0f, "Seconds between diagnostic status logs.");
            _enableHttpStreamConfig = Config.Bind("Streaming", "EnableHttpStream", true, "POST state and level JSON to the configured HTTP endpoint.");
            _httpStreamBaseUrlConfig = Config.Bind("Streaming", "HttpStreamBaseUrl", "http://127.0.0.1:8765", "Base URL for the Python receiver.");
            _httpStreamTimeoutMillisecondsConfig = Config.Bind("Streaming", "HttpTimeoutMilliseconds", 250, "HTTP timeout for localhost streaming.");
            ServicePointManager.Expect100Continue = false;
            SafeLog("Awake");
        }

        private void Start()
        {
            try
            {
                _gameloop = 0;
                _layoutObjects = [];
                _playerStartPositions = [];
                _orderControllers = [];
                Logger.LogInfo("ExtractionMod: Started and scanning for objects...");
                SafeLog("Start");
                RefreshAllObjects();
            }
            catch (Exception ex)
            {
                SafeLog("Start exception: " + ex);
                throw;
            }
        }

        private void Update()
        {
            try
            {
                _scanTimer += Time.deltaTime;
                if (_scanTimer >= 5f)
                {
                    _scanTimer = 0f;
                    RefreshAllObjects();
                }

                _publishTimer += Time.deltaTime;
                _logTimer += Time.deltaTime;
                float publishIntervalSeconds = Mathf.Max(_publishIntervalSecondsConfig.Value, 0.01f);
                if (_publishTimer < publishIntervalSeconds)
                {
                    return;
                }

                _publishTimer = 0f;

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
                _gameloop++;

                string stateJson = JsonGenerator.GenerateGameJson(_players, _objects, _writeJsonFilesConfig.Value);
                string levelJson = JsonGenerator.GenerateLevelJson(_layoutObjects, _playerStartPositions, false);
                bool levelChanged = !string.Equals(_lastLevelJson, levelJson, StringComparison.Ordinal);
                if (levelChanged)
                {
                    _lastLevelJson = levelJson;
                    if (_writeJsonFilesConfig.Value)
                    {
                        File.WriteAllText("level.json", levelJson);
                    }
                }

                if (_enableHttpStreamConfig.Value)
                {
                    EnqueueStreamPayloads(stateJson, levelChanged ? levelJson : null);
                }

                float statusLogIntervalSeconds = Mathf.Max(_statusLogIntervalSecondsConfig.Value, 0.1f);
                if (_logTimer >= statusLogIntervalSeconds)
                {
                    _logTimer = 0f;
                    LogOrderDebugInfo();
                    string streamTarget = _enableHttpStreamConfig.Value ? _httpStreamBaseUrlConfig.Value : "disabled";
                    Logger.LogInfo($"ComputeState ok loop={_gameloop} players={_players.Count} objects={_objects.Count} stream={streamTarget}");
                    SafeLog($"ComputeState ok loop={_gameloop} players={_players.Count} objects={_objects.Count} stream={streamTarget}");
                }
            }
            catch (Exception ex)
            {
                SafeLog("Update exception: " + ex);
                throw;
            }
        }

        private void EnqueueStreamPayloads(string stateJson, string levelJson)
        {
            if (string.IsNullOrEmpty(stateJson) && string.IsNullOrEmpty(levelJson))
            {
                return;
            }

            lock (_streamLock)
            {
                if (!string.IsNullOrEmpty(stateJson))
                {
                    _queuedStateJson = stateJson;
                }

                if (!string.IsNullOrEmpty(levelJson))
                {
                    _queuedLevelJson = levelJson;
                }

                if (_streamWorkerActive)
                {
                    return;
                }

                _streamWorkerActive = true;
            }

            ThreadPool.QueueUserWorkItem(StreamPayloadWorker);
        }

        private void StreamPayloadWorker(object state)
        {
            while (true)
            {
                string stateJson;
                string levelJson;
                lock (_streamLock)
                {
                    stateJson = _queuedStateJson;
                    levelJson = _queuedLevelJson;
                    _queuedStateJson = null;
                    _queuedLevelJson = null;

                    if (string.IsNullOrEmpty(stateJson) && string.IsNullOrEmpty(levelJson))
                    {
                        _streamWorkerActive = false;
                        return;
                    }
                }

                if (!string.IsNullOrEmpty(levelJson))
                {
                    TryPostJson("/level", levelJson);
                }

                if (!string.IsNullOrEmpty(stateJson))
                {
                    TryPostJson("/state", stateJson);
                }
            }
        }

        private void TryPostJson(string relativePath, string json)
        {
            if (string.IsNullOrEmpty(relativePath) || string.IsNullOrEmpty(json))
            {
                return;
            }

            try
            {
                string baseUrl = (_httpStreamBaseUrlConfig.Value ?? string.Empty).TrimEnd('/');
                if (string.IsNullOrEmpty(baseUrl))
                {
                    return;
                }

                string endpoint = baseUrl + relativePath;
                byte[] payload = Encoding.UTF8.GetBytes(json);
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(endpoint);
                request.Method = "POST";
                request.ContentType = "application/json";
                request.Timeout = Math.Max(_httpStreamTimeoutMillisecondsConfig.Value, 50);
                request.ReadWriteTimeout = request.Timeout;
                request.ContentLength = payload.Length;

                using (Stream requestStream = request.GetRequestStream())
                {
                    requestStream.Write(payload, 0, payload.Length);
                }

                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                }
            }
            catch (Exception ex)
            {
                ReportStreamError($"POST {relativePath} failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void ReportStreamError(string message)
        {
            float currentTime = Time.realtimeSinceStartup;
            if (string.Equals(message, _lastStreamErrorMessage, StringComparison.Ordinal) &&
                currentTime - _lastStreamErrorLogTime < 5.0f)
            {
                return;
            }

            _lastStreamErrorMessage = message;
            _lastStreamErrorLogTime = currentTime;
            Logger.LogInfo("ExtractionMod stream error: " + message);
            SafeLog("Stream error: " + message);
        }

        private void ComputeState(int n, int m, float max_x, float min_x, float max_z, float min_z)
        {
            List<PlayerCandidate> playerCandidates = BuildPlayerCandidates();
            List<ObjectCandidate> objectCandidates = BuildObjectCandidates();
            SafeLog($"ComputeState candidates players={playerCandidates.Count} objects={objectCandidates.Count}");

            AssignObjectIds(objectCandidates);
            ResolveObjectRelationships(playerCandidates, objectCandidates);
            PopulateHolderState(objectCandidates);
            PopulatePlayerHeldObjects(playerCandidates, objectCandidates);
            PopulateRackState(objectCandidates);

            _players = playerCandidates
                .OrderBy(player => player.PlayerIndex)
                .Select(player => player.State)
                .ToList();
            _objects = objectCandidates
                .OrderBy(candidate => candidate.State.name)
                .ThenBy(candidate => candidate.State.position[0])
                .ThenBy(candidate => candidate.State.position[1])
                .ThenBy(candidate => candidate.State.id)
                .Select(candidate => candidate.State)
                .ToList();
            _orders = BuildOrdersFromControllers();
            _layoutObjects = BuildLayoutObjects();
            _playerStartPositions = BuildPlayerStartPositions();
        }

        private List<PlayerCandidate> BuildPlayerCandidates()
        {
            List<PlayerCandidate> players = [];
            if (!_trackedObjects.TryGetValue(typeof(PlayerControls), out MonoBehaviour[] trackedPlayers))
            {
                return players;
            }

            for (int i = 0; i < trackedPlayers.Length; i++)
            {
                PlayerControls player = trackedPlayers[i] as PlayerControls;
                if (player == null)
                {
                    continue;
                }

                players.Add(new PlayerCandidate
                {
                    Source = player,
                    PlayerIndex = TryGetPlayerIndex(player) ?? i,
                    State = new PlayerStateDto
                    {
                        position = ToWorldPosition(player.transform.position),
                        facing = ToWorldDirection(player.transform.forward)
                    }
                });
            }

            players = players
                .OrderBy(player => player.PlayerIndex)
                .ThenBy(player => player.Source.name)
                .ThenBy(player => player.State.position[0])
                .ThenBy(player => player.State.position[1])
                .ToList();

            for (int i = 0; i < players.Count; i++)
            {
                players[i].PlayerIndex = i;
                players[i].State.id = $"player_{i}";
                players[i].State.name = players[i].State.id;
            }

            return players;
        }

        private List<ObjectCandidate> BuildObjectCandidates()
        {
            List<ObjectCandidate> objectCandidates = [];
            HashSet<string> seen = [];
            int syntheticKey = -1;

            AddAttachStationCandidates(objectCandidates, seen);
            AddInteractableCandidates(objectCandidates, seen);
            AddIngredientContainerCandidates(objectCandidates, seen);
            AddPlateReturnStationCandidates(objectCandidates, seen);
            AddRubbishBinCandidates(objectCandidates, seen);
            AddIngredientCandidates(objectCandidates, seen);
            AddTrackedDirtyPlateStackCandidates(objectCandidates, seen);
            AddDirtyPlateStackCandidates(objectCandidates, ref syntheticKey);
            AddSoupCandidates(objectCandidates, ref syntheticKey);
            RemoveAliasedStateObjects(objectCandidates);

            return objectCandidates;
        }

        private void RemoveAliasedStateObjects(List<ObjectCandidate> objectCandidates)
        {
            HashSet<int> dryingRackKeys = new HashSet<int>(
                objectCandidates
                    .Where(candidate => candidate.State.name == "drying_rack")
                    .Select(candidate => candidate.SourceKey));
            objectCandidates.RemoveAll(candidate =>
                candidate.State.name == "plate_return_station" &&
                dryingRackKeys.Contains(candidate.SourceKey));
        }

        private void AddAttachStationCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(AttachStation), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                AttachStation station = trackedObjects[i] as AttachStation;
                string name = NormalizeStateObjectName(station);
                if (station == null || string.IsNullOrEmpty(name))
                {
                    continue;
                }

                AddObjectCandidate(objectCandidates, seen, station, name, IsHolderName(name), name == "drying_rack");
            }
        }

        private void AddInteractableCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(Interactable), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                Interactable interactable = trackedObjects[i] as Interactable;
                string name = NormalizeStateObjectName(interactable);
                if (interactable == null || string.IsNullOrEmpty(name))
                {
                    continue;
                }

                AddObjectCandidate(objectCandidates, seen, interactable, name, IsHolderName(name), false);
            }
        }

        private void AddIngredientContainerCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(IngredientContainer), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                IngredientContainer container = trackedObjects[i] as IngredientContainer;
                string name = NormalizeStateObjectName(container);
                if (container == null || string.IsNullOrEmpty(name))
                {
                    continue;
                }

                AddObjectCandidate(objectCandidates, seen, container, name, name == "pot" || name == "plate", false);
            }
        }

        private void AddPlateReturnStationCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(PlateReturnStation), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                PlateReturnStation station = trackedObjects[i] as PlateReturnStation;
                if (station == null)
                {
                    continue;
                }

                AddObjectCandidate(objectCandidates, seen, station, "plate_return_station", true, false);
            }
        }

        private void AddRubbishBinCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(RubbishBin), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                RubbishBin rubbishBin = trackedObjects[i] as RubbishBin;
                if (rubbishBin == null)
                {
                    continue;
                }

                AddObjectCandidate(objectCandidates, seen, rubbishBin, "rubbish_bin", true, false);
            }
        }

        private void AddIngredientCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            GameObject[] ingredients = GetAllIngredients();
            for (int i = 0; i < ingredients.Length; i++)
            {
                GameObject ingredient = ingredients[i];
                string name = NormalizeIngredientName(ingredient);
                if (ingredient == null || string.IsNullOrEmpty(name) || name == "soup")
                {
                    continue;
                }

                ObjectCandidate candidate = AddObjectCandidate(objectCandidates, seen, ingredient, name, false, false);
                if (candidate == null)
                {
                    continue;
                }

                float? progress = InferIngredientProgress(ingredient, name);
                if (progress.HasValue)
                {
                    candidate.State.progress = progress.Value;
                }
            }
        }

        private void AddDirtyPlateStackCandidates(List<ObjectCandidate> objectCandidates, ref int syntheticKey)
        {
            List<ObjectCandidate> parents = objectCandidates
                .Where(candidate => candidate.State.name == "plate_return_station" || candidate.State.name == "sink")
                .ToList();

            for (int i = 0; i < parents.Count; i++)
            {
                int? plateCount = TryGetDirtyPlateCount(parents[i].Source);
                if (!plateCount.HasValue || plateCount.Value <= 0)
                {
                    continue;
                }

                if (
                    parents[i].State.name == "plate_return_station" &&
                    objectCandidates.Any(candidate =>
                        candidate.State.name == "stacked_dirty_plates" &&
                        Distance(candidate.State.position, parents[i].State.position) <= 0.6f))
                {
                    continue;
                }

                objectCandidates.Add(new ObjectCandidate
                {
                    SourceKey = syntheticKey--,
                    Source = parents[i].Source,
                    State = new ObjectStateDto
                    {
                        name = "stacked_dirty_plates",
                        position = [parents[i].State.position[0], parents[i].State.position[1]],
                        plate_count = plateCount.Value
                    }
                });
            }
        }

        private void AddTrackedDirtyPlateStackCandidates(List<ObjectCandidate> objectCandidates, HashSet<string> seen)
        {
            if (!_trackedObjects.TryGetValue(typeof(DirtyPlateStack), out MonoBehaviour[] trackedObjects))
            {
                return;
            }

            for (int i = 0; i < trackedObjects.Length; i++)
            {
                DirtyPlateStack dirtyPlateStack = trackedObjects[i] as DirtyPlateStack;
                if (dirtyPlateStack == null)
                {
                    continue;
                }

                int? plateCount = TryGetDirtyPlateCount(dirtyPlateStack);
                if (!plateCount.HasValue || plateCount.Value <= 0)
                {
                    continue;
                }

                ObjectCandidate candidate = AddObjectCandidate(
                    objectCandidates,
                    seen,
                    dirtyPlateStack,
                    "stacked_dirty_plates",
                    false,
                    false);
                if (candidate == null)
                {
                    continue;
                }

                candidate.State.plate_count = plateCount.Value;
            }
        }

        private ObjectCandidate AddObjectCandidate(
            List<ObjectCandidate> objectCandidates,
            HashSet<string> seen,
            UnityEngine.Object source,
            string name,
            bool isHolder,
            bool allowsMultipleChildren)
        {
            if (source == null)
            {
                return null;
            }

            int sourceKey = GetSourceKey(source);
            string dedupeKey = $"{sourceKey}:{name}";
            if (!seen.Add(dedupeKey))
            {
                return null;
            }

            ObjectCandidate candidate = new ObjectCandidate
            {
                SourceKey = sourceKey,
                Source = source,
                IsHolder = isHolder,
                AllowsMultipleChildren = allowsMultipleChildren,
                State = new ObjectStateDto
                {
                    name = name,
                    position = ToWorldPosition(GetWorldPosition(source))
                }
            };

            if (name == "dispenser")
            {
                string ingredient = GetLayoutObjectIngredient(source as MonoBehaviour);
                if (!string.IsNullOrEmpty(ingredient))
                {
                    candidate.State.ingredient = ingredient;
                    candidate.State.held_object_name = ingredient;
                }
            }

            if (name == "drying_rack")
            {
                candidate.State.plate_ids = [];
                candidate.State.plate_count = 0;
            }

            objectCandidates.Add(candidate);
            return candidate;
        }

        private void AddSoupCandidates(List<ObjectCandidate> objectCandidates, ref int syntheticKey)
        {
            List<int> consumedIngredientKeys = [];
            List<ObjectCandidate> soupCandidates = [];
            List<ObjectCandidate> containers = objectCandidates
                .Where(candidate => candidate.State.name == "pot" || candidate.State.name == "plate")
                .ToList();

            for (int i = 0; i < containers.Count; i++)
            {
                ObjectCandidate container = containers[i];
                IngredientContainer ingredientContainer = container.Source as IngredientContainer;
                if (ingredientContainer == null)
                {
                    continue;
                }

                object assembledNode = TryGetOrderCompositionNode(ingredientContainer);
                List<string> reflectedIngredients = TryGetOrderCompositionIngredientNames(ingredientContainer, assembledNode);
                if (reflectedIngredients.Count == 0)
                {
                    reflectedIngredients = TryGetContainerIngredientNames(ingredientContainer);
                }
                List<ObjectCandidate> nearbyIngredients = FindNearbySoupIngredients(container, objectCandidates);
                int ingredientCount = reflectedIngredients.Count;
                if (ingredientCount == 0)
                {
                    ingredientCount = nearbyIngredients.Count;
                }
                if (ingredientCount == 0)
                {
                    ingredientCount = TryGetContainerIngredientCount(ingredientContainer) ?? 0;
                }

                if (ingredientCount <= 0)
                {
                    continue;
                }

                string cookingState = TryGetContainerCookingState(ingredientContainer, assembledNode) ?? "raw";
                consumedIngredientKeys.AddRange(nearbyIngredients.Select(candidate => candidate.SourceKey));
                soupCandidates.Add(new ObjectCandidate
                {
                    SourceKey = syntheticKey--,
                    Source = ingredientContainer,
                    State = new ObjectStateDto
                    {
                        name = "soup",
                        position = [container.State.position[0], container.State.position[1]],
                        cooking_state = cookingState,
                        ingredients = reflectedIngredients.Count > 0
                            ? BuildSoupIngredientsArray(reflectedIngredients)
                            : BuildSoupIngredientsArray(ingredientCount)
                    }
                });
            }

            if (consumedIngredientKeys.Count > 0)
            {
                objectCandidates.RemoveAll(candidate => consumedIngredientKeys.Contains(candidate.SourceKey));
            }

            objectCandidates.AddRange(soupCandidates);
        }

        private void AssignObjectIds(List<ObjectCandidate> objectCandidates)
        {
            Dictionary<string, int> nameCounts = [];
            List<ObjectCandidate> orderedCandidates = objectCandidates
                .OrderBy(candidate => candidate.State.name)
                .ThenBy(candidate => candidate.State.position[0])
                .ThenBy(candidate => candidate.State.position[1])
                .ThenBy(candidate => candidate.SourceKey)
                .ToList();

            for (int i = 0; i < orderedCandidates.Count; i++)
            {
                ObjectCandidate candidate = orderedCandidates[i];
                int objectIndex = nameCounts.TryGetValue(candidate.State.name, out int currentCount) ? currentCount : 0;
                nameCounts[candidate.State.name] = objectIndex + 1;
                candidate.State.id = $"{candidate.State.name}_{objectIndex}";
            }
        }

        private void ResolveObjectRelationships(List<PlayerCandidate> playerCandidates, List<ObjectCandidate> objectCandidates)
        {
            Dictionary<int, List<ObjectCandidate>> candidatesByKey = objectCandidates
                .GroupBy(candidate => candidate.SourceKey)
                .ToDictionary(group => group.Key, group => group.ToList());

            foreach (KeyValuePair<int, List<ObjectCandidate>> entry in candidatesByKey)
            {
                if (entry.Value.Count <= 1)
                {
                    continue;
                }

                SafeLog($"Shared source key {entry.Key}: {string.Join(", ", entry.Value.Select(candidate => candidate.State.name).ToArray())}");
            }

            for (int i = 0; i < playerCandidates.Count; i++)
            {
                int? heldObjectKey = TryGetHeldObjectKey(playerCandidates[i].Source);
                if (!heldObjectKey.HasValue || !candidatesByKey.TryGetValue(heldObjectKey.Value, out List<ObjectCandidate> heldCandidates))
                {
                    continue;
                }

                ObjectCandidate heldCandidate = SelectHeldObjectCandidate(heldCandidates);
                if (heldCandidate == null)
                {
                    continue;
                }

                AssignParent(heldCandidate, playerCandidates[i].State.id, playerCandidates[i].State.name);
            }

            for (int i = 0; i < objectCandidates.Count; i++)
            {
                ObjectCandidate child = objectCandidates[i];
                if (!string.IsNullOrEmpty(child.State.parent_id) || IsStaticHolderWithoutParent(child))
                {
                    continue;
                }

                string[] allowedParents = GetAllowedParentNames(child.State.name);
                if (allowedParents.Length == 0)
                {
                    continue;
                }

                ObjectCandidate parent = FindClosestParentCandidate(child, objectCandidates, allowedParents);
                if (parent != null)
                {
                    AssignParent(child, parent.State.id, parent.State.name);
                    parent.Children.Add(child);
                    continue;
                }

                if (Array.IndexOf(allowedParents, "player") >= 0)
                {
                    PlayerCandidate player = FindClosestPlayerCandidate(child, playerCandidates);
                    if (player != null)
                    {
                        AssignParent(child, player.State.id, player.State.name);
                    }
                }
            }

            for (int i = 0; i < objectCandidates.Count; i++)
            {
                ObjectCandidate child = objectCandidates[i];
                if (string.IsNullOrEmpty(child.State.parent_id))
                {
                    continue;
                }

                ObjectCandidate parent = objectCandidates.FirstOrDefault(candidate => candidate.State.id == child.State.parent_id);
                if (parent != null && !parent.Children.Contains(child))
                {
                    parent.Children.Add(child);
                }
            }
        }

        private ObjectCandidate SelectHeldObjectCandidate(List<ObjectCandidate> candidates)
        {
            if (candidates == null || candidates.Count == 0)
            {
                return null;
            }

            return candidates
                .OrderBy(candidate => IsStaticHolderWithoutParent(candidate) ? 1 : 0)
                .ThenBy(candidate => candidate.IsHolder ? 1 : 0)
                .ThenBy(candidate => candidate.State.name)
                .FirstOrDefault();
        }

        private void PopulateHolderState(List<ObjectCandidate> objectCandidates)
        {
            for (int i = 0; i < objectCandidates.Count; i++)
            {
                ObjectCandidate holder = objectCandidates[i];
                if (!holder.IsHolder)
                {
                    continue;
                }

                holder.State.held_object_id = null;
                if (holder.State.name != "dispenser")
                {
                    holder.State.held_object_name = null;
                }

                if (holder.State.name == "drying_rack")
                {
                    continue;
                }

                ObjectCandidate child = holder.Children
                    .OrderBy(candidate => Distance(candidate.State.position, holder.State.position))
                    .ThenBy(candidate => candidate.State.id)
                    .FirstOrDefault();
                if (child == null)
                {
                    continue;
                }

                holder.State.held_object_id = child.State.id;
                holder.State.held_object_name = child.State.name;
            }
        }

        private void PopulatePlayerHeldObjects(List<PlayerCandidate> playerCandidates, List<ObjectCandidate> objectCandidates)
        {
            for (int i = 0; i < playerCandidates.Count; i++)
            {
                ObjectCandidate heldObject = objectCandidates.FirstOrDefault(candidate => candidate.State.parent_id == playerCandidates[i].State.id);
                if (heldObject == null)
                {
                    heldObject = InferHeldObjectFromProximity(playerCandidates[i], objectCandidates);
                    if (heldObject != null)
                    {
                        AssignParent(heldObject, playerCandidates[i].State.id, playerCandidates[i].State.name);
                    }
                }

                if (heldObject == null)
                {
                    continue;
                }

                playerCandidates[i].State.held_object_id = heldObject.State.id;
                playerCandidates[i].State.held_object_name = heldObject.State.name;
            }
        }

        private ObjectCandidate InferHeldObjectFromProximity(PlayerCandidate player, List<ObjectCandidate> objectCandidates)
        {
            return objectCandidates
                .Where(candidate => string.IsNullOrEmpty(candidate.State.parent_id))
                .Where(candidate => IsCarryableStateObject(candidate.State.name))
                .Where(candidate => Distance(candidate.State.position, player.State.position) <= 0.6f)
                .OrderBy(candidate => Distance(candidate.State.position, player.State.position))
                .FirstOrDefault();
        }

        private void PopulateRackState(List<ObjectCandidate> objectCandidates)
        {
            List<ObjectCandidate> dryingRacks = objectCandidates
                .Where(candidate => candidate.State.name == "drying_rack")
                .ToList();

            for (int i = 0; i < dryingRacks.Count; i++)
            {
                List<string> plateIds = dryingRacks[i].Children
                    .Where(candidate => candidate.State.name == "plate")
                    .OrderBy(candidate => candidate.State.id)
                    .Select(candidate => candidate.State.id)
                    .ToList();
                dryingRacks[i].State.plate_ids = plateIds.ToArray();
                dryingRacks[i].State.plate_count = plateIds.Count;
            }
        }

        private bool IsStaticHolderWithoutParent(ObjectCandidate candidate)
        {
            return candidate.State.name == "tabletop" ||
                candidate.State.name == "dispenser" ||
                candidate.State.name == "chopping_board" ||
                candidate.State.name == "stove" ||
                candidate.State.name == "delivery_station" ||
                candidate.State.name == "plate_return_station" ||
                candidate.State.name == "drying_rack" ||
                candidate.State.name == "sink" ||
                candidate.State.name == "rubbish_bin";
        }

        private string NormalizeStateObjectName(MonoBehaviour obj)
        {
            if (obj == null)
            {
                return null;
            }

            string layoutName = ClassifyLayoutObject(obj);
            if (layoutName == "attach_station" || layoutName == "interactable" || layoutName == "ingredient_container" || layoutName == "unknown")
            {
                return null;
            }

            if (layoutName == "plate_return")
            {
                return null;
            }

            if (obj is Interactable && layoutName == "chopping_board")
            {
                return null;
            }

            return layoutName;
        }

        private string NormalizeIngredientName(GameObject ingredient)
        {
            if (ingredient == null)
            {
                return null;
            }

            return NormalizeIngredientNameFromRawName(ingredient.name);
        }

        private string NormalizeIngredientNameFromRawName(string name)
        {
            string normalizedName = CleanObjectName(name).ToLowerInvariant();
            if (normalizedName.Contains("onion"))
            {
                return "onion";
            }

            if (normalizedName.Contains("tomato"))
            {
                return "tomato";
            }

            if (normalizedName.Contains("mushroom"))
            {
                return "mushroom";
            }

            if (normalizedName.Contains("lettuce"))
            {
                return "lettuce";
            }

            if (normalizedName.Contains("meat"))
            {
                return "meat";
            }

            if (normalizedName.Contains("bun"))
            {
                return "bun";
            }

            if (normalizedName.Contains("soup"))
            {
                return "soup";
            }

            return null;
        }

        private float? InferIngredientProgress(GameObject ingredient, string ingredientName)
        {
            if (ingredient == null)
            {
                return null;
            }

            if (ingredientName == "onion" || ingredientName == "tomato" || ingredientName == "mushroom" || ingredientName == "lettuce")
            {
                return string.Equals(ingredient.tag, "Ingredient", StringComparison.OrdinalIgnoreCase) ? 1.0f : 0.0f;
            }

            return null;
        }

        private List<ObjectCandidate> FindNearbySoupIngredients(ObjectCandidate container, List<ObjectCandidate> objectCandidates)
        {
            return objectCandidates
                .Where(candidate => candidate.State.name == "onion" && string.IsNullOrEmpty(candidate.State.parent_id))
                .Where(candidate => Distance(candidate.State.position, container.State.position) <= 0.45f)
                .OrderBy(candidate => Distance(candidate.State.position, container.State.position))
                .Take(3)
                .ToList();
        }

        private string[] BuildSoupIngredientsArray(int ingredientCount)
        {
            ingredientCount = Mathf.Clamp(ingredientCount, 0, 3);
            string[] ingredients = [null, null, null];
            for (int i = 0; i < ingredientCount; i++)
            {
                ingredients[i] = "onion";
            }

            return ingredients;
        }

        private string[] BuildSoupIngredientsArray(List<string> ingredientNames)
        {
            string[] ingredients = [null, null, null];
            for (int i = 0; i < ingredientNames.Count && i < ingredients.Length; i++)
            {
                ingredients[i] = ingredientNames[i];
            }

            return ingredients;
        }

        private string[] GetAllowedParentNames(string objectName)
        {
            switch (objectName)
            {
                case "pot":
                    return ["stove", "tabletop", "player"];
                case "plate":
                    return ["drying_rack", "tabletop", "delivery_station", "player"];
                case "onion":
                case "tomato":
                case "mushroom":
                case "lettuce":
                case "meat":
                case "bun":
                    return ["chopping_board", "tabletop", "player"];
                case "fire_extinguisher":
                    return ["tabletop", "player"];
                case "stacked_dirty_plates":
                    return ["sink", "plate_return_station", "player"];
                case "soup":
                    return ["plate", "pot"];
                default:
                    return [];
            }
        }

        private ObjectCandidate FindClosestParentCandidate(ObjectCandidate child, List<ObjectCandidate> objectCandidates, string[] allowedParents)
        {
            ObjectCandidate bestCandidate = null;
            float bestDistance = float.MaxValue;

            for (int i = 0; i < allowedParents.Length; i++)
            {
                if (allowedParents[i] == "player")
                {
                    continue;
                }

                List<ObjectCandidate> parentCandidates = objectCandidates
                    .Where(candidate => candidate.State.name == allowedParents[i])
                    .ToList();

                for (int j = 0; j < parentCandidates.Count; j++)
                {
                    ObjectCandidate parent = parentCandidates[j];
                    if (!parent.AllowsMultipleChildren && parent.Children.Count > 0)
                    {
                        continue;
                    }

                    float distance = Distance(child.State.position, parent.State.position);
                    if (distance > GetParentDistanceThreshold(child.State.name, parent.State.name))
                    {
                        continue;
                    }

                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestCandidate = parent;
                    }
                }

                if (bestCandidate != null)
                {
                    return bestCandidate;
                }
            }

            return null;
        }

        private PlayerCandidate FindClosestPlayerCandidate(ObjectCandidate child, List<PlayerCandidate> playerCandidates)
        {
            PlayerCandidate bestPlayer = null;
            float bestDistance = 0.9f;
            for (int i = 0; i < playerCandidates.Count; i++)
            {
                float distance = Distance(child.State.position, playerCandidates[i].State.position);
                if (distance <= bestDistance)
                {
                    bestDistance = distance;
                    bestPlayer = playerCandidates[i];
                }
            }

            return bestPlayer;
        }

        private float GetParentDistanceThreshold(string childName, string parentName)
        {
            if (childName == "soup")
            {
                return 0.3f;
            }

            if (parentName == "drying_rack")
            {
                return 0.7f;
            }

            return 0.55f;
        }

        private void AssignParent(ObjectCandidate child, string parentId, string parentName)
        {
            child.State.parent_id = parentId;
            child.State.parent_name = parentName;
        }

        private bool IsHolderName(string name)
        {
            return name == "tabletop" ||
                name == "dispenser" ||
                name == "chopping_board" ||
                name == "stove" ||
                name == "delivery_station" ||
                name == "plate_return_station" ||
                name == "drying_rack" ||
                name == "sink" ||
                name == "rubbish_bin" ||
                name == "pot" ||
                name == "plate";
        }

        private bool IsCarryableStateObject(string name)
        {
            return name == "plate" ||
                name == "pot" ||
                name == "onion" ||
                name == "tomato" ||
                name == "mushroom" ||
                name == "lettuce" ||
                name == "meat" ||
                name == "bun" ||
                name == "fire_extinguisher" ||
                name == "stacked_dirty_plates";
        }

        private int GetSourceKey(UnityEngine.Object source)
        {
            if (source == null)
            {
                return 0;
            }

            if (source is Component component)
            {
                return component.gameObject.GetInstanceID();
            }

            return source.GetInstanceID();
        }

        private Vector3 GetWorldPosition(UnityEngine.Object source)
        {
            if (source is Component component)
            {
                return component.transform.position;
            }

            if (source is GameObject gameObject)
            {
                return gameObject.transform.position;
            }

            return Vector3.zero;
        }

        private float[] ToWorldPosition(Vector3 position)
        {
            return [RoundCoordinate(position.x), RoundCoordinate(position.z)];
        }

        private float[] ToWorldDirection(Vector3 direction)
        {
            Vector2 directionXZ = new Vector2(direction.x, direction.z);
            if (directionXZ.sqrMagnitude <= 0.0001f)
            {
                return [0f, 0f];
            }

            directionXZ.Normalize();
            return [RoundCoordinate(directionXZ.x), RoundCoordinate(directionXZ.y)];
        }

        private float RoundCoordinate(float value)
        {
            return (float)Math.Round(value, 3);
        }

        private float Distance(float[] positionA, float[] positionB)
        {
            float deltaX = positionA[0] - positionB[0];
            float deltaY = positionA[1] - positionB[1];
            return Mathf.Sqrt((deltaX * deltaX) + (deltaY * deltaY));
        }

        private string CleanObjectName(string name)
        {
            if (string.IsNullOrEmpty(name))
            {
                return string.Empty;
            }

            int cloneIndex = name.IndexOf("(Clone)", StringComparison.OrdinalIgnoreCase);
            if (cloneIndex >= 0)
            {
                name = name.Substring(0, cloneIndex);
            }

            return name.Trim();
        }

        private int? TryGetPlayerIndex(PlayerControls player)
        {
            object playerIdProvider = TryGetMemberValue(player, "m_playerIDProvider", "PlayerIDProvider");
            object playerId = TryInvokeMethod(playerIdProvider, "GetID");
            int? index = ConvertObjectToInt(playerId);
            if (index.HasValue)
            {
                return index.Value;
            }

            index = TryGetIntMember(player,
                "m_playerIndex",
                "m_playerID",
                "m_iPlayerID",
                "m_controllerIndex",
                "m_rewiredPlayerID");
            if (!index.HasValue)
            {
                index = TryGetIntMember(playerIdProvider,
                    "m_player",
                    "m_playerIndex",
                    "m_playerID",
                    "m_iPlayerID",
                    "m_controllerIndex",
                    "m_rewiredPlayerID",
                    "PlayerIndex",
                    "PlayerID",
                    "ID",
                    "Id");
            }
            if (!index.HasValue)
            {
                WarnMissingReflectionData("PlayerControls.PlayerIndex", player);
            }

            return index;
        }

        private int? TryGetHeldObjectKey(PlayerControls player)
        {
            object carrier = TryGetMemberValue(player, "m_carrier", "Carrier");
            UnityEngine.Object heldObject = ConvertToUnityObject(TryInvokeMethod(carrier, "InspectCarriedItem"));
            if (heldObject != null)
            {
                return GetSourceKey(heldObject);
            }

            if (!HasAnyMember(player,
                "m_heldItem",
                "m_carriedItem",
                "m_pickedUpItem",
                "m_heldObject",
                "m_itemInHands",
                "HeldItem",
                "HeldObject") && !HasAnyMember(carrier, "InspectCarriedItem"))
            {
                WarnMissingReflectionData("PlayerControls.HeldObject", player);
            }

            heldObject = TryGetUnityObjectMember(player,
                "m_heldItem",
                "m_carriedItem",
                "m_pickedUpItem",
                "m_heldObject",
                "m_itemInHands",
                "HeldItem",
                "HeldObject");
            if (heldObject == null)
            {
                heldObject = TryGetNestedUnityObjectMember(
                    carrier,
                    new[]
                    {
                        "m_heldItem",
                        "m_carriedItem",
                        "m_pickedUpItem",
                        "m_heldObject",
                        "m_itemInHands",
                        "HeldItem",
                        "HeldObject",
                        "m_attachedObject",
                        "AttachedObject",
                        "m_carriedObject",
                        "CarriedObject",
                        "m_item",
                        "Item",
                    },
                    new[]
                    {
                        "m_attachment",
                        "Attachment",
                        "m_physicalAttachment",
                        "PhysicalAttachment",
                        "m_attachedItem",
                        "AttachedItem",
                    });
                if (heldObject == null)
                {
                    heldObject = TryFindStateLikeUnityObject(carrier, GetSourceKey(player));
                }
                if (heldObject == null)
                {
                    return null;
                }
            }

            return GetSourceKey(heldObject);
        }

        private int? TryGetDirtyPlateCount(UnityEngine.Object source)
        {
            if (!HasAnyMember(source,
                "m_plateCount",
                "m_dirtyPlateCount",
                "m_numPlates",
                "PlateCount",
                "DirtyPlateCount"))
            {
                WarnMissingReflectionData(source.GetType().Name + ".DirtyPlateCount", source);
            }

            int? count = TryGetIntMember(source,
                "m_contents",
                "m_plateCount",
                "m_dirtyPlateCount",
                "m_numPlates",
                "PlateCount",
                "DirtyPlateCount");
            if (count.HasValue)
            {
                return count;
            }

            object stack = TryGetMemberValue(source, "m_stack", "Stack");
            count = TryGetStackLikeCount(stack, true);
            if (count.HasValue)
            {
                return count;
            }

            object attachStation = TryGetMemberValue(source, "m_attachStation", "AttachStation");
            count = TryGetStackLikeCount(attachStation, false);
            if (count.HasValue)
            {
                return count;
            }

            if (source is Component component)
            {
                Component[] siblingComponents = component.GetComponents<Component>();
                for (int i = 0; i < siblingComponents.Length; i++)
                {
                    Component sibling = siblingComponents[i];
                    if (sibling == null || ReferenceEquals(sibling, source))
                    {
                        continue;
                    }

                    count = TryGetIntMember(sibling, "m_contents");
                    if (count.HasValue)
                    {
                        return count;
                    }

                    count = TryGetStackLikeCount(TryGetMemberValue(sibling, "m_stack", "Stack"), true);
                    if (count.HasValue)
                    {
                        return count;
                    }

                    count = TryGetStackLikeCount(TryGetMemberValue(sibling, "m_attachStation", "AttachStation"), false);
                    if (count.HasValue)
                    {
                        return count;
                    }
                }
            }

            return null;
        }

        private int? TryGetContainerIngredientCount(IngredientContainer container)
        {
            bool hasCountMember = HasAnyMember(container,
                "m_ingredientCount",
                "m_numIngredients",
                "IngredientCount",
                "NumIngredients");
            bool hasCollectionMember = HasAnyMember(container,
                "m_ingredients",
                "m_contents",
                "m_items",
                "Ingredients",
                "Contents");
            if (!hasCountMember && !hasCollectionMember)
            {
                WarnMissingReflectionData("IngredientContainer.Ingredients", container);
            }

            int? count = TryGetIntMember(container,
                "m_ingredientCount",
                "m_numIngredients",
                "IngredientCount",
                "NumIngredients");
            if (count.HasValue)
            {
                return Mathf.Clamp(count.Value, 0, 3);
            }

            object collection = TryGetMemberValue(container,
                "m_ingredients",
                "m_contents",
                "m_items",
                "Ingredients",
                "Contents");
            if (collection is IEnumerable enumerable)
            {
                int ingredientCount = 0;
                foreach (object entry in enumerable)
                {
                    string ingredientName = NormalizeUnknownIngredientName(entry);
                    if (!string.IsNullOrEmpty(ingredientName) && ingredientName != "soup")
                    {
                        ingredientCount++;
                    }
                }

                return Mathf.Clamp(ingredientCount, 0, 3);
            }

            return null;
        }

        private List<string> TryGetContainerIngredientNames(IngredientContainer container)
        {
            List<string> ingredientNames = [];
            object collection = TryGetMemberValue(container,
                "m_ingredients",
                "m_contents",
                "m_items",
                "Ingredients",
                "Contents");
            if (collection is not IEnumerable enumerable)
            {
                return ingredientNames;
            }

            foreach (object entry in enumerable)
            {
                string ingredientName = NormalizeUnknownIngredientName(entry);
                if (!string.IsNullOrEmpty(ingredientName) && ingredientName != "soup")
                {
                    ingredientNames.Add(ingredientName);
                }
            }

            return ingredientNames;
        }

        private List<string> TryGetOrderCompositionIngredientNames(IngredientContainer container, object assembledNode = null)
        {
            List<string> ingredientNames = [];
            assembledNode ??= TryGetOrderCompositionNode(container);
            ExtractIngredientNamesFromAssembledNode(assembledNode, ingredientNames);
            return ingredientNames;
        }

        private object TryGetOrderCompositionNode(IngredientContainer container)
        {
            object orderDefinition = TryGetOrderDefinitionComponent(container);
            if (orderDefinition == null)
            {
                return null;
            }

            return TryInvokeMethod(orderDefinition, "GetOrderComposition");
        }

        private string TryGetContainerCookingState(IngredientContainer container, object assembledNode = null)
        {
            assembledNode ??= TryGetOrderCompositionNode(container);
            string cookingState = NormalizeCookingState(TryGetMemberValue(assembledNode, "m_progress"));
            if (!string.IsNullOrEmpty(cookingState))
            {
                return cookingState;
            }

            object cookingHandler = TryGetSiblingComponent(container, "CookingHandler");
            cookingState = NormalizeCookingState(TryInvokeMethod(cookingHandler, "GetCookedOrderState"));
            if (!string.IsNullOrEmpty(cookingState))
            {
                return cookingState;
            }

            return NormalizeCookingStateFromProgress(TryGetContainerProgress(container));
        }

        private float? TryGetContainerProgress(IngredientContainer container)
        {
            object cookingHandler = TryGetSiblingComponent(container, "CookingHandler");
            if (cookingHandler != null)
            {
                float? cookingProgress = ConvertObjectToFloat(TryInvokeMethod(cookingHandler, "GetCookingProgress"));
                float? cookingTime = ConvertObjectToFloat(TryGetMemberValue(cookingHandler, "AccessCookingTime")) ?? ConvertObjectToFloat(TryInvokeMethod(cookingHandler, "get_AccessCookingTime"));
                if (cookingProgress.HasValue && cookingTime.HasValue && cookingTime.Value > 0f)
                {
                    return Mathf.Max(cookingProgress.Value / cookingTime.Value, 0f);
                }
            }

            if (!HasAnyMember(container,
                "m_cookProgress",
                "m_cookingProgress",
                "m_progress",
                "CookProgress",
                "Progress"))
            {
                WarnMissingReflectionData("IngredientContainer.Progress", container);
            }

            float? progress = TryGetFloatMember(container,
                "m_cookProgress",
                "m_cookingProgress",
                "m_progress",
                "CookProgress",
                "Progress");
            if (progress.HasValue)
            {
                return NormalizeProgress(progress.Value);
            }

            return null;
        }

        private int? TryGetStackLikeCount(object instance, bool assumeOneIfPresent)
        {
            if (instance == null)
            {
                return null;
            }

            int? count = ConvertObjectToInt(TryInvokeMethod(instance, "GetSize"));
            if (count.HasValue)
            {
                return count;
            }

            count = TryGetIntMember(instance,
                "m_contents",
                "m_plateCount",
                "m_dirtyPlateCount",
                "m_numPlates",
                "m_count",
                "m_itemCount",
                "m_numItems",
                "PlateCount",
                "DirtyPlateCount",
                "Count",
                "ItemCount",
                "NumItems");
            if (count.HasValue)
            {
                return count;
            }

            object collection = TryGetMemberValue(instance,
                "m_contents",
                "m_items",
                "m_stackItems",
                "m_plates",
                "Contents",
                "Items");
            if (collection is IEnumerable enumerable)
            {
                int itemCount = 0;
                foreach (object entry in enumerable)
                {
                    if (entry != null)
                    {
                        itemCount++;
                    }
                }

                return itemCount;
            }

            object nestedStack = TryGetMemberValue(instance, "m_stack", "Stack");
            if (!ReferenceEquals(nestedStack, instance))
            {
                count = TryGetStackLikeCount(nestedStack, assumeOneIfPresent);
                if (count.HasValue)
                {
                    return count;
                }
            }

            object nestedAttachStation = TryGetMemberValue(instance, "m_attachStation", "AttachStation");
            if (!ReferenceEquals(nestedAttachStation, instance))
            {
                count = TryGetStackLikeCount(nestedAttachStation, false);
                if (count.HasValue)
                {
                    return count;
                }
            }

            if (assumeOneIfPresent && instance is UnityEngine.Object unityObject)
            {
                SafeLog($"Stack-like object with unknown count: {unityObject.GetType().FullName} '{unityObject.name}'");
                return 1;
            }

            return null;
        }

        private UnityEngine.Object TryGetNestedUnityObjectMember(object instance, string[] directMemberNames, string[] containerMemberNames)
        {
            if (instance == null)
            {
                return null;
            }

            UnityEngine.Object directObject = TryGetUnityObjectMember(instance, directMemberNames);
            if (directObject != null)
            {
                return directObject;
            }

            for (int i = 0; i < containerMemberNames.Length; i++)
            {
                object container = TryGetMemberValue(instance, containerMemberNames[i]);
                if (container == null)
                {
                    continue;
                }

                directObject = TryGetUnityObjectMember(container, directMemberNames);
                if (directObject != null)
                {
                    return directObject;
                }
            }

            return null;
        }

        private UnityEngine.Object TryFindStateLikeUnityObject(object instance, int excludedKey)
        {
            if (instance == null)
            {
                return null;
            }

            Type instanceType = instance.GetType();
            IEnumerable<MemberInfo> members = instanceType
                .GetMembers(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(member => member.MemberType == MemberTypes.Field || member.MemberType == MemberTypes.Property);

            foreach (MemberInfo member in members)
            {
                object value = TryGetMemberValue(instance, member.Name);
                if (value is Component component)
                {
                    int key = GetSourceKey(component);
                    if (key != excludedKey && IsStateLikeName(component.name))
                    {
                        return component;
                    }
                }
                else if (value is GameObject gameObject)
                {
                    int key = GetSourceKey(gameObject);
                    if (key != excludedKey && IsStateLikeName(gameObject.name))
                    {
                        return gameObject;
                    }
                }
            }

            return null;
        }

        private bool IsStateLikeName(string name)
        {
            string normalizedName = NormalizeIngredientNameFromRawName(name);
            if (!string.IsNullOrEmpty(normalizedName))
            {
                return true;
            }

            string cleaned = CleanObjectName(name).ToLowerInvariant();
            return cleaned.Contains("pot") ||
                cleaned.Contains("plate") ||
                cleaned.Contains("extinguisher");
        }

        private bool HasAnyMember(object instance, params string[] memberNames)
        {
            if (instance == null)
            {
                return false;
            }

            Type instanceType = instance.GetType();
            for (int i = 0; i < memberNames.Length; i++)
            {
                if (instanceType.GetField(memberNames[i], BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic) != null)
                {
                    return true;
                }

                if (instanceType.GetProperty(memberNames[i], BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic) != null)
                {
                    return true;
                }
            }

            return false;
        }

        private object TryGetOrderDefinitionComponent(IngredientContainer container)
        {
            if (container == null)
            {
                return null;
            }

            Component[] siblings = container.gameObject.GetComponents<Component>();
            for (int i = 0; i < siblings.Length; i++)
            {
                Component sibling = siblings[i];
                if (sibling == null)
                {
                    continue;
                }

                if (TryGetParameterlessMethod(sibling.GetType(), "GetOrderComposition") != null)
                {
                    return sibling;
                }
            }

            return null;
        }

        private object TryGetSiblingComponent(Component component, string typeName)
        {
            if (component == null || string.IsNullOrEmpty(typeName))
            {
                return null;
            }

            Component[] siblings = component.gameObject.GetComponents<Component>();
            for (int i = 0; i < siblings.Length; i++)
            {
                Component sibling = siblings[i];
                if (sibling != null && string.Equals(sibling.GetType().Name, typeName, StringComparison.Ordinal))
                {
                    return sibling;
                }
            }

            return null;
        }

        private void ExtractIngredientNamesFromAssembledNode(object node, List<string> ingredientNames)
        {
            if (node == null || ingredientNames == null)
            {
                return;
            }

            object ingredientOrderNode = TryGetMemberValue(node, "m_ingriedientOrderNode", "m_ingredientOrderNode");
            if (ingredientOrderNode != null)
            {
                string ingredientName = NormalizeUnknownIngredientName(ingredientOrderNode);
                if (!string.IsNullOrEmpty(ingredientName))
                {
                    ingredientNames.Add(ingredientName);
                }
                return;
            }

            object composition = TryGetMemberValue(node, "m_composition");
            if (composition is IEnumerable enumerable)
            {
                foreach (object child in enumerable)
                {
                    ExtractIngredientNamesFromAssembledNode(child, ingredientNames);
                }
            }
        }

        private string NormalizeUnknownIngredientName(object value)
        {
            if (value == null)
            {
                return null;
            }

            if (value is GameObject gameObject)
            {
                return NormalizeIngredientName(gameObject);
            }

            if (value is Component component)
            {
                return NormalizeIngredientName(component.gameObject);
            }

            if (value is UnityEngine.Object unityObject)
            {
                return NormalizeIngredientNameFromRawName(unityObject.name);
            }

            object ingredientOrderNode = TryGetMemberValue(value, "m_ingriedientOrderNode", "m_ingredientOrderNode");
            if (ingredientOrderNode is UnityEngine.Object ingredientUnityObject)
            {
                return NormalizeIngredientNameFromRawName(ingredientUnityObject.name);
            }

            return value.ToString();
        }

        private float? NormalizeProgress(float rawProgress)
        {
            if (rawProgress < 0.0f)
            {
                return null;
            }

            if (rawProgress <= 2.001f)
            {
                return rawProgress;
            }

            if (rawProgress <= 200.001f)
            {
                return rawProgress / 100.0f;
            }

            return null;
        }

        private string NormalizeCookingState(object value)
        {
            if (value == null)
            {
                return null;
            }

            if (value is Enum enumValue)
            {
                return enumValue.ToString().ToLowerInvariant();
            }

            if (value is string text)
            {
                string normalized = text.Trim();
                return string.IsNullOrEmpty(normalized) ? null : normalized.ToLowerInvariant();
            }

            int? numericState = ConvertObjectToInt(value);
            if (!numericState.HasValue)
            {
                return null;
            }

            return numericState.Value switch
            {
                0 => "raw",
                1 => "cooked",
                2 => "burnt",
                _ => null,
            };
        }

        private string NormalizeCookingStateFromProgress(float? progress)
        {
            if (!progress.HasValue)
            {
                return null;
            }

            if (progress.Value >= 2f)
            {
                return "burnt";
            }

            if (progress.Value >= 1f)
            {
                return "cooked";
            }

            return "raw";
        }

        private int? TryGetIntMember(object instance, params string[] memberNames)
        {
            object value = TryGetMemberValue(instance, memberNames);
            if (value == null)
            {
                return null;
            }

            if (value is int intValue)
            {
                return intValue;
            }

            if (value is short shortValue)
            {
                return shortValue;
            }

            if (value is byte byteValue)
            {
                return byteValue;
            }

            if (value is Enum enumValue)
            {
                return Convert.ToInt32(enumValue, CultureInfo.InvariantCulture);
            }

            return null;
        }

        private float? TryGetFloatMember(object instance, params string[] memberNames)
        {
            object value = TryGetMemberValue(instance, memberNames);
            if (value == null)
            {
                return null;
            }

            if (value is float floatValue)
            {
                return floatValue;
            }

            if (value is double doubleValue)
            {
                return (float)doubleValue;
            }

            if (value is int intValue)
            {
                return intValue;
            }

            return null;
        }

        private UnityEngine.Object TryGetUnityObjectMember(object instance, params string[] memberNames)
        {
            object value = TryGetMemberValue(instance, memberNames);
            if (value is UnityEngine.Object unityObject)
            {
                return unityObject;
            }

            return null;
        }

        private object TryGetMemberValue(object instance, params string[] memberNames)
        {
            if (instance == null)
            {
                return null;
            }

            Type instanceType = instance.GetType();
            for (int i = 0; i < memberNames.Length; i++)
            {
                FieldInfo field = instanceType.GetField(memberNames[i], BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (field != null)
                {
                    return field.GetValue(instance);
                }

                PropertyInfo property = instanceType.GetProperty(memberNames[i], BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (property != null && property.CanRead)
                {
                    try
                    {
                        return property.GetValue(instance, null);
                    }
                    catch (Exception ex)
                    {
                        SafeLog($"Property getter exception for {instanceType.FullName}.{memberNames[i]}: {ex}");
                    }
                }
            }

            return null;
        }

        private void WarnMissingReflectionData(string key, object instance)
        {
            if (instance == null || !_reflectionWarnings.Add(key))
            {
                return;
            }

            Type instanceType = instance.GetType();
            string fieldSummary = string.Join(", ",
                instanceType
                    .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                    .Select(field => $"{field.FieldType.Name} {field.Name}")
                    .ToArray());
            string propertySummary = string.Join(", ",
                instanceType
                    .GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                    .Where(property => property.CanRead)
                    .Select(property => $"{property.PropertyType.Name} {property.Name}")
                    .ToArray());

            Logger.LogInfo($"ExtractionMod reflection gap '{key}' on type '{instanceType.FullName}'. Fields: [{fieldSummary}] Properties: [{propertySummary}]");
            SafeLog($"Reflection gap {key} on {instanceType.FullName}. Fields: [{fieldSummary}] Properties: [{propertySummary}]");
        }

        private object TryInvokeMethod(object instance, string methodName)
        {
            if (instance == null || string.IsNullOrEmpty(methodName))
            {
                return null;
            }

            try
            {
                MethodInfo method = TryGetParameterlessMethod(instance.GetType(), methodName);
                if (method == null)
                {
                    return null;
                }

                return method.Invoke(instance, null);
            }
            catch (Exception ex)
            {
                SafeLog($"Method invoke exception for {instance.GetType().FullName}.{methodName}: {ex}");
                return null;
            }
        }

        private MethodInfo TryGetParameterlessMethod(Type instanceType, string methodName)
        {
            if (instanceType == null || string.IsNullOrEmpty(methodName))
            {
                return null;
            }

            MethodInfo[] methods = instanceType.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            MethodInfo bestMatch = null;
            int bestDepth = int.MinValue;
            for (int i = 0; i < methods.Length; i++)
            {
                MethodInfo method = methods[i];
                if (!string.Equals(method.Name, methodName, StringComparison.Ordinal) ||
                    method.ContainsGenericParameters ||
                    method.GetParameters().Length != 0)
                {
                    continue;
                }

                int declaringDepth = GetTypeInheritanceDepth(method.DeclaringType);
                if (bestMatch == null || declaringDepth > bestDepth)
                {
                    bestMatch = method;
                    bestDepth = declaringDepth;
                }
            }

            return bestMatch;
        }

        private int GetTypeInheritanceDepth(Type type)
        {
            int depth = 0;
            while (type != null)
            {
                depth++;
                type = type.BaseType;
            }

            return depth;
        }

        private UnityEngine.Object ConvertToUnityObject(object value)
        {
            return value as UnityEngine.Object;
        }

        private int? ConvertObjectToInt(object value)
        {
            if (value == null)
            {
                return null;
            }

            if (value is int intValue)
            {
                return intValue;
            }

            if (value is short shortValue)
            {
                return shortValue;
            }

            if (value is byte byteValue)
            {
                return byteValue;
            }

            if (value is Enum enumValue)
            {
                return Convert.ToInt32(enumValue, CultureInfo.InvariantCulture);
            }

            return null;
        }

        private float? ConvertObjectToFloat(object value)
        {
            if (value == null)
            {
                return null;
            }

            if (value is float floatValue)
            {
                return floatValue;
            }

            if (value is double doubleValue)
            {
                return (float)doubleValue;
            }

            if (value is int intValue)
            {
                return intValue;
            }

            return null;
        }

        private void SafeLog(string message)
        {
            try
            {
                if (string.IsNullOrEmpty(_debugLogPath))
                {
                    _debugLogPath = Path.Combine(Paths.GameRootPath, "extraction_mod_debug.log");
                }

                string line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {message}{Environment.NewLine}";
                File.AppendAllText(_debugLogPath, line);
            }
            catch
            {
            }
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
            RefreshObjects<DirtyPlateStack>(); // Carried and returned dirty plate stacks
            RefreshObjects<PlayerControls>(); // Players
            RefreshObjects<PickupItemSpawner>(); // Dispenser crates
            RefreshObjects<AttachStation>(); // TableTops DryingPart (clean plates), PlateStation, and chopping boards.
            RefreshObjects<Interactable>(); // Fire extinguisher, WashPart (sink), and chopping boards.
            RefreshObjects<RubbishBin>(); // Rubbish bins, they're British!
            RefreshOrderControllers();
            LogPickupItemSpawnerDetails();
            int ingredientCount = GetAllIngredients().Length;
            Logger.LogInfo($"ExtractionMod: Found {ingredientCount} tagged ingredient GameObjects.");
            SafeLog(
                "RefreshAllObjects " +
                $"IngredientContainer={GetTrackedCount(typeof(IngredientContainer))} " +
                $"PlateReturnStation={GetTrackedCount(typeof(PlateReturnStation))} " +
                $"DirtyPlateStack={GetTrackedCount(typeof(DirtyPlateStack))} " +
                $"PlayerControls={GetTrackedCount(typeof(PlayerControls))} " +
                $"PickupItemSpawner={GetTrackedCount(typeof(PickupItemSpawner))} " +
                $"AttachStation={GetTrackedCount(typeof(AttachStation))} " +
                $"Interactable={GetTrackedCount(typeof(Interactable))} " +
                $"RubbishBin={GetTrackedCount(typeof(RubbishBin))} " +
                $"Ingredients={ingredientCount}");

            // RefreshObjects<PlateStation>(); // Delivery station. Included in AttachStation.
            // RefreshObjects<Workstation>(); // Cutting board. Included in AttachStation.
            // RefreshObjects<PlacementContainer>(); // The same as IngredientContainer.
            // RefreshObjects<CookingStation>(); // Stove. Included in AttachStation.
            // RefreshObjects<PickupItemSpawner>(); // Dispenser crates. Included in AttacheStation.
        }

        private int GetTrackedCount(Type type)
        {
            if (type == null || !_trackedObjects.TryGetValue(type, out MonoBehaviour[] trackedObjects) || trackedObjects == null)
            {
                return 0;
            }

            return trackedObjects.Length;
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
