using System.Collections.Generic;
using UnityEngine;
using System.IO;

public static class JsonGenerator{

    public static string GenerateGameJson(List<Player> players, List<Element> objects, List<Order> orders, string[,] layoutGrid,
                                          string layoutName, float timeLeft, float timeElapsed, int gameloop )
    {
        //Prepare and Stringify the GameState
        GameState internalState = new()
        {
            players = players,
            objects = objects,
            all_orders = orders,
            timestep = gameloop-1
        };
        string stateString = ConvertGameStateToString(internalState);

        string layoutString = ConvertLayoutToString(layoutGrid);

        GameRecord record = new()
        {
            state = stateString,
            time_left = timeLeft,
            score = 0,
            time_elapsed = timeElapsed,
            cur_gameloop = gameloop,
            layout = layoutString,
            layout_name = layoutName,
            player_0_is_human = true,
            player_1_is_human = true
        };

        string json = JsonUtility.ToJson(record, true);
        File.WriteAllText("state.json", json); //TODO: Change this to the desired output path
        return stateString;
    }

    public static void GenerateLevelJson(List<LayoutObject> layoutObjects, List<float[]> playerStartPositions) {
        string layoutString = ConvertLevelLayoutToString(layoutObjects);
        string playerStartsString = ConvertPlayerStartPositionsToString(playerStartPositions);
        string json = "{\"layout\": " + layoutString + ", \"player_starts\": " + playerStartsString + "}";
        File.WriteAllText("level.json", json);
    }

    private static string ConvertLayoutToString(string[,] grid) {
        List<string> rows = [];
        for (int i = 0; i < grid.GetLength(0); i++) {
            List<string> row = [];
            for (int j = 0; j < grid.GetLength(1); j++) {
                row.Add(grid[i, j]);
            }
            rows.Add("[\"" + string.Join("\", \"", [..row]) + "\"]");
        }
        return "[" + string.Join(", ", [.. rows]) + "]";
    }

    private static string ConvertGameStateToString(GameState gs) {
        // 1. Serialize Players
        List<string> playerEntries = [];
        foreach (var p in gs.players) {
            playerEntries.Add($"{{\"position\": [{p.position[0]}, {p.position[1]}], \"orientation\": [{p.orientation[0]}, {p.orientation[1]}], \"held_object\": \"{p.held_object ?? ""}\"}}");
        }
        string playersJson = "[" + string.Join(", ", [..playerEntries]) + "]";

        // 2. Serialize Objects (Elements)
        List<string> objectEntries = [];
        foreach (var obj in gs.objects) {
            objectEntries.Add($"{{\"name\": \"{obj.name}\", \"position\": [{obj.position[0]}, {obj.position[1]}]}}");
        }
        string objectsJson = "[" + string.Join(", ", [..objectEntries]) + "]";

        // 3. Serialize Orders
        List<string> orderEntries = [];
        foreach (var order in gs.all_orders) {
            string ingredients = "[\"" + string.Join("\", \"", order.ingredients) + "\"]";
            orderEntries.Add($"{{\"ingredients\": {ingredients}}}");
        }
        string ordersJson = "[" + string.Join(", ", [..orderEntries]) + "]";

        // 4. Combine into final GameState JSON object
        return $"{{\"players\": {playersJson}, \"objects\": {objectsJson}, \"all_orders\": {ordersJson}, \"timestep\": {gs.timestep}}}";
    }

    private static string ConvertLevelLayoutToString(List<LayoutObject> layoutObjects) {
        if (layoutObjects == null || layoutObjects.Count == 0) {
            return "[]";
        }

        List<string> layoutEntries = [];
        foreach (var layoutObject in layoutObjects) {
            string position = layoutObject.position != null && layoutObject.position.Length == 2
                ? $"[{layoutObject.position[0]}, {layoutObject.position[1]}]"
                : "[]";
            string ingredientField = string.IsNullOrEmpty(layoutObject.ingredient)
                ? string.Empty
                : $", \"ingredient\": \"{EscapeJsonString(layoutObject.ingredient)}\"";

            layoutEntries.Add(
                $"{{\"type\": \"{EscapeJsonString(layoutObject.type)}\"{ingredientField}, \"position\": {position}}}");
        }

        return "[" + string.Join(", ", layoutEntries.ToArray()) + "]";
    }

    private static string ConvertPlayerStartPositionsToString(List<float[]> playerStartPositions) {
        if (playerStartPositions == null || playerStartPositions.Count == 0) {
            return "[]";
        }

        List<string> playerEntries = [];
        foreach (var position in playerStartPositions) {
            if (position == null || position.Length != 2) {
                playerEntries.Add("[]");
                continue;
            }

            playerEntries.Add($"[{position[0]}, {position[1]}]");
        }

        return "[" + string.Join(", ", playerEntries.ToArray()) + "]";
    }

    private static string EscapeJsonString(string value) {
        if (string.IsNullOrEmpty(value)) {
            return string.Empty;
        }

        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"");
    }
}
