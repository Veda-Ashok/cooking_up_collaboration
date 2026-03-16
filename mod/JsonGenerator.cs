using System.Collections.Generic;
using UnityEngine;
using System.IO;

public static class JsonGenerator{

    public static string GenerateGameJson(List<Player> players, List<Element> objects, List<Order> orders, string[,] layoutGrid,
                                          string layoutName, float timeLeft, float timeElapsed, int gameloop ) 
    {
        //Prepare and Stringify the GameState
        GameState internalState = new GameState {
            players = players,
            objects = objects,
            all_orders = orders,
            timestep = gameloop-1
        };
        string stateString = ConvertGameStateToString(internalState);
        
        string layoutString = ConvertLayoutToString(layoutGrid);

        GameRecord record = new GameRecord {
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

        // 4. Final Serialization
        string json = JsonUtility.ToJson(record, true);
        File.WriteAllText("state.json", json); //TODO: Change this to the desired output path
        return stateString;
    }

    private static string ConvertLayoutToString(string[,] grid) {
        List<string> rows = new List<string>();
        for (int i = 0; i < grid.GetLength(0); i++) {
            List<string> row = new List<string>();
            for (int j = 0; j < grid.GetLength(1); j++) {
                row.Add(grid[i, j]);
            }
            rows.Add("[\"" + string.Join("\", \"", [..row]) + "\"]");
        }
        return "[" + string.Join(", ", [.. rows]) + "]";
    }

    private static string ConvertGameStateToString(GameState gs) {
        // 1. Serialize Players
        List<string> playerEntries = new List<string>();
        foreach (var p in gs.players) {
            playerEntries.Add($"{{\"position\": [{p.position[0]}, {p.position[1]}], \"orientation\": [{p.orientation[0]}, {p.orientation[1]}], \"held_object\": \"{p.held_object ?? ""}\"}}");
        }
        string playersJson = "[" + string.Join(", ", [..playerEntries]) + "]";

        // 2. Serialize Objects (Elements)
        List<string> objectEntries = new List<string>();
        foreach (var obj in gs.objects) {
            objectEntries.Add($"{{\"name\": \"{obj.name}\", \"position\": [{obj.position[0]}, {obj.position[1]}]}}");
        }
        string objectsJson = "[" + string.Join(", ", [..objectEntries]) + "]";

        // 3. Serialize Orders
        List<string> orderEntries = new List<string>();
        foreach (var order in gs.all_orders) {
            string ingredients = "[\"" + string.Join("\", \"", order.ingredients) + "\"]";
            orderEntries.Add($"{{\"ingredients\": {ingredients}}}");
        }
        string ordersJson = "[" + string.Join(", ", [..orderEntries]) + "]";

        // 4. Combine into final GameState JSON object
        return $"{{\"players\": {playersJson}, \"objects\": {objectsJson}, \"all_orders\": {ordersJson}, \"timestep\": {gs.timestep}}}";
    }
}