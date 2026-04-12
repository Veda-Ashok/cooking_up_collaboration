using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

public static class JsonGenerator
{
    public static string GenerateGameJson(List<PlayerStateDto> players, List<ObjectStateDto> objects)
    {
        ExtractedStateDto state = new()
        {
            players = players ?? new List<PlayerStateDto>(),
            objects = objects ?? new List<ObjectStateDto>()
        };

        string json = ConvertStateToString(state);
        File.WriteAllText("state.json", json);
        return json;
    }

    public static void GenerateLevelJson(List<LayoutObject> layoutObjects, List<float[]> playerStartPositions)
    {
        string layoutString = ConvertLevelLayoutToString(layoutObjects);
        string playerStartsString = ConvertPlayerStartPositionsToString(playerStartPositions);
        string json = "{\"layout\": " + layoutString + ", \"player_starts\": " + playerStartsString + "}";
        File.WriteAllText("level.json", json);
    }

    private static string ConvertStateToString(ExtractedStateDto state)
    {
        List<string> playerEntries = new List<string>();
        foreach (PlayerStateDto player in state.players)
        {
            StringBuilder builder = new StringBuilder();
            builder.Append("{");
            builder.Append("\"id\": \"").Append(EscapeJsonString(player.id)).Append("\"");
            builder.Append(", \"name\": \"").Append(EscapeJsonString(player.name)).Append("\"");
            builder.Append(", \"position\": ").Append(SerializeFloatArray(player.position));

            if (!string.IsNullOrEmpty(player.held_object_id))
            {
                builder.Append(", \"held_object_id\": \"").Append(EscapeJsonString(player.held_object_id)).Append("\"");
            }

            if (!string.IsNullOrEmpty(player.held_object_name))
            {
                builder.Append(", \"held_object_name\": \"").Append(EscapeJsonString(player.held_object_name)).Append("\"");
            }

            builder.Append("}");
            playerEntries.Add(builder.ToString());
        }

        List<string> objectEntries = new List<string>();
        foreach (ObjectStateDto obj in state.objects)
        {
            StringBuilder builder = new StringBuilder();
            builder.Append("{");
            builder.Append("\"id\": \"").Append(EscapeJsonString(obj.id)).Append("\"");
            builder.Append(", \"name\": \"").Append(EscapeJsonString(obj.name)).Append("\"");
            builder.Append(", \"position\": ").Append(SerializeFloatArray(obj.position));

            if (!string.IsNullOrEmpty(obj.parent_id))
            {
                builder.Append(", \"parent_id\": \"").Append(EscapeJsonString(obj.parent_id)).Append("\"");
            }

            if (!string.IsNullOrEmpty(obj.parent_name))
            {
                builder.Append(", \"parent_name\": \"").Append(EscapeJsonString(obj.parent_name)).Append("\"");
            }

            if (!string.IsNullOrEmpty(obj.held_object_id))
            {
                builder.Append(", \"held_object_id\": \"").Append(EscapeJsonString(obj.held_object_id)).Append("\"");
            }

            if (!string.IsNullOrEmpty(obj.held_object_name))
            {
                builder.Append(", \"held_object_name\": \"").Append(EscapeJsonString(obj.held_object_name)).Append("\"");
            }

            if (obj.progress.HasValue)
            {
                builder.Append(", \"progress\": ").Append(FormatFloat(obj.progress.Value));
            }

            if (!string.IsNullOrEmpty(obj.cooking_state))
            {
                builder.Append(", \"cooking_state\": \"").Append(EscapeJsonString(obj.cooking_state)).Append("\"");
            }

            if (obj.ingredients != null)
            {
                builder.Append(", \"ingredients\": ").Append(SerializeStringArrayAllowNulls(obj.ingredients));
            }

            if (obj.plate_ids != null)
            {
                builder.Append(", \"plate_ids\": ").Append(SerializeStringArray(obj.plate_ids));
            }

            if (obj.plate_count.HasValue)
            {
                builder.Append(", \"plate_count\": ").Append(obj.plate_count.Value.ToString(CultureInfo.InvariantCulture));
            }

            if (!string.IsNullOrEmpty(obj.ingredient))
            {
                builder.Append(", \"ingredient\": \"").Append(EscapeJsonString(obj.ingredient)).Append("\"");
            }

            builder.Append("}");
            objectEntries.Add(builder.ToString());
        }

        return "{\"players\": [" + string.Join(", ", playerEntries.ToArray()) + "], \"objects\": [" + string.Join(", ", objectEntries.ToArray()) + "]}";
    }

    private static string ConvertLevelLayoutToString(List<LayoutObject> layoutObjects)
    {
        if (layoutObjects == null || layoutObjects.Count == 0)
        {
            return "[]";
        }

        List<string> layoutEntries = new List<string>();
        foreach (LayoutObject layoutObject in layoutObjects)
        {
            string position = layoutObject.position != null && layoutObject.position.Length == 2
                ? SerializeFloatArray(layoutObject.position)
                : "[]";
            string ingredientField = string.IsNullOrEmpty(layoutObject.ingredient)
                ? string.Empty
                : ", \"ingredient\": \"" + EscapeJsonString(layoutObject.ingredient) + "\"";

            layoutEntries.Add("{\"type\": \"" + EscapeJsonString(layoutObject.type) + "\"" + ingredientField + ", \"position\": " + position + "}");
        }

        return "[" + string.Join(", ", layoutEntries.ToArray()) + "]";
    }

    private static string ConvertPlayerStartPositionsToString(List<float[]> playerStartPositions)
    {
        if (playerStartPositions == null || playerStartPositions.Count == 0)
        {
            return "[]";
        }

        List<string> playerEntries = new List<string>();
        foreach (float[] position in playerStartPositions)
        {
            if (position == null || position.Length != 2)
            {
                playerEntries.Add("[]");
                continue;
            }

            playerEntries.Add(SerializeFloatArray(position));
        }

        return "[" + string.Join(", ", playerEntries.ToArray()) + "]";
    }

    private static string SerializeFloatArray(float[] values)
    {
        if (values == null || values.Length == 0)
        {
            return "[]";
        }

        List<string> entries = new List<string>();
        for (int i = 0; i < values.Length; i++)
        {
            entries.Add(FormatFloat(values[i]));
        }

        return "[" + string.Join(", ", entries.ToArray()) + "]";
    }

    private static string SerializeStringArray(string[] values)
    {
        if (values == null || values.Length == 0)
        {
            return "[]";
        }

        List<string> entries = new List<string>();
        for (int i = 0; i < values.Length; i++)
        {
            entries.Add("\"" + EscapeJsonString(values[i] ?? string.Empty) + "\"");
        }

        return "[" + string.Join(", ", entries.ToArray()) + "]";
    }

    private static string SerializeStringArrayAllowNulls(string[] values)
    {
        if (values == null || values.Length == 0)
        {
            return "[]";
        }

        List<string> entries = new List<string>();
        for (int i = 0; i < values.Length; i++)
        {
            entries.Add(values[i] == null ? "null" : "\"" + EscapeJsonString(values[i]) + "\"");
        }

        return "[" + string.Join(", ", entries.ToArray()) + "]";
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("0.0###", CultureInfo.InvariantCulture);
    }

    private static string EscapeJsonString(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"");
    }
}
