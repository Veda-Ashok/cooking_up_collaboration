using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class PlayerStateDto {
    public string id;
    public string name;
    public float[] position;
    public float[] facing;
    public string held_object_id;
    public string held_object_name;
}

[Serializable]
public class Order {
    public string[] ingredients;
}

[Serializable]
public class ObjectStateDto {
    public string id;
    public string name;
    public float[] position;
    public string parent_id;
    public string parent_name;
    public string held_object_id;
    public string held_object_name;
    public float? progress;
    public string cooking_state;
    public string[] ingredients;
    public string[] plate_ids;
    public int? plate_count;
    public string ingredient;
}

[Serializable]
public class LayoutObject {
    public string type;
    public string ingredient;
    public float[] position;
}

[Serializable]
public class ExtractedStateDto {
    public List<PlayerStateDto> players;
    public List<ObjectStateDto> objects;
}
