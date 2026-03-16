using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class Player {
    public int[] position;
    public int[] orientation;
    public string held_object;
}

[Serializable]
public class Order {
    public string[] ingredients;
}

[Serializable]
public class Element {
    public string name;
    public int[] position;
}

[Serializable]
public class GameState {
    public List<Player> players;
    public List<Element> objects;
    public List<Order> all_orders;
    public int timestep;
}

[Serializable]
public class GameRecord {
    public string state;
    public float time_left;
    public int score;
    public float time_elapsed;
    public int cur_gameloop;
    public string layout;
    public string layout_name;
    public bool player_0_is_human;
    public bool player_1_is_human;
}