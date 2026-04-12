import heapq
import math

from enum import Enum

from .base import Agent


class Goal(str, Enum):
    DELIVER_SOUP = "DELIVER_SOUP"
    COLLECT_SOUP = "COLLECT_SOUP"
    COLLECT_PLATE = "COLLECT_PLATE"
    MAKE_SOUP = "MAKE_SOUP"
    COLLECT_CHOPPED_ONION = "COLLECT_CHOPPED_ONION"
    COLLECT_ONION = "COLLECT_ONION"
    CHOP_ONION = "CHOP_ONION"
    PUT_ONION_ON_CHOPPING_BLOCK = "PUT_ONION_ON_CHOPPING_BLOCK"
    MOVE_CHOPPED_ONION_TO_TABLETOP = "MOVE_CHOPPED_ONION_TO_TABLETOP"
    COLLECT_DIRTY_DISHES = "COLLECT_DIRTY_DISHES"
    PUT_DIRTY_DISHES_IN_SINK = "PUT_DIRTY_DISHES_IN_SINK"
    WASH_DISHES = "WASH_DISHES"
    PUT_DOWN_HELD_OBJECT = "PUT_DOWN_HELD_OBJECT"
    IDLE = "IDLE"


class AutoAgent(Agent):
    GRID_STEP = 0.6
    APPROACH_DISTANCE = 1.0
    ACTION_DISTANCE = 1.1
    WAYPOINT_TOLERANCE = 0.15
    BLOCKER_CLEARANCE = 0.95
    HOLDER_NAMES = {
        "tabletop",
        "dispenser",
        "chopping_board",
        "stove",
        "delivery_station",
        "plate_return_station",
        "drying_rack",
        "sink",
        "rubbish_bin",
    }
    BLOCKER_NAMES = HOLDER_NAMES

    def __init__(self, player_num: int) -> None:
        super().__init__(player_num)
        self.current_goal = Goal.IDLE.value
        self.current_target_id: str | None = None

    def update(self, delta_time: float, state: dict) -> dict:
        del delta_time
        world = self.build_world(state)
        goal, target = self.choose_goal(world)
        self.current_goal = goal.value
        self.current_target_id = target.get("id") if target is not None else None
        return self.execute_goal(world, goal, target)

    def build_world(self, state: dict) -> dict:
        players = state.get("players", [])
        objects = state.get("objects", [])
        players_by_id = {player["id"]: player for player in players}
        objects_by_id = {obj["id"]: obj for obj in objects}
        objects_by_name: dict[str, list[dict]] = {}
        for obj in objects:
            objects_by_name.setdefault(obj["name"], []).append(obj)

        return {
            "players": players,
            "players_by_id": players_by_id,
            "objects": objects,
            "objects_by_id": objects_by_id,
            "objects_by_name": objects_by_name,
            "self_player": players_by_id.get(f"player_{self.player_num}"),
            "other_players": [player for player in players if player["id"] != f"player_{self.player_num}"][0],
        }

    def choose_goal(self, world: dict) -> tuple[Goal, dict | None]:
        plated_soups = self.get_plated_soups(world)
        cooked_soups = self.get_cooked_soups(world)
        clean_plates = self.get_clean_plates(world)
        dirty_stacks = self.get_dirty_stacks(world)
        sinks_with_dirty_dishes = self.get_sinks_with_dirty_dishes(world)
        pots_needing_onions = self.get_pots_needing_onions(world)
        chopped_onions = self.get_chopped_onions(world)
        chopping_boards_with_raw_onions = self.get_chopping_boards_with_raw_onions(world)
        empty_chopping_boards = self.get_empty_chopping_boards(world)
        onion_dispensers = self.get_onion_dispensers(world)

        held_object = self.get_held_object(world)

        if self.is_plate_with_soup(held_object):
            return Goal.DELIVER_SOUP, self.nearest_object(world, self.get_objects(world, "delivery_station"))

        if plated_soups:
            return Goal.DELIVER_SOUP, self.nearest_object(world, plated_soups)

        if cooked_soups:
            if self.is_clean_plate(held_object):
                return Goal.COLLECT_SOUP, self.nearest_object(world, cooked_soups)
            if held_object is None and clean_plates:
                return Goal.COLLECT_PLATE, self.nearest_object(world, clean_plates)

        if not clean_plates and not self.is_clean_plate(held_object):
            if sinks_with_dirty_dishes:
                return Goal.WASH_DISHES, self.nearest_object(world, sinks_with_dirty_dishes)
            if self.is_dirty_stack(held_object):
                return Goal.PUT_DIRTY_DISHES_IN_SINK, self.nearest_object(world, self.get_objects(world, "sink"))
            if dirty_stacks:
                return Goal.COLLECT_DIRTY_DISHES, self.nearest_object(world, dirty_stacks)

        if pots_needing_onions:
            if self.is_chopped_onion(held_object):
                return Goal.MAKE_SOUP, self.nearest_object(world, pots_needing_onions)
            if chopped_onions:
                return Goal.COLLECT_CHOPPED_ONION, self.nearest_object(world, chopped_onions)
            if self.is_raw_onion(held_object):
                if empty_chopping_boards:
                    return Goal.PUT_ONION_ON_CHOPPING_BLOCK, self.nearest_object(world, empty_chopping_boards)
                if chopping_boards_with_raw_onions:
                    return Goal.CHOP_ONION, self.nearest_object(world, chopping_boards_with_raw_onions)
            if chopping_boards_with_raw_onions:
                return Goal.CHOP_ONION, self.nearest_object(world, chopping_boards_with_raw_onions)
            if held_object is None and onion_dispensers:
                return Goal.COLLECT_ONION, self.nearest_object(world, onion_dispensers)

        if sinks_with_dirty_dishes:
            return Goal.WASH_DISHES, self.nearest_object(world, sinks_with_dirty_dishes)

        if self.is_dirty_stack(held_object):
            return Goal.PUT_DIRTY_DISHES_IN_SINK, self.nearest_object(world, self.get_objects(world, "sink"))

        if held_object is not None:
            return Goal.PUT_DOWN_HELD_OBJECT, self.find_drop_target(world, held_object)

        return Goal.IDLE, None

    def execute_goal(self, world: dict, goal: Goal, target: dict | None) -> dict:
        if goal == Goal.DELIVER_SOUP:
            return self.execute_deliver_soup(world, target)
        if goal == Goal.COLLECT_SOUP:
            return self.execute_collect_soup(world, target)
        if goal == Goal.COLLECT_PLATE:
            return self.execute_collect_plate(world, target)
        if goal == Goal.MAKE_SOUP:
            return self.execute_make_soup(world, target)
        if goal == Goal.COLLECT_CHOPPED_ONION:
            return self.execute_collect_chopped_onion(world, target)
        if goal == Goal.COLLECT_ONION:
            return self.execute_collect_onion(world, target)
        if goal == Goal.PUT_ONION_ON_CHOPPING_BLOCK:
            return self.execute_put_onion_on_chopping_board(world, target)
        if goal == Goal.CHOP_ONION:
            return self.execute_chop_onion(world, target)
        if goal == Goal.MOVE_CHOPPED_ONION_TO_TABLETOP:
            return self.execute_move_chopped_onion_to_tabletop(world, target)
        if goal == Goal.COLLECT_DIRTY_DISHES:
            return self.execute_collect_dirty_dishes(world, target)
        if goal == Goal.PUT_DIRTY_DISHES_IN_SINK:
            return self.execute_put_dirty_dishes_in_sink(world, target)
        if goal == Goal.WASH_DISHES:
            return self.execute_wash_dishes(world, target)
        if goal == Goal.PUT_DOWN_HELD_OBJECT:
            return self.execute_put_down_held_object(world, target)
        return self.zero_input()

    def execute_deliver_soup(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        delivery_station = self.nearest_object(world, self.get_objects(world, "delivery_station"))
        if delivery_station is None:
            return self.zero_input()
        if self.is_plate_with_soup(held_object):
            return self.move_to_object(world, delivery_station, carry=True)
        if held_object is None and target is not None:
            plate = self.get_parent_entity(world, target)
            if plate is not None:
                return self.move_to_object(world, self.get_holder_target(world, plate), carry=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_collect_soup(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_plate_with_soup(held_object):
            return self.execute_deliver_soup(world, target)
        if target is None:
            return self.zero_input()
        if self.is_clean_plate(held_object):
            return self.move_to_object(world, self.get_holder_target(world, target), carry=True)
        if held_object is None:
            clean_plate = self.nearest_object(world, self.get_clean_plates(world))
            return self.execute_collect_plate(world, clean_plate)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_collect_plate(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_clean_plate(held_object):
            return self.zero_input()
        if held_object is None and target is not None:
            return self.move_to_object(world, self.get_holder_target(world, target), carry=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_make_soup(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_chopped_onion(held_object) and target is not None:
            return self.move_to_object(world, self.get_holder_target(world, target), carry=True)
        if held_object is None:
            chopped_onion = self.nearest_object(world, self.get_chopped_onions(world))
            return self.execute_collect_chopped_onion(world, chopped_onion)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_collect_chopped_onion(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_chopped_onion(held_object):
            pot = self.nearest_object(world, self.get_pots_needing_onions(world))
            return self.execute_make_soup(world, pot)
        if held_object is None and target is not None:
            return self.move_to_object(world, self.get_holder_target(world, target), carry=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_collect_onion(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_raw_onion(held_object):
            chopping_board = self.nearest_object(world, self.get_empty_chopping_boards(world))
            return self.execute_put_onion_on_chopping_board(world, chopping_board)
        if held_object is None and target is not None:
            return self.move_to_object(world, target, carry=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_put_onion_on_chopping_board(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_raw_onion(held_object) and target is not None:
            return self.move_to_object(world, target, carry=True)
        if held_object is None:
            onion_dispenser = self.nearest_object(world, self.get_onion_dispensers(world))
            return self.execute_collect_onion(world, onion_dispenser)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_chop_onion(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if target is None:
            return self.zero_input()
        if held_object is None or self.is_raw_onion(held_object):
            return self.move_to_object(world, target, interact=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_move_chopped_onion_to_tabletop(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_chopped_onion(held_object):
            tabletop = target if target is not None and target["name"] == "tabletop" else self.nearest_object(world, self.get_empty_tabletops(world))
            return self.move_to_object(world, tabletop, carry=True)
        if target is not None:
            return self.move_to_object(world, target, carry=True)
        return self.zero_input()

    def execute_collect_dirty_dishes(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_dirty_stack(held_object):
            sink = self.nearest_object(world, self.get_objects(world, "sink"))
            return self.execute_put_dirty_dishes_in_sink(world, sink)
        if held_object is None and target is not None:
            return self.move_to_object(world, self.get_holder_target(world, target), carry=True)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_put_dirty_dishes_in_sink(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_dirty_stack(held_object) and target is not None:
            return self.move_to_object(world, target, carry=True)
        if held_object is None:
            dirty_stack = self.nearest_object(world, self.get_dirty_stacks(world))
            return self.execute_collect_dirty_dishes(world, dirty_stack)
        return self.execute_put_down_held_object(world, self.find_drop_target(world, held_object))

    def execute_wash_dishes(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if self.is_dirty_stack(held_object):
            sink = self.nearest_object(world, self.get_objects(world, "sink"))
            return self.execute_put_dirty_dishes_in_sink(world, sink)
        if target is not None:
            return self.move_to_object(world, target, interact=True)
        return self.zero_input()

    def execute_put_down_held_object(self, world: dict, target: dict | None) -> dict:
        held_object = self.get_held_object(world)
        if held_object is None or target is None:
            return self.zero_input()
        return self.move_to_object(world, target, carry=True)

    def move_to_object(self, world: dict, obj: dict | None, carry: bool = False, interact: bool = False) -> dict:
        if obj is None:
            return self.zero_input()
        return self.move_to_position(world, obj["position"], carry=carry, interact=interact)

    def move_to_position(self, world: dict, target_position: list[float], carry: bool = False, interact: bool = False) -> dict:
        player = world["self_player"]
        if player is None:
            return self.zero_input()

        player_position = player["position"]
        if self.distance(player_position, target_position) <= self.ACTION_DISTANCE:
            return self.input_state(carry=carry, interact=interact)

        path = self.find_path(world, player_position, target_position)
        next_position = path[1] if len(path) > 1 else path[0] if path else target_position
        move_x = next_position[0] - player_position[0]
        move_y = player_position[1] - next_position[1]

        if abs(move_x) < self.WAYPOINT_TOLERANCE and abs(move_y) < self.WAYPOINT_TOLERANCE:
            move_x = target_position[0] - player_position[0]
            move_y = player_position[1] - target_position[1]

        return self.input_state(move_x=move_x, move_y=move_y)

    def find_path(self, world: dict, start_position: list[float], target_position: list[float]) -> list[list[float]]:
        goal_positions = self.get_approach_positions(world, target_position)
        if not goal_positions:
            goal_positions = [target_position]

        start_node = self.find_nearest_walkable_node(world, self.snap_to_grid(start_position))
        goal_nodes = {
            self.find_nearest_walkable_node(world, self.snap_to_grid(goal_position))
            for goal_position in goal_positions
        }
        goal_nodes.discard(None)
        if start_node is None or not goal_nodes:
            return [target_position]
        if start_node in goal_nodes:
            return [list(start_node)]

        open_nodes: list[tuple[float, tuple[float, float]]] = []
        heapq.heappush(open_nodes, (0.0, start_node))
        came_from: dict[tuple[float, float], tuple[float, float]] = {}
        g_score = {start_node: 0.0}

        while open_nodes:
            _, current_node = heapq.heappop(open_nodes)
            if current_node in goal_nodes:
                return self.reconstruct_path(came_from, current_node)

            for neighbor in self.get_neighbors(world, current_node):
                tentative_score = g_score[current_node] + self.distance(current_node, neighbor)
                if tentative_score >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_score
                priority = tentative_score + min(self.distance(neighbor, goal_node) for goal_node in goal_nodes)
                heapq.heappush(open_nodes, (priority, neighbor))

        return [target_position]

    def reconstruct_path(
        self,
        came_from: dict[tuple[float, float], tuple[float, float]],
        current_node: tuple[float, float],
    ) -> list[list[float]]:
        path = [current_node]
        while current_node in came_from:
            current_node = came_from[current_node]
            path.append(current_node)
        path.reverse()
        return [[node[0], node[1]] for node in path]

    def get_neighbors(self, world: dict, node: tuple[float, float]) -> list[tuple[float, float]]:
        neighbors = []
        for delta_x, delta_y in (
            (self.GRID_STEP, 0.0),
            (-self.GRID_STEP, 0.0),
            (0.0, self.GRID_STEP),
            (0.0, -self.GRID_STEP),
            (self.GRID_STEP, self.GRID_STEP),
            (self.GRID_STEP, -self.GRID_STEP),
            (-self.GRID_STEP, self.GRID_STEP),
            (-self.GRID_STEP, -self.GRID_STEP),
        ):
            neighbor = (round(node[0] + delta_x, 3), round(node[1] + delta_y, 3))
            if self.is_walkable(world, neighbor):
                neighbors.append(neighbor)
        return neighbors

    def get_approach_positions(self, world: dict, target_position: list[float]) -> list[list[float]]:
        candidates = []
        for delta_x, delta_y in (
            (self.APPROACH_DISTANCE, 0.0),
            (-self.APPROACH_DISTANCE, 0.0),
            (0.0, self.APPROACH_DISTANCE),
            (0.0, -self.APPROACH_DISTANCE),
            (self.APPROACH_DISTANCE, self.APPROACH_DISTANCE * 0.5),
            (self.APPROACH_DISTANCE, -self.APPROACH_DISTANCE * 0.5),
            (-self.APPROACH_DISTANCE, self.APPROACH_DISTANCE * 0.5),
            (-self.APPROACH_DISTANCE, -self.APPROACH_DISTANCE * 0.5),
        ):
            candidate = [round(target_position[0] + delta_x, 3), round(target_position[1] + delta_y, 3)]
            if self.distance(candidate, target_position) <= self.ACTION_DISTANCE + 0.1:
                snapped_candidate = self.find_nearest_walkable_node(world, self.snap_to_grid(candidate))
                if snapped_candidate is not None and self.distance(snapped_candidate, target_position) <= self.ACTION_DISTANCE + 0.2:
                    candidates.append([snapped_candidate[0], snapped_candidate[1]])
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            candidate_key = tuple(candidate)
            if candidate_key not in seen:
                seen.add(candidate_key)
                unique_candidates.append(candidate)
        return unique_candidates

    def snap_to_grid(self, position: list[float] | tuple[float, float]) -> tuple[float, float]:
        return (
            round(round(position[0] / self.GRID_STEP) * self.GRID_STEP, 3),
            round(round(position[1] / self.GRID_STEP) * self.GRID_STEP, 3),
        )

    def find_nearest_walkable_node(
        self,
        world: dict,
        node: tuple[float, float],
    ) -> tuple[float, float] | None:
        if self.is_walkable(world, node):
            return node
        search_offsets = [0.0, self.GRID_STEP, -self.GRID_STEP, 2 * self.GRID_STEP, -2 * self.GRID_STEP]
        for delta_x in search_offsets:
            for delta_y in search_offsets:
                candidate = (round(node[0] + delta_x, 3), round(node[1] + delta_y, 3))
                if self.is_walkable(world, candidate):
                    return candidate
        return None

    def is_walkable(self, world: dict, node: tuple[float, float]) -> bool:
        blockers = [
            obj["position"]
            for obj in world["objects"]
            if obj["name"] in self.BLOCKER_NAMES
        ]
        blockers.append(world["other_players"]["position"])

        for blocker in blockers:
            if self.distance(node, blocker) < self.BLOCKER_CLEARANCE:
                return False
        return True

    def get_held_object(self, world: dict) -> dict | None:
        player = world["self_player"]
        if player is None:
            return None
        held_object_id = player.get("held_object_id")
        if held_object_id is None:
            return None
        return world["objects_by_id"].get(held_object_id)

    def get_parent_entity(self, world: dict, entity: dict | None) -> dict | None:
        if entity is None:
            return None
        parent_id = entity.get("parent_id")
        if parent_id is None:
            return None
        return world["players_by_id"].get(parent_id) or world["objects_by_id"].get(parent_id)

    def get_holder_target(self, world: dict, entity: dict) -> dict:
        current_entity = entity
        while True:
            parent_entity = self.get_parent_entity(world, current_entity)
            if parent_entity is None or parent_entity["id"].startswith("player_"):
                return current_entity
            current_entity = parent_entity

    def get_root_player(self, world: dict, entity: dict | None) -> dict | None:
        current_entity = entity
        while current_entity is not None:
            parent_entity = self.get_parent_entity(world, current_entity)
            if parent_entity is None:
                return None
            if parent_entity["id"].startswith("player_"):
                return parent_entity
            current_entity = parent_entity
        return None

    def is_held_by_other_player(self, world: dict, entity: dict) -> bool:
        root_player = self.get_root_player(world, entity)
        return root_player is not None and root_player["id"] != f"player_{self.player_num}"

    def is_accessible(self, world: dict, entity: dict) -> bool:
        return not self.is_held_by_other_player(world, entity)

    def get_objects(self, world: dict, name: str) -> list[dict]:
        return world["objects_by_name"].get(name, [])

    def get_plated_soups(self, world: dict) -> list[dict]:
        return [
            soup for soup in self.get_objects(world, "soup")
            if self.is_plated_soup(soup) and self.is_accessible(world, soup)
        ]

    def get_cooked_soups(self, world: dict) -> list[dict]:
        return [
            soup for soup in self.get_objects(world, "soup")
            if self.is_cooked_soup(soup) and self.is_accessible(world, soup)
        ]

    def get_clean_plates(self, world: dict) -> list[dict]:
        return [
            plate for plate in self.get_objects(world, "plate")
            if self.is_clean_plate(plate) and self.is_accessible(world, plate)
        ]

    def get_dirty_stacks(self, world: dict) -> list[dict]:
        return [
            stack for stack in self.get_objects(world, "stacked_dirty_plates")
            if stack.get("parent_name") != "sink" and self.is_accessible(world, stack)
        ]

    def get_sinks_with_dirty_dishes(self, world: dict) -> list[dict]:
        dirty_sink_ids = {
            stack.get("parent_id")
            for stack in self.get_objects(world, "stacked_dirty_plates")
            if stack.get("parent_name") == "sink"
        }
        return [sink for sink in self.get_objects(world, "sink") if sink["id"] in dirty_sink_ids]

    def get_pots_needing_onions(self, world: dict) -> list[dict]:
        pots = []
        for pot in self.get_objects(world, "pot"):
            held_object_id = pot.get("held_object_id")
            if held_object_id is None:
                pots.append(pot)
                continue
            soup = world["objects_by_id"].get(held_object_id)
            if soup is None or soup["name"] != "soup":
                continue
            if self.soup_ingredient_count(soup) < 3:
                pots.append(pot)
        return [pot for pot in pots if self.is_accessible(world, pot)]

    def get_chopped_onions(self, world: dict) -> list[dict]:
        return [
            onion for onion in self.get_objects(world, "onion")
            if self.is_chopped_onion(onion) and self.is_accessible(world, onion)
        ]

    def get_onion_dispensers(self, world: dict) -> list[dict]:
        return [
            dispenser for dispenser in self.get_objects(world, "dispenser")
            if dispenser.get("ingredient") == "onion"
        ]

    def get_empty_chopping_boards(self, world: dict) -> list[dict]:
        return [
            board for board in self.get_objects(world, "chopping_board")
            if board.get("held_object_id") is None
        ]

    def get_empty_tabletops(self, world: dict) -> list[dict]:
        return [
            tabletop for tabletop in self.get_objects(world, "tabletop")
            if tabletop.get("held_object_id") is None
        ]

    def get_chopping_boards_with_raw_onions(self, world: dict) -> list[dict]:
        boards = []
        for board in self.get_objects(world, "chopping_board"):
            held_object_id = board.get("held_object_id")
            if held_object_id is None:
                continue
            onion = world["objects_by_id"].get(held_object_id)
            if onion is not None and self.is_raw_onion(onion):
                boards.append(board)
        return boards

    def get_chopping_boards_with_chopped_onions(self, world: dict) -> list[dict]:
        boards = []
        for board in self.get_objects(world, "chopping_board"):
            held_object_id = board.get("held_object_id")
            if held_object_id is None:
                continue
            onion = world["objects_by_id"].get(held_object_id)
            if onion is not None and self.is_chopped_onion(onion):
                boards.append(board)
        return boards

    def find_drop_target(self, world: dict, held_object: dict | None) -> dict | None:
        if held_object is None:
            return None
        if self.is_dirty_stack(held_object):
            return self.nearest_object(world, self.get_objects(world, "sink"))
        if self.is_clean_plate(held_object):
            return self.nearest_object(world, self.get_empty_tabletops(world))
        if self.is_chopped_onion(held_object):
            pot = self.nearest_object(world, self.get_pots_needing_onions(world))
            if pot is not None:
                return self.get_holder_target(world, pot)
        if self.is_raw_onion(held_object):
            chopping_board = self.nearest_object(world, self.get_empty_chopping_boards(world))
            if chopping_board is not None:
                return chopping_board
        return self.nearest_object(world, self.get_empty_tabletops(world))

    def nearest_object(self, world: dict, objects: list[dict]) -> dict | None:
        player = world["self_player"]
        if player is None or not objects:
            return None
        return min(objects, key=lambda obj: self.distance(player["position"], obj["position"]))

    def is_clean_plate(self, obj: dict | None) -> bool:
        return obj is not None and obj["name"] == "plate" and obj.get("held_object_name") is None

    def is_plate_with_soup(self, obj: dict | None) -> bool:
        return obj is not None and obj["name"] == "plate" and obj.get("held_object_name") == "soup"

    def is_dirty_stack(self, obj: dict | None) -> bool:
        return obj is not None and obj["name"] == "stacked_dirty_plates"

    def is_chopped_onion(self, obj: dict | None) -> bool:
        return obj is not None and obj["name"] == "onion" and float(obj.get("progress", 0.0)) >= 1.0

    def is_raw_onion(self, obj: dict | None) -> bool:
        return obj is not None and obj["name"] == "onion" and float(obj.get("progress", 0.0)) < 1.0

    def is_plated_soup(self, soup: dict) -> bool:
        return (
            soup["name"] == "soup"
            and soup.get("parent_name") == "plate"
            and self.get_soup_cooking_state(soup) == "cooked"
            and self.soup_ingredient_count(soup) == 3
        )

    def is_cooked_soup(self, soup: dict) -> bool:
        return (
            soup["name"] == "soup"
            and soup.get("parent_name") == "pot"
            and self.get_soup_cooking_state(soup) == "cooked"
            and self.soup_ingredient_count(soup) == 3
        )

    def soup_ingredient_count(self, soup: dict) -> int:
        return sum(1 for ingredient in soup.get("ingredients", []) if ingredient is not None)

    def get_soup_cooking_state(self, soup: dict | None) -> str:
        if soup is None:
            return "raw"
        cooking_state = soup.get("cooking_state")
        if cooking_state is not None:
            return str(cooking_state).lower()
        progress = float(soup.get("progress", 0.0))
        if progress >= 2.0:
            return "burnt"
        if progress >= 1.0:
            return "cooked"
        return "raw"

    def distance(self, position_a: list[float] | tuple[float, float], position_b: list[float] | tuple[float, float]) -> float:
        return math.hypot(position_a[0] - position_b[0], position_a[1] - position_b[1])

    def input_state(
        self,
        move_x: float = 0.0,
        move_y: float = 0.0,
        interact: bool = False,
        carry: bool = False,
    ) -> dict:
        return {
            "move_x": move_x,
            "move_y": move_y,
            "interact": interact,
            "carry": carry,
        }

    def zero_input(self) -> dict:
        return self.input_state()


class AutoOnionAgent(AutoAgent):
    def choose_goal(self, world: dict) -> tuple[Goal, dict | None]:
        held_object = self.get_held_object(world)
        empty_tabletops = self.get_empty_tabletops(world)
        empty_chopping_boards = self.get_empty_chopping_boards(world)
        chopping_boards_with_raw_onions = self.get_chopping_boards_with_raw_onions(world)
        chopping_boards_with_chopped_onions = self.get_chopping_boards_with_chopped_onions(world)
        onion_dispensers = self.get_onion_dispensers(world)

        if self.is_chopped_onion(held_object):
            return Goal.MOVE_CHOPPED_ONION_TO_TABLETOP, self.nearest_object(world, empty_tabletops)

        if chopping_boards_with_chopped_onions:
            return Goal.MOVE_CHOPPED_ONION_TO_TABLETOP, self.nearest_object(world, chopping_boards_with_chopped_onions)

        if self.is_raw_onion(held_object):
            if empty_chopping_boards:
                return Goal.PUT_ONION_ON_CHOPPING_BLOCK, self.nearest_object(world, empty_chopping_boards)
            return Goal.PUT_DOWN_HELD_OBJECT, self.nearest_object(world, empty_tabletops)

        if chopping_boards_with_raw_onions:
            return Goal.CHOP_ONION, self.nearest_object(world, chopping_boards_with_raw_onions)

        if held_object is None and onion_dispensers:
            return Goal.COLLECT_ONION, self.nearest_object(world, onion_dispensers)

        if held_object is not None:
            return Goal.PUT_DOWN_HELD_OBJECT, self.nearest_object(world, empty_tabletops)

        return Goal.IDLE, None

    def find_drop_target(self, world: dict, held_object: dict | None) -> dict | None:
        if self.is_chopped_onion(held_object):
            return self.nearest_object(world, self.get_empty_tabletops(world))
        return super().find_drop_target(world, held_object)


class AutoSoupAgent(AutoAgent):
    def choose_goal(self, world: dict) -> tuple[Goal, dict | None]:
        plated_soups = self.get_plated_soups(world)
        cooked_soups = self.get_cooked_soups(world)
        clean_plates = self.get_clean_plates(world)
        dirty_stacks = self.get_dirty_stacks(world)
        sinks_with_dirty_dishes = self.get_sinks_with_dirty_dishes(world)
        pots_needing_onions = self.get_pots_needing_onions(world)
        chopped_onions = self.get_chopped_onions(world)
        held_object = self.get_held_object(world)

        if self.is_plate_with_soup(held_object):
            return Goal.DELIVER_SOUP, self.nearest_object(world, self.get_objects(world, "delivery_station"))

        if plated_soups:
            return Goal.DELIVER_SOUP, self.nearest_object(world, plated_soups)

        if cooked_soups:
            if self.is_clean_plate(held_object):
                return Goal.COLLECT_SOUP, self.nearest_object(world, cooked_soups)
            if held_object is None and clean_plates:
                return Goal.COLLECT_PLATE, self.nearest_object(world, clean_plates)

        if not clean_plates and not self.is_clean_plate(held_object):
            if sinks_with_dirty_dishes:
                return Goal.WASH_DISHES, self.nearest_object(world, sinks_with_dirty_dishes)
            if self.is_dirty_stack(held_object):
                return Goal.PUT_DIRTY_DISHES_IN_SINK, self.nearest_object(world, self.get_objects(world, "sink"))
            if dirty_stacks:
                return Goal.COLLECT_DIRTY_DISHES, self.nearest_object(world, dirty_stacks)

        if self.is_chopped_onion(held_object):
            return Goal.MAKE_SOUP, self.nearest_object(world, pots_needing_onions)

        if pots_needing_onions and chopped_onions:
            return Goal.COLLECT_CHOPPED_ONION, self.nearest_object(world, chopped_onions)

        if sinks_with_dirty_dishes:
            return Goal.WASH_DISHES, self.nearest_object(world, sinks_with_dirty_dishes)

        if self.is_dirty_stack(held_object):
            return Goal.PUT_DIRTY_DISHES_IN_SINK, self.nearest_object(world, self.get_objects(world, "sink"))

        if held_object is not None:
            return Goal.PUT_DOWN_HELD_OBJECT, self.find_drop_target(world, held_object)

        return Goal.IDLE, None
