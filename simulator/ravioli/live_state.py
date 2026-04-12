import json
import math
import threading
import time

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .agents import create_agent, get_agent_id

DEFAULT_TUTORIAL_CLEAR_DURATION = 2.0
DIRTY_DISH_OBJECT_NAMES = {"stacked_dirty_plates", "dirty_plate"}
LIVE_INVERT_STICK_X = False
LIVE_INVERT_STICK_Y = True
LIVE_SWAP_STICK_AXES = False
LIVE_SWAP_AGENT_STATE_PLAYERS = True
ACTION_FACING_STICK_STRENGTH = 0.35


@dataclass
class LivePayload:
    data: dict[str, Any] | None = None
    raw_json: str | None = None
    received_at: float | None = None
    count: int = 0


class LiveStateStore:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._state = LivePayload()
        self._level = LivePayload()

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def update_state(self, raw_json: str) -> int:
        return self._update_payload("state", raw_json)

    def update_level(self, raw_json: str) -> int:
        return self._update_payload("level", raw_json)

    def get_state(self) -> dict[str, Any] | None:
        with self._lock:
            return self._state.data

    def get_level(self) -> dict[str, Any] | None:
        with self._lock:
            return self._level.data

    def get_state_payload(self) -> LivePayload:
        with self._lock:
            return LivePayload(
                data=self._state.data,
                raw_json=self._state.raw_json,
                received_at=self._state.received_at,
                count=self._state.count,
            )

    def get_level_payload(self) -> LivePayload:
        with self._lock:
            return LivePayload(
                data=self._level.data,
                raw_json=self._level.raw_json,
                received_at=self._level.received_at,
                count=self._level.count,
            )

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            level = self._level
            return {
                "state_count": state.count,
                "level_count": level.count,
                "last_state_at": state.received_at,
                "last_level_at": level.received_at,
                "state_summary": self._summarize_state(state.data),
                "level_summary": self._summarize_level(level.data),
            }

    def _update_payload(self, payload_name: str, raw_json: str) -> int:
        parsed = json.loads(raw_json)
        received_at = time.time()

        with self._lock:
            if payload_name == "state":
                self._state = LivePayload(
                    data=parsed,
                    raw_json=raw_json,
                    received_at=received_at,
                    count=self._state.count + 1,
                )
                payload = self._state
            else:
                self._level = LivePayload(
                    data=parsed,
                    raw_json=raw_json,
                    received_at=received_at,
                    count=self._level.count + 1,
                )
                payload = self._level

        if self.output_dir is not None:
            file_name = f"latest_{payload_name}.json"
            (self.output_dir / file_name).write_text(raw_json, encoding="utf-8")

        return payload.count

    def _summarize_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {"players": 0, "objects": 0}
        return {
            "players": len(state.get("players", [])),
            "objects": len(state.get("objects", [])),
        }

    def _summarize_level(self, level: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(level, dict):
            return {"layout": 0, "player_starts": 0}
        return {
            "layout": len(level.get("layout", [])),
            "player_starts": len(level.get("player_starts", [])),
        }


class LiveStateHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def zero_input_state() -> dict[str, Any]:
    return {
        "move_x": 0.0,
        "move_y": 0.0,
        "carry": False,
        "interact": False,
    }


def normalize_input_state(input_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(input_state, dict):
        return zero_input_state()

    return {
        "move_x": float(input_state.get("move_x", 0.0)),
        "move_y": float(input_state.get("move_y", 0.0)),
        "carry": bool(input_state.get("carry", False)),
        "interact": bool(input_state.get("interact", False)),
    }


def normalize_move_vector(move_x: float, move_y: float) -> tuple[float, float]:
    magnitude = math.hypot(move_x, move_y)
    if magnitude <= 0.0:
        return 0.0, 0.0

    return max(min(move_x / magnitude, 1.0), -1.0), max(min(move_y / magnitude, 1.0), -1.0)


def transform_move_vector(
    move_x: float,
    move_y: float,
    *,
    invert_x_axis: bool = LIVE_INVERT_STICK_X,
    invert_y_axis: bool = LIVE_INVERT_STICK_Y,
    swap_axes: bool = LIVE_SWAP_STICK_AXES,
) -> tuple[float, float]:
    transformed_x = move_x
    transformed_y = move_y

    if swap_axes:
        transformed_x, transformed_y = transformed_y, transformed_x
    if invert_x_axis:
        transformed_x = -transformed_x
    if invert_y_axis:
        transformed_y = -transformed_y

    return transformed_x, transformed_y


class Xbox360GamepadAdapter:
    def __init__(
        self,
        player_num: int,
        button_press_duration: float = 0.06,
        *,
        button_repeat_interval: float = 0.18,
    ) -> None:
        self.player_num = player_num
        self.button_press_duration = max(button_press_duration, 0.01)
        self.button_repeat_interval = max(button_repeat_interval, self.button_press_duration)

        try:
            import vgamepad as vg
        except Exception as error:
            raise RuntimeError("vgamepad is required for live controller output.") from error

        self._vg = vg
        try:
            self._gamepad = vg.VX360Gamepad()
        except Exception as error:
            raise RuntimeError(
                "Failed to create a virtual Xbox 360 controller. Ensure ViGEmBus is installed."
            ) from error

        self._left_stick = (0.0, 0.0)
        self._carry_signal = False
        self._interact_signal = False
        self._carry_pressed = False
        self._interact_pressed = False
        self._carry_release_at = 0.0
        self._interact_release_at = 0.0
        self._next_carry_press_at = 0.0
        self._next_interact_press_at = 0.0
        self.clear()

    def apply_input_state(
        self,
        input_state: dict[str, Any] | None,
        now: float,
        facing: list[float] | tuple[float, float] | None = None,
    ) -> None:
        normalized_input = normalize_input_state(input_state)
        move_x, move_y = self._transform_move_vector(normalized_input["move_x"], normalized_input["move_y"])
        move_x, move_y = normalize_move_vector(move_x, move_y)

        carry_signal = normalized_input["carry"]
        interact_signal = normalized_input["interact"]
        if (carry_signal or interact_signal) and move_x == 0.0 and move_y == 0.0:
            move_x, move_y = self._get_action_facing_stick(facing)

        changed = self._set_left_stick(move_x, move_y)

        if carry_signal and (not self._carry_signal or now >= self._next_carry_press_at):
            changed = self._press_button("carry", now) or changed
        if interact_signal and (not self._interact_signal or now >= self._next_interact_press_at):
            changed = self._press_button("interact", now) or changed

        self._carry_signal = carry_signal
        self._interact_signal = interact_signal
        changed = self.tick(now) or changed
        if changed:
            self._gamepad.update()

    def tick(self, now: float) -> bool:
        changed = False
        if self._carry_pressed and now >= self._carry_release_at:
            self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self._carry_pressed = False
            changed = True
        if self._interact_pressed and now >= self._interact_release_at:
            self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            self._interact_pressed = False
            changed = True
        if changed:
            self._gamepad.update()
        return changed

    def pulse_carry(self, now: float) -> None:
        if not self._carry_pressed:
            self._press_button("carry", now)
            self._gamepad.update()

    def clear(self) -> None:
        self._left_stick = (0.0, 0.0)
        self._carry_signal = False
        self._interact_signal = False
        self._carry_pressed = False
        self._interact_pressed = False
        self._carry_release_at = 0.0
        self._interact_release_at = 0.0
        self._next_carry_press_at = 0.0
        self._next_interact_press_at = 0.0
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self._gamepad.update()

    def _set_left_stick(self, move_x: float, move_y: float) -> bool:
        next_stick = (round(move_x, 4), round(move_y, 4))
        if next_stick == self._left_stick:
            return False
        self._left_stick = next_stick
        self._gamepad.left_joystick_float(x_value_float=next_stick[0], y_value_float=next_stick[1])
        return True

    def _press_button(self, button_name: str, now: float) -> bool:
        if button_name == "carry":
            self._gamepad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self._carry_pressed = True
            self._carry_release_at = now + self.button_press_duration
            self._next_carry_press_at = now + self.button_repeat_interval
            return True

        self._gamepad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self._interact_pressed = True
        self._interact_release_at = now + self.button_press_duration
        self._next_interact_press_at = now + self.button_repeat_interval
        return True

    def _transform_move_vector(self, move_x: float, move_y: float) -> tuple[float, float]:
        return transform_move_vector(move_x, move_y)

    def _get_action_facing_stick(
        self,
        facing: list[float] | tuple[float, float] | None,
    ) -> tuple[float, float]:
        if not isinstance(facing, (list, tuple)) or len(facing) != 2:
            return 0.0, 0.0
        facing_x = float(facing[0])
        facing_y = float(facing[1])
        facing_x, facing_y = normalize_move_vector(facing_x, facing_y)
        if facing_x == 0.0 and facing_y == 0.0:
            return 0.0, 0.0
        return facing_x * ACTION_FACING_STICK_STRENGTH, facing_y * ACTION_FACING_STICK_STRENGTH


class LiveAgentRunner:
    def __init__(
        self,
        store: LiveStateStore,
        player_agents: dict[int, str | dict[str, object]],
        *,
        button_press_duration: float = 0.06,
        button_repeat_interval: float = 0.18,
        stale_state_timeout: float = 0.5,
        menu_join_interval: float = 0.5,
        tutorial_clear_duration: float = DEFAULT_TUTORIAL_CLEAR_DURATION,
        trace_output_path: Path | None = None,
        controller_factory: Callable[[int, float], Any] | None = None,
    ) -> None:
        self.store = store
        self.player_agents = dict(player_agents)
        self.button_press_duration = max(button_press_duration, 0.01)
        self.button_repeat_interval = max(button_repeat_interval, self.button_press_duration)
        self.stale_state_timeout = max(stale_state_timeout, 0.1)
        self.menu_join_interval = max(menu_join_interval, 0.1)
        self.tutorial_clear_duration = max(tutorial_clear_duration, 0.0)
        self.trace_output_path = trace_output_path
        self.controller_factory = controller_factory or (
            lambda player_num, duration: Xbox360GamepadAdapter(
                player_num,
                duration,
                button_repeat_interval=self.button_repeat_interval,
            )
        )
        self.agents: dict[int, Any] = {}
        self.controllers: dict[int, Any] = {}
        self.last_input_states: dict[int, dict[str, Any]] = {}
        self.state_player_numbers: dict[int, int] = {}
        self.last_processed_state_count = 0
        self.last_processed_at: float | None = None
        self._is_cleared = False
        self._menu_join_active = False
        self._next_menu_join_at = 0.0
        self._players_loaded = False
        self._dirty_dish_seen = False
        self._tutorial_clear_until = 0.0
        self._next_tutorial_clear_at = 0.0
        self._last_error_text: str | None = None
        self._last_error_time = 0.0

        if self.trace_output_path is not None:
            self.trace_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_output_path.write_text("", encoding="utf-8")

        highest_player_num = max(max(self.player_agents.keys(), default=-1), 1)
        for player_num in range(highest_player_num + 1):
            self.controllers[player_num] = self.controller_factory(player_num, self.button_press_duration)
            agent_info = self.player_agents.get(player_num)
            if agent_info is None:
                continue

            agent_id = get_agent_id(agent_info)
            if agent_id == "human":
                raise ValueError("Human agents are not supported in the live bridge.")
            state_player_num = self._map_state_player_num(player_num)
            self.state_player_numbers[player_num] = state_player_num
            self.agents[player_num] = create_agent(agent_info, state_player_num)

    def tick(self) -> None:
        if not self.controllers:
            return

        now = time.monotonic()
        wall_time = time.time()
        for controller in self.controllers.values():
            controller.tick(now)

        state_payload = self.store.get_state_payload()
        state = state_payload.data
        if not isinstance(state, dict):
            self._reset_tutorial_state()
            self._exit_menu_join_mode()
            self.clear_controllers()
            return

        if state_payload.received_at is None or wall_time - state_payload.received_at > self.stale_state_timeout:
            self._reset_tutorial_state()
            self._exit_menu_join_mode()
            self.clear_controllers()
            return

        players = state.get("players", [])
        player_count = len(players) if isinstance(players, list) else 0
        if player_count == 0:
            self._reset_tutorial_state()
            self.last_processed_state_count = state_payload.count
            self.last_processed_at = None
            self._tick_menu_join(now)
            return

        self._exit_menu_join_mode()
        self._update_tutorial_state(state, now)

        if not self._state_is_ready(state):
            self.last_processed_state_count = state_payload.count
            self.last_processed_at = None
            self._apply_zero_inputs(now)
            self._tick_tutorial_clear(now)
            return

        if state_payload.count == self.last_processed_state_count:
            self._tick_tutorial_clear(now)
            return

        delta_time = 1.0 / 60.0 if self.last_processed_at is None else max(min(now - self.last_processed_at, 0.25), 1.0 / 240.0)
        self.last_processed_at = now
        self.last_processed_state_count = state_payload.count
        self._is_cleared = False

        active_players = set()
        for player_num, agent in self.agents.items():
            try:
                input_state = normalize_input_state(agent.update(delta_time, state))
                self.controllers[player_num].apply_input_state(
                    input_state,
                    now,
                    facing=self._get_state_player_facing(state, player_num),
                )
                self.last_input_states[player_num] = input_state
                active_players.add(player_num)
            except Exception as error:
                self.controllers[player_num].clear()
                self.last_input_states[player_num] = zero_input_state()
                self._log_error_once(f"agent player_{player_num}: {error}")

        for player_num, controller in self.controllers.items():
            if player_num not in active_players:
                controller.clear()

        self._write_trace_event(state, state_payload.count, wall_time)
        self._tick_tutorial_clear(now)

    def clear_controllers(self) -> None:
        if self._is_cleared:
            return
        for controller in self.controllers.values():
            controller.clear()
        self.last_input_states = {player_num: zero_input_state() for player_num in self.agents}
        self._is_cleared = True

    def shutdown(self) -> None:
        self._exit_menu_join_mode()
        self.clear_controllers()

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "players": {
                f"player_{player_num + 1}": get_agent_id(agent_info)
                for player_num, agent_info in sorted(self.player_agents.items())
            },
            "last_processed_state_count": self.last_processed_state_count,
            "last_inputs": self.last_input_states,
            "menu_join_active": self._menu_join_active,
            "tutorial_clear_active": self._tutorial_clear_until > time.monotonic(),
            "dirty_dish_seen": self._dirty_dish_seen,
            "player_mapping": {
                f"controller_{player_num + 1}": f"player_{state_player_num}"
                for player_num, state_player_num in sorted(self.state_player_numbers.items())
            },
            "button_timing": {
                "press_duration": self.button_press_duration,
                "repeat_interval": self.button_repeat_interval,
            },
            "trace_output_path": str(self.trace_output_path) if self.trace_output_path is not None else None,
            "agent_debug": {
                f"player_{player_num + 1}": {
                    "state_player_id": f"player_{self.state_player_numbers.get(player_num, player_num)}",
                    "goal": getattr(agent, "current_goal", None),
                    "target_id": getattr(agent, "current_target_id", None),
                }
                for player_num, agent in sorted(self.agents.items())
            },
        }

    def _state_is_ready(self, state: dict[str, Any]) -> bool:
        players = state.get("players", [])
        objects = state.get("objects", [])
        return isinstance(players, list) and len(players) >= 2 and isinstance(objects, list) and len(objects) > 0

    def _log_error_once(self, message: str) -> None:
        now = time.monotonic()
        if message == self._last_error_text and now - self._last_error_time < 2.0:
            return
        self._last_error_text = message
        self._last_error_time = now
        print(f"live-agent error: {message}")

    def _apply_zero_inputs(self, now: float) -> None:
        for player_num, controller in self.controllers.items():
            idle_input = zero_input_state()
            if hasattr(controller, "apply_input_state"):
                controller.apply_input_state(idle_input, now, facing=None)
            else:
                controller.clear()
            if player_num in self.agents:
                self.last_input_states[player_num] = idle_input
        self._is_cleared = False

    def _tick_menu_join(self, now: float) -> None:
        if not self._menu_join_active:
            self._menu_join_active = True
            self._next_menu_join_at = 0.0
            for controller in self.controllers.values():
                controller.clear()
            self.last_input_states = {player_num: zero_input_state() for player_num in self.agents}
            self._is_cleared = False

        if now < self._next_menu_join_at:
            return

        for controller in self.controllers.values():
            if hasattr(controller, "pulse_carry"):
                controller.pulse_carry(now)
            else:
                controller.apply_input_state({"carry": True}, now)
        self._next_menu_join_at = now + self.menu_join_interval

    def _update_tutorial_state(self, state: dict[str, Any], now: float) -> None:
        if not self._players_loaded:
            self._players_loaded = True
            self._start_tutorial_clear_window(now)

        if not self._dirty_dish_seen and self._state_has_dirty_dishes(state):
            self._dirty_dish_seen = True
            self._start_tutorial_clear_window(now)

    def _start_tutorial_clear_window(self, now: float) -> None:
        if self.tutorial_clear_duration <= 0.0:
            return
        self._tutorial_clear_until = max(self._tutorial_clear_until, now + self.tutorial_clear_duration)
        if self._next_tutorial_clear_at <= 0.0:
            self._next_tutorial_clear_at = now
        else:
            self._next_tutorial_clear_at = min(self._next_tutorial_clear_at, now)

    def _tick_tutorial_clear(self, now: float) -> None:
        if self._tutorial_clear_until <= 0.0:
            return
        if now > self._tutorial_clear_until:
            self._tutorial_clear_until = 0.0
            self._next_tutorial_clear_at = 0.0
            return
        if now < self._next_tutorial_clear_at:
            return

        for controller in self.controllers.values():
            if hasattr(controller, "pulse_carry"):
                controller.pulse_carry(now)
            else:
                controller.apply_input_state({"carry": True}, now)
        self._next_tutorial_clear_at = now + self.menu_join_interval

    def _state_has_dirty_dishes(self, state: dict[str, Any]) -> bool:
        objects = state.get("objects", [])
        if not isinstance(objects, list):
            return False
        for obj in objects:
            if self._is_real_dirty_dish_object(obj):
                return True
        return False

    def _is_real_dirty_dish_object(self, obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False

        object_name = obj.get("name")
        if object_name not in DIRTY_DISH_OBJECT_NAMES:
            return False

        if object_name == "stacked_dirty_plates":
            try:
                return int(obj.get("plate_count", 0)) > 0
            except (TypeError, ValueError):
                return False

        return True

    def _reset_tutorial_state(self) -> None:
        self._players_loaded = False
        self._dirty_dish_seen = False
        self._tutorial_clear_until = 0.0
        self._next_tutorial_clear_at = 0.0

    def _get_state_player_facing(
        self,
        state: dict[str, Any],
        player_num: int,
    ) -> list[float] | None:
        players = state.get("players", [])
        if not isinstance(players, list):
            return None

        state_player_num = self.state_player_numbers.get(player_num, player_num)
        player_id = f"player_{state_player_num}"
        for player in players:
            if not isinstance(player, dict) or player.get("id") != player_id:
                continue
            facing = player.get("facing")
            if isinstance(facing, list) and len(facing) == 2:
                return facing
            return None
        return None

    def _map_state_player_num(self, controller_player_num: int) -> int:
        if not LIVE_SWAP_AGENT_STATE_PLAYERS:
            return controller_player_num
        if controller_player_num == 0:
            return 1
        if controller_player_num == 1:
            return 0
        return controller_player_num

    def _write_trace_event(self, state: dict[str, Any], state_count: int, wall_time: float) -> None:
        if self.trace_output_path is None:
            return

        players_by_id = {
            player.get("id"): player
            for player in state.get("players", [])
            if isinstance(player, dict) and player.get("id") is not None
        }
        objects_by_id = {
            obj.get("id"): obj
            for obj in state.get("objects", [])
            if isinstance(obj, dict) and obj.get("id") is not None
        }

        event = {
            "timestamp": wall_time,
            "state_count": state_count,
            "player_mapping": {
                f"controller_{player_num + 1}": f"player_{state_player_num}"
                for player_num, state_player_num in sorted(self.state_player_numbers.items())
            },
            "players": {},
        }

        for player_num, agent in sorted(self.agents.items()):
            state_player_num = self.state_player_numbers.get(player_num, player_num)
            player_id = f"player_{state_player_num}"
            player_state = players_by_id.get(player_id, {})
            input_state = self.last_input_states.get(player_num, zero_input_state())
            transformed_move = transform_move_vector(
                float(input_state.get("move_x", 0.0)),
                float(input_state.get("move_y", 0.0)),
            )
            normalized_move = normalize_move_vector(*transformed_move)
            target_id = getattr(agent, "current_target_id", None)
            target_state = players_by_id.get(target_id) or objects_by_id.get(target_id) or {}
            player_position = player_state.get("position")
            target_position = target_state.get("position")
            distance_to_target = None
            if isinstance(player_position, list) and isinstance(target_position, list):
                distance_to_target = math.hypot(
                    float(target_position[0]) - float(player_position[0]),
                    float(target_position[1]) - float(player_position[1]),
                )

            event["players"][f"player_{player_num + 1}"] = {
                "controller_slot": player_num + 1,
                "state_player_id": player_id,
                "agent_id": get_agent_id(self.player_agents[player_num]),
                "goal": getattr(agent, "current_goal", None),
                "target_id": target_id,
                "target_name": target_state.get("name"),
                "player_position": player_position,
                "target_position": target_position,
                "distance_to_target": distance_to_target,
                "held_object_id": player_state.get("held_object_id"),
                "held_object_name": player_state.get("held_object_name"),
                "input_state": input_state,
                "transformed_move": {
                    "move_x": transformed_move[0],
                    "move_y": transformed_move[1],
                },
                "normalized_stick": {
                    "move_x": normalized_move[0],
                    "move_y": normalized_move[1],
                },
                "state_facing": player_state.get("facing"),
            }

        with self.trace_output_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event))
            trace_file.write("\n")

    def _exit_menu_join_mode(self) -> None:
        if not self._menu_join_active:
            return
        self._menu_join_active = False
        self._next_menu_join_at = 0.0
        for controller in self.controllers.values():
            controller.clear()


def create_live_state_handler(store: LiveStateStore) -> type[BaseHTTPRequestHandler]:
    class LiveStateRequestHandler(BaseHTTPRequestHandler):
        server_version = "RavioliLiveState/1.0"

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")

            try:
                if self.path == "/state":
                    count = store.update_state(raw_body)
                    self._write_json_response(HTTPStatus.OK, {"ok": True, "kind": "state", "count": count})
                    return

                if self.path == "/level":
                    count = store.update_level(raw_body)
                    self._write_json_response(HTTPStatus.OK, {"ok": True, "kind": "level", "count": count})
                    return

                self._write_json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown_path"})
            except json.JSONDecodeError as error:
                self._write_json_response(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {error}"})
            except Exception as error:
                self._write_json_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(error)})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json_response(HTTPStatus.OK, {"ok": True})
                return

            if self.path == "/snapshot":
                self._write_json_response(HTTPStatus.OK, store.get_snapshot())
                return

            if self.path == "/state":
                state = store.get_state()
                if state is None:
                    self._write_json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "no_state"})
                    return
                self._write_json_response(HTTPStatus.OK, state)
                return

            if self.path == "/level":
                level = store.get_level()
                if level is None:
                    self._write_json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "no_level"})
                    return
                self._write_json_response(HTTPStatus.OK, level)
                return

            self._write_json_response(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown_path"})

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _write_json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            response_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

    return LiveStateRequestHandler


def start_live_state_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: Path | None = None,
) -> tuple[LiveStateHttpServer, LiveStateStore]:
    store = LiveStateStore(output_dir=output_dir)
    handler = create_live_state_handler(store)
    server = LiveStateHttpServer((host, port), handler)
    return server, store
