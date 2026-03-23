import pyray as rl

from ravioli.agents import AGENT_TYPES, get_agent_label
from ravioli.agents.human import HumanAgent
from ravioli.game import Level


class Menu:
    GAMEPAD_AXIS_THRESHOLD = 0.5

    def __init__(self):
        self.player_options = ["player_1", "player_2"]
        self.agent_ids = [agent_type["id"] for agent_type in AGENT_TYPES]
        self.level_files = self.get_level_filenames()
        self.selected_level_index = 0
        self.selected_agents = {player_key: self.agent_ids[0] for player_key in self.player_options}
        self.selected_input_types = {
            player_key: HumanAgent.default_input_type(player_num)
            for player_num, player_key in enumerate(self.player_options)
        }
        if "human" in self.agent_ids:
            self.selected_agents["player_1"] = "human"
            self.selected_agents["player_2"] = "human"
        self.selected_option = self.player_options[0]
        self.gamepad_menu_state: dict[int, dict[str, bool]] = {}

    def get_visible_options(self) -> list[str]:
        options = []
        for player_key in self.player_options:
            options.append(player_key)
            if self.selected_agents[player_key] == "human":
                options.append(f"{player_key}_input")
        options.extend(["level_file", "start", "exit"])
        return options

    def ensure_selected_option_visible(self) -> None:
        visible_options = self.get_visible_options()
        if self.selected_option in visible_options:
            return

        if self.selected_option.endswith("_input"):
            self.selected_option = self.selected_option[:-6]
        if self.selected_option not in visible_options:
            self.selected_option = visible_options[0]

    def draw(self):
        visible_options = self.get_visible_options()
        self.ensure_selected_option_visible()
        connected_gamepads = self.get_connected_gamepads()
        rl.draw_text("Ravioli Simulator", 400, 100, 50, rl.DARKGRAY)
        rl.draw_text("Arrows/WASD/Gamepad to navigate, Enter/Space/Gamepad A to confirm", 210, 150, 24, rl.GRAY)
        rl.draw_text(f"Gamepads attached: {len(connected_gamepads)}", 470, 180, 24, rl.GRAY)
        for i, option in enumerate(visible_options):
            color = rl.RED if option == self.selected_option else rl.GRAY
            if option == "level_file":
                text = f"Level > {self.get_selected_level_file() or 'No levels found'}"
            elif option.endswith("_input"):
                player_key = option[:-6]
                player_number = player_key.split("_")[1]
                text = f"Player {player_number} Input > {self.get_selected_input_label(player_key)}"
            elif option.startswith("player_"):
                player_number = option.split("_")[1]
                text = f"Player {player_number} > {get_agent_label(self.selected_agents[option])}"
            else:
                text = option.capitalize()
            rl.draw_text(text, 470, 250 + i * 60, 30, color)

    def get_level_filenames(self) -> list[str]:
        paths = sorted(Level.LEVELS_DIR.glob("*.json"))
        return [path.name for path in paths]

    def get_selected_level_file(self) -> str | None:
        if not self.level_files:
            return None
        return self.level_files[self.selected_level_index]

    def get_connected_gamepads(self) -> list[int]:
        max_gamepads = getattr(rl, "MAX_GAMEPADS", 8)
        return [gamepad_number for gamepad_number in range(max_gamepads) if rl.is_gamepad_available(gamepad_number)]

    def get_gamepad_direction_state(self, gamepad_number: int) -> dict[str, bool]:
        axis_x = rl.get_gamepad_axis_movement(gamepad_number, rl.GAMEPAD_AXIS_LEFT_X)
        axis_y = rl.get_gamepad_axis_movement(gamepad_number, rl.GAMEPAD_AXIS_LEFT_Y)

        return {
            "up": rl.is_gamepad_button_down(gamepad_number, rl.GAMEPAD_BUTTON_LEFT_FACE_UP) or axis_y < -self.GAMEPAD_AXIS_THRESHOLD,
            "down": rl.is_gamepad_button_down(gamepad_number, rl.GAMEPAD_BUTTON_LEFT_FACE_DOWN) or axis_y > self.GAMEPAD_AXIS_THRESHOLD,
            "left": rl.is_gamepad_button_down(gamepad_number, rl.GAMEPAD_BUTTON_LEFT_FACE_LEFT) or axis_x < -self.GAMEPAD_AXIS_THRESHOLD,
            "right": rl.is_gamepad_button_down(gamepad_number, rl.GAMEPAD_BUTTON_LEFT_FACE_RIGHT) or axis_x > self.GAMEPAD_AXIS_THRESHOLD,
        }

    def get_menu_input(self) -> dict[str, bool]:
        menu_input = {
            "up": rl.is_key_pressed(rl.KEY_UP) or rl.is_key_pressed(rl.KEY_W),
            "down": rl.is_key_pressed(rl.KEY_DOWN) or rl.is_key_pressed(rl.KEY_S),
            "left": rl.is_key_pressed(rl.KEY_LEFT) or rl.is_key_pressed(rl.KEY_A),
            "right": rl.is_key_pressed(rl.KEY_RIGHT) or rl.is_key_pressed(rl.KEY_D),
            "select": rl.is_key_pressed(rl.KEY_ENTER) or rl.is_key_pressed(rl.KEY_SPACE),
        }

        connected_gamepads = self.get_connected_gamepads()
        active_gamepads = set(connected_gamepads)
        stale_gamepads = [gamepad_number for gamepad_number in self.gamepad_menu_state if gamepad_number not in active_gamepads]
        for gamepad_number in stale_gamepads:
            del self.gamepad_menu_state[gamepad_number]

        for gamepad_number in connected_gamepads:
            previous_state = self.gamepad_menu_state.get(
                gamepad_number,
                {"up": False, "down": False, "left": False, "right": False},
            )
            current_state = self.get_gamepad_direction_state(gamepad_number)
            for direction in ("up", "down", "left", "right"):
                menu_input[direction] = menu_input[direction] or (
                    current_state[direction] and not previous_state[direction]
                )
            menu_input["select"] = menu_input["select"] or rl.is_gamepad_button_pressed(
                gamepad_number,
                rl.GAMEPAD_BUTTON_RIGHT_FACE_DOWN,
            )
            self.gamepad_menu_state[gamepad_number] = current_state

        return menu_input

    @staticmethod
    def get_keyboard_scheme_label(scheme: str) -> str:
        if scheme == "wasd":
            return "Keyboard WASD"
        if scheme == "arrows":
            return "Keyboard Arrows"
        return f"Keyboard {scheme.title()}"

    def get_input_choices(self, player_key: str) -> list[tuple[str, dict]]:
        del player_key
        choices = [
            (self.get_keyboard_scheme_label(scheme), {"type": "keyboard", "scheme": scheme})
            for scheme in HumanAgent.KEYBOARD_SCHEMES
        ]
        choices.extend(
            (f"Gamepad {gamepad_number + 1}", {"type": "gamepad", "gamepad_number": gamepad_number})
            for gamepad_number in self.get_connected_gamepads()
        )
        return choices

    def normalize_selected_input_type(self, player_key: str) -> None:
        choices = self.get_input_choices(player_key)
        if not choices:
            return

        selected_input = self.selected_input_types[player_key]
        for _, input_type in choices:
            if input_type == selected_input:
                return

        player_num = self.player_options.index(player_key)
        default_input = HumanAgent.default_input_type(player_num)
        for _, input_type in choices:
            if input_type == default_input:
                self.selected_input_types[player_key] = input_type
                return

        self.selected_input_types[player_key] = choices[0][1]

    def get_selected_input_label(self, player_key: str) -> str:
        self.normalize_selected_input_type(player_key)
        selected_input = self.selected_input_types[player_key]
        if selected_input["type"] == "keyboard":
            return self.get_keyboard_scheme_label(selected_input["scheme"])
        return f"Gamepad {selected_input['gamepad_number'] + 1}"

    def cycle_selection(self, direction: int) -> None:
        option = self.selected_option
        if option == "level_file":
            if self.level_files:
                self.selected_level_index = (self.selected_level_index + direction) % len(self.level_files)
            return

        if option.endswith("_input"):
            player_key = option[:-6]
            choices = self.get_input_choices(player_key)
            if not choices:
                return

            self.normalize_selected_input_type(player_key)
            selected_input = self.selected_input_types[player_key]
            current_index = next(
                (index for index, (_, input_type) in enumerate(choices) if input_type == selected_input),
                0,
            )
            next_index = (current_index + direction) % len(choices)
            self.selected_input_types[player_key] = choices[next_index][1]
            return

        if not option.startswith("player_") or not self.agent_ids:
            return

        current_agent_id = self.selected_agents[option]
        current_index = self.agent_ids.index(current_agent_id)
        next_index = (current_index + direction) % len(self.agent_ids)
        self.selected_agents[option] = self.agent_ids[next_index]
        self.ensure_selected_option_visible()

    def update(self) -> dict:
        self.ensure_selected_option_visible()
        visible_options = self.get_visible_options()
        current_index = visible_options.index(self.selected_option)
        menu_input = self.get_menu_input()
        if menu_input["down"]:
            self.selected_option = visible_options[(current_index + 1) % len(visible_options)]
        elif menu_input["up"]:
            self.selected_option = visible_options[(current_index - 1) % len(visible_options)]
        elif menu_input["right"]:
            self.cycle_selection(1)
        elif menu_input["left"]:
            self.cycle_selection(-1)
        elif menu_input["select"]:
            selected_option = self.selected_option
            if selected_option == "start":
                selected_level_file = self.get_selected_level_file()
                if selected_level_file is not None:
                    player_config = {}
                    for player_key in self.player_options:
                        agent_id = self.selected_agents[player_key]
                        if agent_id == "human":
                            self.normalize_selected_input_type(player_key)
                            player_config[player_key] = {
                                "type": "human",
                                "input_type": dict(self.selected_input_types[player_key]),
                            }
                        else:
                            player_config[player_key] = agent_id
                    return {
                        "level_file": selected_level_file,
                        **player_config,
                    }
            if selected_option == "exit":
                rl.close_window()
        return None
