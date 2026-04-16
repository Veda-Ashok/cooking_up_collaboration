import pyray as rl

from .base import Agent


class HumanAgent(Agent):
    KEYBOARD_SCHEMES = {
        "wasd": {
            "left": rl.KEY_A,
            "right": rl.KEY_D,
            "up": rl.KEY_W,
            "down": rl.KEY_S,
            "interact": rl.KEY_LEFT_CONTROL,
            "carry": rl.KEY_SPACE,
        },
        "arrows": {
            "left": rl.KEY_LEFT,
            "right": rl.KEY_RIGHT,
            "up": rl.KEY_UP,
            "down": rl.KEY_DOWN,
            "interact": rl.KEY_RIGHT_CONTROL,
            "carry": rl.KEY_RIGHT_SHIFT,
        },
    }
    GAMEPAD_DEADZONE = 0.1
    GAMEPAD_AXIS_X = rl.GAMEPAD_AXIS_LEFT_X
    GAMEPAD_AXIS_Y = rl.GAMEPAD_AXIS_LEFT_Y
    GAMEPAD_DPAD_LEFT = rl.GAMEPAD_BUTTON_LEFT_FACE_LEFT
    GAMEPAD_DPAD_RIGHT = rl.GAMEPAD_BUTTON_LEFT_FACE_RIGHT
    GAMEPAD_DPAD_UP = rl.GAMEPAD_BUTTON_LEFT_FACE_UP
    GAMEPAD_DPAD_DOWN = rl.GAMEPAD_BUTTON_LEFT_FACE_DOWN
    GAMEPAD_INTERACT_BUTTON = rl.GAMEPAD_BUTTON_RIGHT_FACE_LEFT
    GAMEPAD_CARRY_BUTTON = rl.GAMEPAD_BUTTON_RIGHT_FACE_DOWN

    def __init__(self, player_num: int, input_type: dict | None = None) -> None:
        super().__init__(player_num)
        self.input_type = self.normalize_input_type(input_type, player_num)

    def normalize_input_type(self, input_type: dict | None, player_num: int) -> dict:
        if input_type is None:
            return self.default_input_type(player_num)

        input_kind = str(input_type.get("type", "keyboard")).lower()
        if input_kind == "keyboard":
            scheme = str(input_type.get("scheme", self.default_input_type(player_num)["scheme"])).lower()
            if scheme not in self.KEYBOARD_SCHEMES:
                raise ValueError(f"Unknown keyboard input scheme: {scheme}")
            return {"type": "keyboard", "scheme": scheme}

        if input_kind == "gamepad":
            gamepad_number = int(input_type.get("gamepad_number", input_type.get("gamepad", 0)))
            return {"type": "gamepad", "gamepad_number": gamepad_number}

        raise ValueError(f"Unknown human input type: {input_kind}")

    @staticmethod
    def default_input_type(player_num: int) -> dict:
        scheme = "wasd" if player_num == 0 else "arrows"
        return {"type": "keyboard", "scheme": scheme}

    def get_keyboard_input_state(self, scheme: str) -> dict:
        controls = self.KEYBOARD_SCHEMES[scheme]
        return {
            "move_x": float(rl.is_key_down(controls["right"]) - rl.is_key_down(controls["left"])),
            "move_y": float(rl.is_key_down(controls["down"]) - rl.is_key_down(controls["up"])),
            "interact": rl.is_key_pressed(controls["interact"]),
            "carry": rl.is_key_pressed(controls["carry"]),
        }

    def get_gamepad_input_state(self, gamepad_number: int) -> dict:
        if not rl.is_gamepad_available(gamepad_number):
            return {"move_x": 0.0, "move_y": 0.0, "interact": False, "carry": False}

        stick_x = rl.get_gamepad_axis_movement(gamepad_number, self.GAMEPAD_AXIS_X)
        stick_y = rl.get_gamepad_axis_movement(gamepad_number, self.GAMEPAD_AXIS_Y)

        if abs(stick_x) < self.GAMEPAD_DEADZONE:
            stick_x = 0.0
        if abs(stick_y) < self.GAMEPAD_DEADZONE:
            stick_y = 0.0

        dpad_x = float(
            rl.is_gamepad_button_down(gamepad_number, self.GAMEPAD_DPAD_RIGHT)
            - rl.is_gamepad_button_down(gamepad_number, self.GAMEPAD_DPAD_LEFT)
        )
        dpad_y = float(
            rl.is_gamepad_button_down(gamepad_number, self.GAMEPAD_DPAD_DOWN)
            - rl.is_gamepad_button_down(gamepad_number, self.GAMEPAD_DPAD_UP)
        )

        move_x = dpad_x if dpad_x != 0.0 else stick_x
        move_y = dpad_y if dpad_y != 0.0 else stick_y

        return {
            "move_x": move_x,
            "move_y": move_y,
            "interact": rl.is_gamepad_button_pressed(gamepad_number, self.GAMEPAD_INTERACT_BUTTON),
            "carry": rl.is_gamepad_button_pressed(gamepad_number, self.GAMEPAD_CARRY_BUTTON),
        }

    def update(self, delta_time: float, state: dict) -> dict:
        if self.input_type["type"] == "keyboard":
            return self.get_keyboard_input_state(self.input_type["scheme"])
        if self.input_type["type"] == "gamepad":
            return self.get_gamepad_input_state(self.input_type["gamepad_number"])
        raise ValueError(f"Unsupported human input type: {self.input_type['type']}")
