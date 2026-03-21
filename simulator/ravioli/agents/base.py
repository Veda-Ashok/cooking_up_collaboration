class InputState:
    def __init__(self, player_num: int) -> None:
        self.player_num = player_num
        self.move_x = 0.0
        self.move_y = 0.0
        self.interact = False
        self.carry = False


class Agent:
    def __init__(self, player_num: int) -> None:
        self.player_num = player_num

    def update(self, delta_time: float, state: dict) -> InputState:
        pass
