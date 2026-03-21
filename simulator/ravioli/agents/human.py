import pyray as rl

from ravioli.agents.base import Agent, InputState

class HumanAgent(Agent):
    def update(self, delta_time: float, state: dict) -> InputState:
        input_state = InputState(self.player_num)
        if self.player_num == 0:
            input_state.move_x = (rl.is_key_down(rl.KEY_D) - rl.is_key_down(rl.KEY_A))
            input_state.move_y = (rl.is_key_down(rl.KEY_S) - rl.is_key_down(rl.KEY_W))
        else:
            input_state.move_x = (rl.is_key_down(rl.KEY_RIGHT) - rl.is_key_down(rl.KEY_LEFT))
            input_state.move_y = (rl.is_key_down(rl.KEY_DOWN) - rl.is_key_down(rl.KEY_UP))
        input_state.interact = rl.is_key_pressed(rl.KEY_LEFT_CONTROL) if self.player_num == 0 else rl.is_key_pressed(rl.KEY_RIGHT_CONTROL)
        input_state.carry = rl.is_key_pressed(rl.KEY_SPACE) if self.player_num == 0 else rl.is_key_pressed(rl.KEY_RIGHT_SHIFT)
        return input_state
