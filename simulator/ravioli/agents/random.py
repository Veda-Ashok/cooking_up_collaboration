import random
from ravioli.agents import Agent, InputState

class RandomAgent(Agent):
    def update(self, delta_time: float, state: dict) -> InputState:
        input_state = InputState(self.player_num)
        input_state.move_x = random.choice([-1.0, 0.0, 1.0])
        input_state.move_y = random.choice([-1.0, 0.0, 1.0])
        input_state.interact = random.choice([True, False])
        input_state.carry = random.choice([True, False])
        return input_state
