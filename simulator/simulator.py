import json
from pathlib import Path

import pyray as rl

from ravioli.game import Level


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LEVEL_PATH = Path(__file__).with_name("level_1_1.json")


def load_level_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Ravioli Simulator")
    rl.set_target_fps(60)

    level_data = load_level_data(LEVEL_PATH)
    level = Level(level_data)

    while not rl.window_should_close():
        level.update(rl.get_frame_time())

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        level.draw()

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
