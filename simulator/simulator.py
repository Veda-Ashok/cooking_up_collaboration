import pyray as rl
from pathlib import Path

from ravioli.game import Level
from ravioli.menu import Menu


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def main():
    rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Ravioli Simulator")
    rl.set_target_fps(60)
    GAMECONTROLLER_DB = Path(__file__).with_name("gamecontrollerdb.txt")
    if GAMECONTROLLER_DB.is_file():
        rl.set_gamepad_mappings(GAMECONTROLLER_DB.read_text(encoding="utf-8"))

    menu = Menu()
    level = None

    while not rl.window_should_close():
        if level is None:
            level_info = menu.update()
            if level_info is not None:
                level = Level(
                    level_info,
                    export_state=True,
                    export_every_n_frames=60 / 15,
                )
        else:
            level.update(rl.get_frame_time())

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        if level is None:
            menu.draw()
        else:
            level.draw()

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
