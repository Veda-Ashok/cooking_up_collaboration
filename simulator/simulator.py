import argparse
import pyray as rl

from pathlib import Path

from ravioli.agents import AGENT_TYPES
from ravioli.game import Level
from ravioli.menu import Menu


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
EXPORT_EVERY_N_FRAMES = 4
DEFAULT_LEVEL = "level_1_1"


def build_parser() -> argparse.ArgumentParser:
    agent_ids = [agent_type["id"] for agent_type in AGENT_TYPES]
    parser = argparse.ArgumentParser(description="Run the Ravioli simulator.")
    parser.add_argument("--player-1", choices=agent_ids, help="Agent for player 1.")
    parser.add_argument("--player-2", choices=agent_ids, help="Agent for player 2.")
    parser.add_argument("--level", default=DEFAULT_LEVEL, help="Level file to load, with or without the .json suffix.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a window and simulate as fast as possible.")
    return parser


def resolve_level_file(level_arg: str) -> str:
    candidate = level_arg if level_arg.endswith(".json") else f"{level_arg}.json"
    level_path = Level.LEVELS_DIR / candidate
    if not level_path.is_file():
        available_levels = ", ".join(path.stem for path in sorted(Level.LEVELS_DIR.glob("*.json")))
        raise ValueError(f"Unknown level '{level_arg}'. Available levels: {available_levels}")
    return candidate


def build_level_info(args: argparse.Namespace, level_file: str) -> dict[str, str]:
    return {
        "level_file": level_file,
        "player_1": args.player_1,
        "player_2": args.player_2,
    }


def should_skip_menu(args: argparse.Namespace) -> bool:
    return args.player_1 is not None and args.player_2 is not None


def apply_cli_defaults_to_menu(menu: Menu, args: argparse.Namespace, level_file: str) -> None:
    if level_file in menu.level_files:
        menu.selected_level_index = menu.level_files.index(level_file)
    if args.player_1 is not None:
        menu.selected_agents["player_1"] = args.player_1
    if args.player_2 is not None:
        menu.selected_agents["player_2"] = args.player_2


def init_window() -> None:
    rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Ravioli Simulator")
    rl.set_target_fps(TARGET_FPS)
    gamecontroller_db = Path(__file__).with_name("gamecontrollerdb.txt")
    if gamecontroller_db.is_file():
        rl.set_gamepad_mappings(gamecontroller_db.read_text(encoding="utf-8"))


def create_level(level_info: dict[str, str], headless: bool) -> Level:
    return Level(
        level_info,
        export_state=True,
        export_every_n_frames=EXPORT_EVERY_N_FRAMES,
        headless=headless,
        screen_size=(WINDOW_WIDTH, WINDOW_HEIGHT),
    )


def run_headless(level_info: dict[str, str]) -> None:
    level = create_level(level_info, headless=True)
    fixed_delta_time = 1.0 / TARGET_FPS
    try:
        while True:
            level.update(fixed_delta_time)
    except KeyboardInterrupt:
        return


def run_windowed(args: argparse.Namespace, level_file: str) -> None:
    init_window()
    menu = None
    level = None

    if should_skip_menu(args):
        level = create_level(build_level_info(args, level_file), headless=False)
    else:
        menu = Menu()
        apply_cli_defaults_to_menu(menu, args, level_file)

    while not rl.window_should_close():
        if level is None:
            level_info = menu.update()
            if level_info is not None:
                level = create_level(level_info, headless=False)
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


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    try:
        level_file = resolve_level_file(args.level)
    except ValueError as error:
        parser.error(str(error))

    if args.headless and not should_skip_menu(args):
        parser.error("--headless requires both --player-1 and --player-2.")

    if args.headless and (args.player_1 == "human" or args.player_2 == "human"):
        parser.error("--headless does not support human agents.")

    return level_file


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    level_file = validate_args(parser, args)

    if args.headless:
        run_headless(build_level_info(args, level_file))
        return

    run_windowed(args, level_file)


if __name__ == "__main__":
    main()
