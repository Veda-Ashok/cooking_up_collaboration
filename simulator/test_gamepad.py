import argparse
import math
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit a repeating Xbox 360 test pattern through vgamepad.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of pattern cycles to run. Use 0 to loop forever.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.6,
        help="Seconds to hold each discrete stick direction.",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=0.2,
        help="Seconds to rest at neutral between pattern steps.",
    )
    parser.add_argument(
        "--button-seconds",
        type=float,
        default=0.15,
        help="Seconds to hold A and X button presses.",
    )
    parser.add_argument(
        "--circle-seconds",
        type=float,
        default=3.0,
        help="Seconds to spend drawing a full circle on the left stick.",
    )
    parser.add_argument(
        "--circle-steps",
        type=int,
        default=48,
        help="Number of stick updates per circle.",
    )
    return parser


def set_stick(gamepad, x_value: float, y_value: float) -> None:
    gamepad.left_joystick_float(
        x_value_float=max(min(x_value, 1.0), -1.0),
        y_value_float=max(min(y_value, 1.0), -1.0),
    )
    gamepad.update()


def press_button(gamepad, button, button_name: str, hold_seconds: float) -> None:
    print(f"button {button_name}")
    gamepad.press_button(button=button)
    gamepad.update()
    time.sleep(hold_seconds)
    gamepad.release_button(button=button)
    gamepad.update()


def reset_gamepad(gamepad, vg) -> None:
    set_stick(gamepad, 0.0, 0.0)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    gamepad.update()


def run_pattern(gamepad, vg, args: argparse.Namespace) -> None:
    directions = (
        ("neutral", 0.0, 0.0),
        ("up", 0.0, 1.0),
        ("right", 1.0, 0.0),
        ("down", 0.0, -1.0),
        ("left", -1.0, 0.0),
        ("up-right", 0.707, 0.707),
        ("down-right", 0.707, -0.707),
        ("down-left", -0.707, -0.707),
        ("up-left", -0.707, 0.707),
    )

    cycle_index = 0
    while args.cycles == 0 or cycle_index < args.cycles:
        cycle_index += 1
        print(f"cycle {cycle_index}")

        for label, x_value, y_value in directions:
            print(f"stick {label}")
            set_stick(gamepad, x_value, y_value)
            time.sleep(args.hold_seconds)
            set_stick(gamepad, 0.0, 0.0)
            time.sleep(args.transition_seconds)

        press_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "A", args.button_seconds)
        time.sleep(args.transition_seconds)
        press_button(gamepad, vg.XUSB_BUTTON.XUSB_GAMEPAD_X, "X", args.button_seconds)
        time.sleep(args.transition_seconds)

        print("stick circle")
        steps = max(args.circle_steps, 4)
        sleep_per_step = max(args.circle_seconds / steps, 0.001)
        for step in range(steps):
            angle = (step / steps) * (2.0 * math.pi)
            set_stick(gamepad, math.cos(angle), math.sin(angle))
            time.sleep(sleep_per_step)

        set_stick(gamepad, 0.0, 0.0)
        time.sleep(args.transition_seconds)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        import vgamepad as vg
    except Exception as error:
        raise RuntimeError("vgamepad is required to run this script.") from error

    try:
        gamepad = vg.VX360Gamepad()
    except Exception as error:
        raise RuntimeError("Failed to create a virtual Xbox 360 controller. Ensure ViGEmBus is installed.") from error

    print("Created virtual Xbox 360 controller. Open `joy.cpl` to inspect the test pattern.")
    print("Press Ctrl+C to stop.")

    reset_gamepad(gamepad, vg)
    try:
        run_pattern(gamepad, vg, args)
    except KeyboardInterrupt:
        pass
    finally:
        reset_gamepad(gamepad, vg)


if __name__ == "__main__":
    main()
