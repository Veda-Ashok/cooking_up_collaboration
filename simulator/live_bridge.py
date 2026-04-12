import argparse
import time

from pathlib import Path

from ravioli.agents import AGENT_TYPES
from ravioli.live_state import LiveAgentRunner, start_live_state_server


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ingest"
DEFAULT_TRACE_OUTPUT = DEFAULT_OUTPUT_DIR / "live_agent_trace.jsonl"
SERVER_TICK_SECONDS = 0.01


def build_parser() -> argparse.ArgumentParser:
    agent_ids = [agent_type["id"] for agent_type in AGENT_TYPES if agent_type["id"] != "human"]
    parser = argparse.ArgumentParser(description="Receive live Overcooked state over HTTP.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on.")
    parser.add_argument("--player-1", choices=agent_ids, help="Agent to run on player 1.")
    parser.add_argument("--player-2", choices=agent_ids, help="Agent to run on player 2.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write latest_state.json and latest_level.json.",
    )
    parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Keep the latest payloads in memory only.",
    )
    parser.add_argument(
        "--trace-output",
        default=str(DEFAULT_TRACE_OUTPUT),
        help="Write live agent decision traces to this JSONL file. Use an empty string to disable.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="Seconds between status prints.",
    )
    parser.add_argument(
        "--button-press-duration",
        type=float,
        default=0.06,
        help="Seconds to hold A/X button presses when an agent triggers carry or interact.",
    )
    parser.add_argument(
        "--button-repeat-interval",
        type=float,
        default=0.18,
        help="Seconds between repeated A/X pulses while an agent keeps carry or interact held high.",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=0.5,
        help="Seconds after the last state update before controllers are cleared.",
    )
    parser.add_argument(
        "--menu-join-interval",
        type=float,
        default=0.5,
        help="Seconds between automatic A-button join/ready presses while streamed state reports zero players.",
    )
    parser.add_argument(
        "--tutorial-clear-duration",
        type=float,
        default=2.0,
        help="Seconds to keep pulsing A after players first appear and after the first dirty dish appears.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = None if args.no_write_files else Path(args.output_dir)
    trace_output_path = Path(args.trace_output) if args.trace_output else None
    server, store = start_live_state_server(host=args.host, port=args.port, output_dir=output_dir)
    player_agents = {}
    if args.player_1 is not None:
        player_agents[0] = args.player_1
    if args.player_2 is not None:
        player_agents[1] = args.player_2

    live_agent_runner = None
    if player_agents:
        live_agent_runner = LiveAgentRunner(
            store,
            player_agents,
            button_press_duration=args.button_press_duration,
            button_repeat_interval=args.button_repeat_interval,
            stale_state_timeout=args.state_timeout,
            menu_join_interval=args.menu_join_interval,
            tutorial_clear_duration=args.tutorial_clear_duration,
            trace_output_path=trace_output_path,
        )

    print(f"Listening for Overcooked state on http://{args.host}:{args.port}")
    print("POST /state and /level, GET /health and /snapshot")
    if output_dir is not None:
        print(f"Writing latest payloads to {output_dir}")
    if live_agent_runner is not None:
        configured_agents = ", ".join(
            f"player_{player_num + 1}={agent_id}" for player_num, agent_id in sorted(player_agents.items())
        )
        print(f"Live agents enabled: {configured_agents}")
        if trace_output_path is not None:
            print(f"Writing live trace to {trace_output_path}")

    next_status_time = 0.0
    try:
        while True:
            server.timeout = SERVER_TICK_SECONDS
            server.handle_request()
            if live_agent_runner is not None:
                live_agent_runner.tick()
            current_time = time.time()
            if current_time >= next_status_time:
                snapshot = store.get_snapshot()
                status_line = (
                    "status "
                    f"state_count={snapshot['state_count']} "
                    f"level_count={snapshot['level_count']} "
                    f"players={snapshot['state_summary']['players']} "
                    f"objects={snapshot['state_summary']['objects']}"
                )
                if live_agent_runner is not None:
                    live_snapshot = live_agent_runner.get_snapshot()
                    configured_agents = ",".join(
                        f"{player_key}:{agent_id}" for player_key, agent_id in live_snapshot["players"].items()
                    )
                    status_line += (
                        f" live_state_count={live_snapshot['last_processed_state_count']} "
                        f"agents={configured_agents} "
                        f"menu_join_active={int(live_snapshot['menu_join_active'])} "
                        f"tutorial_clear_active={int(live_snapshot['tutorial_clear_active'])} "
                        f"dirty_dish_seen={int(live_snapshot['dirty_dish_seen'])} "
                        f"buttons=hold:{live_snapshot['button_timing']['press_duration']:.2f},"
                        f"repeat:{live_snapshot['button_timing']['repeat_interval']:.2f}"
                    )
                    agent_debug = live_snapshot["agent_debug"]
                    if agent_debug:
                        status_line += " goals=" + ",".join(
                            f"{player_key}->{debug['state_player_id']}:{debug['goal']}@{debug['target_id']}"
                            for player_key, debug in agent_debug.items()
                        )
                print(status_line)
                next_status_time = current_time + max(args.status_interval, 0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if live_agent_runner is not None:
            live_agent_runner.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
