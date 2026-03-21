import json
from pathlib import Path

import pyray as rl

from game import Level, TABLETOP_SIZE, WORLD_SCALE


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LEVEL_PATH = Path(__file__).with_name("level_1_1.json")
OBJECT_SIZE = WORLD_SCALE * TABLETOP_SIZE.x
CAMERA_PADDING_PIXELS = 100
CAMERA_MIN_ZOOM = 0.05
CAMERA_MAX_ZOOM = 10.0
CAMERA_ZOOM_STEP = 0.1


def load_level_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_world_xy(position: list[float]) -> rl.Vector2:
    return rl.Vector2(position[0] * WORLD_SCALE, -position[1] * WORLD_SCALE)


def collect_world_points(layout_objects: list[dict], player_starts: list[list[float]]) -> list[rl.Vector2]:
    points = [to_world_xy(layout_object.get("position", [0.0, 0.0])) for layout_object in layout_objects]
    points.extend(to_world_xy(player_start) for player_start in player_starts)
    return points


def build_camera(layout_objects: list[dict], player_starts: list[list[float]]) -> rl.Camera2D:
    points = collect_world_points(layout_objects, player_starts)
    if not points:
        return rl.Camera2D(
            rl.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
            rl.Vector2(0.0, 0.0),
            0.0,
            1.0,
        )

    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)

    world_width = max(max_x - min_x, OBJECT_SIZE * 2)
    world_height = max(max_y - min_y, OBJECT_SIZE * 2)
    usable_width = max(WINDOW_WIDTH - (2 * CAMERA_PADDING_PIXELS), 1)
    usable_height = max(WINDOW_HEIGHT - (2 * CAMERA_PADDING_PIXELS), 1)
    zoom_x = usable_width / world_width
    zoom_y = usable_height / world_height
    zoom = max(CAMERA_MIN_ZOOM, min(min(zoom_x, zoom_y), CAMERA_MAX_ZOOM))

    return rl.Camera2D(
        rl.Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2),
        rl.Vector2((min_x + max_x) / 2, (min_y + max_y) / 2),
        0.0,
        zoom,
    )


def update_camera(camera: rl.Camera2D, layout_objects: list[dict], player_starts: list[list[float]]) -> None:
    if rl.is_key_pressed(rl.KEY_R):
        fitted_camera = build_camera(layout_objects, player_starts)
        camera.offset = fitted_camera.offset
        camera.target = fitted_camera.target
        camera.rotation = fitted_camera.rotation
        camera.zoom = fitted_camera.zoom

    wheel_move = rl.get_mouse_wheel_move()
    if wheel_move != 0:
        zoom_multiplier = 1.0 + (wheel_move * CAMERA_ZOOM_STEP)
        camera.zoom = max(CAMERA_MIN_ZOOM, min(camera.zoom * zoom_multiplier, CAMERA_MAX_ZOOM))

    if rl.is_mouse_button_down(rl.MOUSE_BUTTON_MIDDLE):
        delta = rl.get_mouse_delta()
        camera.target.x -= delta.x / camera.zoom
        camera.target.y -= delta.y / camera.zoom


def main():
    rl.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Ravioli Simulator")
    rl.set_target_fps(60)

    level_data = load_level_data(LEVEL_PATH)
    level = Level(level_data)
    camera = build_camera(level.layout_objects, level.player_starts)

    while not rl.window_should_close():
        update_camera(camera, level.layout_objects, level.player_starts)
        level.update(rl.get_frame_time())

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_2d(camera)

        level.draw()

        rl.end_mode_2d()
        rl.draw_text(
            "WASD: P1 | Arrow keys: P2 | Mouse wheel: zoom | Middle drag: pan | R: refit camera",
            20,
            20,
            20,
            rl.DARKGRAY,
        )

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
