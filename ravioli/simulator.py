import json
from pathlib import Path

import pyray as rl


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LEVEL_PATH = Path(__file__).with_name("level_1_1.json")
WORLD_SCALE = 50.0
OBJECT_SIZE = WORLD_SCALE * 1.25
PLAYER_START_RADIUS = WORLD_SCALE * 0.35
LABEL_FONT_SIZE = 10
CAMERA_PADDING_PIXELS = 100
CAMERA_MIN_ZOOM = 0.05
CAMERA_MAX_ZOOM = 10.0
CAMERA_ZOOM_STEP = 0.1

CATEGORY_COLORS = {
    "tabletop": rl.LIGHTGRAY,
    "chopping_board": rl.BROWN,
    "stove": rl.RED,
    "dispenser": rl.DARKGREEN,
    "delivery_station": rl.GOLD,
    "plate_return": rl.SKYBLUE,
    "plate_return_station": rl.BLUE,
    "drying_rack": rl.DARKBLUE,
    "sink": rl.DARKBLUE,
    "fire_extinguisher": rl.MAROON,
    "pot": rl.DARKGRAY,
    "plate": rl.WHITE,
}
PLAYER_START_COLOR = rl.MAGENTA


def load_level_data(path: Path) -> tuple[list[dict], list[list[float]]]:
    with path.open("r", encoding="utf-8") as file:
        level_data = json.load(file)

    if "layout" in level_data:
        return level_data["layout"], level_data.get("player_starts", [])

    if "layout_objects" in level_data:
        return level_data["layout_objects"], []

    if "raw_layout_objects" in level_data:
        raw_layout_objects = level_data["raw_layout_objects"]
        if isinstance(raw_layout_objects, str):
            return json.loads(raw_layout_objects), []
        return raw_layout_objects, []

    raise ValueError(f"No layout objects found in {path}")


def to_world_xy(position: list[float]) -> rl.Vector2:
    return rl.Vector2(position[0] * WORLD_SCALE, -position[1] * WORLD_SCALE)


def get_layout_object_position(layout_object: dict) -> list[float]:
    if "position" in layout_object:
        return layout_object["position"]

    world_position = layout_object["world_position"]
    return [world_position[0], world_position[2]]


def collect_world_points(layout_objects: list[dict], player_starts: list[list[float]]) -> list[rl.Vector2]:
    points = [to_world_xy(get_layout_object_position(layout_object)) for layout_object in layout_objects]
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


def get_object_color(layout_object: dict) -> rl.Color:
    object_type = layout_object.get("type", layout_object.get("category"))
    return CATEGORY_COLORS.get(object_type, rl.GRAY)


def draw_layout_object(layout_object: dict) -> None:
    world_position = get_layout_object_position(layout_object)
    world_xy = to_world_xy(world_position)
    rect = rl.Rectangle(
        world_xy.x - (OBJECT_SIZE / 2),
        world_xy.y - (OBJECT_SIZE / 2),
        OBJECT_SIZE,
        OBJECT_SIZE,
    )

    color = get_object_color(layout_object)
    rl.draw_rectangle_rec(rect, color)
    rl.draw_rectangle_lines_ex(rect, 2.0, rl.BLACK)

    label = layout_object.get("name", layout_object["type"])
    ingredient = layout_object.get("ingredient")
    if ingredient:
        label = f"{label}:{ingredient}"
    # rl.draw_text(
    #     label,
    #     int(world_xy.x - (OBJECT_SIZE / 2)),
    #     int(world_xy.y - (OBJECT_SIZE * 1.2)),
    #     LABEL_FONT_SIZE,
    #     rl.BLACK,
    # )


def draw_player_start(player_start: list[float], index: int) -> None:
    world_xy = to_world_xy(player_start)

    rl.draw_circle_v(world_xy, PLAYER_START_RADIUS, PLAYER_START_COLOR)
    rl.draw_circle_lines(int(world_xy.x), int(world_xy.y), PLAYER_START_RADIUS, rl.BLACK)
    # rl.draw_text(f"P{index + 1}", int(world_xy.x - 0.15), int(world_xy.y - 0.1), 16, rl.BLACK)


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

    layout_objects, player_starts = load_level_data(LEVEL_PATH)
    camera = build_camera(layout_objects, player_starts)

    while not rl.window_should_close():
        update_camera(camera, layout_objects, player_starts)

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_2d(camera)

        for layout_object in layout_objects:
            draw_layout_object(layout_object)

        for index, player_start in enumerate(player_starts):
            draw_player_start(player_start, index)

        rl.end_mode_2d()
        rl.draw_text("Mouse wheel: zoom | Middle drag: pan | R: refit camera", 20, 20, 20, rl.DARKGRAY)

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
