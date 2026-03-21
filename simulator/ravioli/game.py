import math
import pyray as rl

from Box2D import b2CircleShape, b2PolygonShape, b2Vec2, b2World
from ravioli.agents import RandomAgent, HumanAgent, Agent, InputState


WORLD_SCALE = 50.0
TABLETOP_SIZE = b2Vec2(1.25, 1.25)


def world_to_physics(position: list[float]) -> b2Vec2:
    return b2Vec2(position[0], -position[1])


def physics_to_world(position: b2Vec2) -> list[float]:
    return [position.x, -position.y]


def physics_to_screen(position: b2Vec2) -> rl.Vector2:
    return rl.Vector2(position.x * WORLD_SCALE, position.y * WORLD_SCALE)


def draw_centered_rectangle(center: rl.Vector2, width: float, height: float, color: rl.Color) -> None:
    rect = rl.Rectangle(
        center.x - (width / 2),
        center.y - (height / 2),
        width,
        height,
    )
    rl.draw_rectangle_rec(rect, color)
    rl.draw_rectangle_lines_ex(rect, 2.0, rl.BLACK)


def draw_centered_square(center: rl.Vector2, size: float, color: rl.Color) -> None:
    rect = rl.Rectangle(
        center.x - (size / 2),
        center.y - (size / 2),
        size,
        size,
    )
    rl.draw_rectangle_rec(rect, color)
    rl.draw_rectangle_lines_ex(rect, 2.0, rl.BLACK)


def draw_centered_circle(center: rl.Vector2, radius: float, color: rl.Color) -> None:
    rl.draw_circle_v(center, radius, color)
    rl.draw_circle_lines(int(center.x), int(center.y), radius, rl.BLACK)


def get_root(holder: "ObjectHolder") -> "ObjectHolder":
    current = holder
    while current.parent is not None:
        current = current.parent
    return current


def get_leaf(holder: "ObjectHolder") -> "ObjectHolder":
    current = holder
    while True:
        if isinstance(current, ObjectHolder) and current.held_object is not None and isinstance(current.held_object, ObjectHolder):
            current = current.held_object
        else:
            return current


def get_composite(holder: "ObjectHolder") -> "Composite | None":
    current = get_root(holder)
    while current is not None:
        if isinstance(current, Composite):
            return current
        elif isinstance(current, ObjectHolder):
            current = current.held_object
        else:
            return None
    return None


def draw_progress_bar(position: rl.Vector2, progress: float) -> None:
    progress_width = WORLD_SCALE * 1.25
    progress_height = 5.0
    progress_rect = rl.Rectangle(
        position.x - (progress_width / 2),
        position.y - progress_width - progress_height - 2.0,
        progress_width * progress,
        progress_height,
    )
    rl.draw_rectangle_rec(progress_rect, rl.GREEN)
    rl.draw_rectangle_lines_ex(progress_rect, 1.0, rl.BLACK)


class GameObject:
    def __init__(self, level: "Level"):
        self.level: "Level" = level

    def update(self, delta_time: float) -> None:
        del delta_time

    def draw(self) -> None:
        pass


class Holdable(GameObject):
    """
    Base class for holdable objects in the game world.
    Holdable objects can be picked up and held by players, and can be put down onto object holders.
    """
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.parent: ObjectHolder | None = None
    pass


class Composite(Holdable):
    pass


class ObjectHolder(GameObject):
    """
    Base class for objects that can hold other objects that the player can grab from when they press the grab key.
    For example, Tabletops, Stoves, Chopping Boards, Dispensers, and Sinks are all ObjectHolders.
    """
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.held_object: Holdable | None = None
        self.parent: ObjectHolder | None = None

    def update(self, delta_time: float) -> None:
        super().update(delta_time)
        if self.held_object is not None:
            self.held_object.body.position = b2Vec2(self.body.position.x, self.body.position.y)

    def put_down(self, obj: Holdable) -> bool:
        if self.held_object is None:
            self.held_object = obj
            obj.parent = self
            return True
        return False

    def pick_up(self, player: "Player") -> Holdable | None:
        if self.held_object is not None:
            obj = self.held_object
            self.held_object = None
            obj.parent = None
            return obj
        return None


class IngredientHolder(ObjectHolder):
    """
    Base class for objects that can hold ingredients.
    """
    pass


class Interactable(ObjectHolder):
    """
    Base class for interactable objects in the game world.
    When a player is within range of an interactable, they can interact with it by pressing the interact button.
    """
    def __init__(self, level):
        super().__init__(level)
        self.total_time = 2.0
        self.current_time = 0.0
        self.progress = 0.0
        self.do_interact = False

    def update(self, delta_time: float) -> None:
        super().update(delta_time)
        if self.do_interact:
            self.current_time += delta_time
            # TODO: We can make progress go over 1.0 if we want things to catch on fire.
            self.progress = min(self.current_time / self.total_time, 1.0)
        if self.progress >= 1.0:
            self.do_interact = False

    def interact(self) -> None:
        self.do_interact = not self.do_interact
        if self.progress >= 1.0:
            self.do_interact = False

    def put_down(self, obj: "Holdable") -> bool:
        result = super().put_down(obj)
        if result and isinstance(obj, Onion):
            self.progress = obj.progress
            self.current_time = self.progress * self.total_time
        return result


class Onion(Holdable):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.progress = 0.0
        self.body = level.world.CreateStaticBody()

    def update(self, delta_time: float) -> None:
        del delta_time

    def draw(self) -> None:
        color = rl.GRAY
        if self.progress == 1.0:
            color = rl.YELLOW

        draw_centered_circle(center=physics_to_screen(self.body.position), radius=WORLD_SCALE * 0.2, color=color)


class Soup(Composite):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()
        self.progress = 0.0
        self.ingredients = []

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_screen(self.body.position), radius=WORLD_SCALE * 0.2, color=rl.ORANGE)

    def add_ingredient(self, ingredient: Onion) -> bool:
        if len(self.ingredients) < 3:
            if isinstance(ingredient, Onion) and ingredient.progress == 1.0:
                self.ingredients.append(ingredient)
                self.level.remove_game_object(ingredient)
                return True
        return False


class Dispenser(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2, held_object: GameObject | None = None):
        super().__init__(level)
        self.level = level
        self.held_object = held_object
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * TABLETOP_SIZE.x,
            color=rl.DARKGREEN,
        )

    def pick_up(self, player: "Player") -> Onion | None:
        # Dispensers can never run out. Construct a new held_object and return it.
        object = None
        if isinstance(self.held_object, Onion):
            object = Onion(self.level)
        # We need to add the object to the level's game_objects list so that it gets updated and drawn properly.
        if object is not None:
            self.level.game_objects.append(object)
        return object

    def put_down(self, object: GameObject) -> bool:
        # You can't put things into the dispenser.
        return False


class Tabletop(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2, size: b2Vec2 = TABLETOP_SIZE):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(size.x / 2, size.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(size.x, size.y)
        self.color = rl.LIGHTGRAY

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )


class Stove(Interactable):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)
        self.color = rl.RED

    def update(self, delta_time):
        self.do_interact = False
        if isinstance(self.held_object, Pot)and isinstance(self.held_object.held_object, Soup):
                # Make progress if there is a soup.
                self.do_interact = True
                self.held_object.held_object.progress = self.progress
        super().update(delta_time)

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )
        if self.held_object is not None and isinstance(self.held_object, Pot) and isinstance(self.held_object.held_object, Soup):
            draw_progress_bar(physics_to_screen(self.body.position), self.progress)

    def interact(self) -> None:
        # Stove is always on.
        pass

    def pick_up(self, player: "Player") -> Holdable | None:
        result = super().pick_up(player)
        if result:
            self.progress = 0.0
            self.current_time = 0.0
            self.do_interact = False
        return result

    def put_down(self, obj: Holdable) -> bool:
        result = super().put_down(obj)
        if result and isinstance(obj, Pot) and isinstance(obj.held_object, Soup):
            # If we put down a pot with soup on the stove, take its progress.
            self.progress = obj.held_object.progress
            self.current_time = self.progress * self.total_time
        return result


class ChoppingBoard(Interactable):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)
        self.color = rl.BROWN

    def update(self, delta_time):
        super().update(delta_time)
        if isinstance(self.held_object, Onion):
            self.held_object.progress = self.progress

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )
        if self.held_object is not None and isinstance(self.held_object, Onion):
            draw_progress_bar(physics_to_screen(self.body.position), self.progress)

    def interact(self) -> None:
        if isinstance(self.held_object, Onion):
            super().interact()


class Plate(Holdable, IngredientHolder):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_screen(self.body.position), radius=WORLD_SCALE * 0.25, color=rl.WHITE)

    def put_down(self, obj: GameObject) -> bool:
        # When putting down an IngredientHolder, take its contents instead.
        if self.held_object is None:
            if isinstance(obj, IngredientHolder) and obj.held_object is not None and isinstance(obj.held_object, Soup):
                self.held_object = obj.held_object
                self.held_object.parent = self
                obj.held_object = None
        elif isinstance(obj, IngredientHolder) and obj.held_object is None:
            # When putting down an IngredientHolder that is empty, transfer our contents instead.
            obj.held_object = self.held_object
            obj.held_object.parent = obj
            self.held_object = None
            # Return False because we don't want the player to drop the IngredientHolder.
            return False
        # Always return False because we don't want the player to drop the IngredientHolder.
        return False


class Pot(Holdable, IngredientHolder):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def put_down(self, obj: GameObject) -> bool:
        # Pots can only hold cut onions.
        if self.held_object is None:
            if isinstance(obj, Onion) and obj.progress == 1.0:
                soup = Soup(self.level)
                soup.add_ingredient(obj)
                self.level.game_objects.append(soup)
                self.held_object = soup
                # Set the stove progress to 0.
                if isinstance(obj.parent, Stove):
                    obj.parent.progress = 0.0
                    obj.parent.current_time = 0.0
                return True
            elif isinstance(obj, IngredientHolder) and obj.held_object is not None and isinstance(obj.held_object, Soup):
                # When putting down an IngredientHolder, take its contents instead.
                self.held_object = obj.held_object
                self.held_object.parent = self
                obj.held_object = None
                # Return False because we don't want the player to drop the IngredientHolder.
                return False
        elif isinstance(self.held_object, Soup):
            if isinstance(obj, Onion) and obj.progress == 1.0:
                added = self.held_object.add_ingredient(obj)
                if added:
                    # Reduce stove progress by 0.33 for each added ingredient, but not below 0.
                    if isinstance(self.parent, Stove):
                        self.parent.progress = max(0.0, self.parent.progress - 0.33)
                        self.parent.current_time = self.parent.progress * self.parent.total_time
                return added
            elif isinstance(obj, IngredientHolder) and obj.held_object is None:
                # When putting down an IngredientHolder that is empty, transfer our contents instead.
                obj.held_object = self.held_object
                obj.held_object.parent = obj
                self.held_object = None
                # Return False because we don't want the player to drop the IngredientHolder.
                return False
        return False

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_screen(self.body.position), radius=WORLD_SCALE * 0.3, color=rl.DARKGRAY)


class FireExtinguisher(Holdable):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_screen(self.body.position), radius=WORLD_SCALE * 0.2, color=rl.MAROON)


class DeliveryStation(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y * 2)
        self.color = rl.GOLD

    def draw(self) -> None:
        draw_centered_rectangle(
            center=physics_to_screen(self.body.position),
            width=WORLD_SCALE * self.size.x,
            height=WORLD_SCALE * self.size.y,
            color=self.color,
        )

    def put_down(self, obj: Plate) -> bool:
        # Only accept plates, and only if they have a cooked soup on them.
        # DeliveryStation can accept an unlimited number of plates.
        if isinstance(obj, Plate) and isinstance(obj.held_object, Soup) and obj.held_object.progress == 1.0:
            if super().put_down(obj):
                self.level.game_objects.remove(obj.held_object)
            obj.held_object = None
            return True
        return False


class PlateReturnStation(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)
        self.color = rl.SKYBLUE

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )


class DryingRack(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)
        self.color = rl.DARKBLUE

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )


class Sink(Interactable):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)
        self.color = rl.BLUE

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )


class RubbishBin(ObjectHolder):
    def __init__(self, level: "Level", position: b2Vec2):
        super().__init__(level)
        self.body = level.world.CreateStaticBody(position=position)
        self.body.CreateFixture(
            shape=b2PolygonShape(box=(TABLETOP_SIZE.x / 2, TABLETOP_SIZE.y / 2)),
            density=1.0,
            friction=0.0,
        )
        self.size = b2Vec2(TABLETOP_SIZE.x, TABLETOP_SIZE.y)

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_screen(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=rl.GREEN,
        )

    def put_down(self, obj: GameObject) -> bool:
        # Rubbish bins can accept ingredients.
        if isinstance(obj, Onion):
            # Remove the object from the level's game_objects list so that it no longer gets updated or drawn.
            if obj in self.level.game_objects:
                self.level.game_objects.remove(obj)
            return True
        return False


class Player:
    def __init__(
        self,
        player_num: int,
        level: "Level",
        position: b2Vec2,
        agent: Agent,
    ):
        self.player_num = player_num
        self.radius = 0.4
        self.move_speed = 7.2 * 60
        self.agent = agent
        self.level = level
        self.held_object: GameObject | None = None
        self.body = level.world.CreateDynamicBody(
            position=position,
            fixedRotation=True,
            linearDamping=12.0,
            allowSleep=False,
        )
        self.body.CreateFixture(
            shape=b2CircleShape(radius=self.radius),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

    def get_move_direction(self, input_state: InputState) -> tuple[float, float]:
        if self.agent is None:
            return 0.0, 0.0

        horizontal = input_state.move_x
        vertical = input_state.move_y

        magnitude = math.hypot(horizontal, vertical)
        if magnitude == 0.0:
            return 0.0, 0.0

        return horizontal / magnitude, vertical / magnitude

    def pick_up_or_put_down(self, input_state: InputState) -> None:
        # Check for nearby object_holders to pick up from or put down onto.
        # Priority is given to putting down over picking up, and to the first object_holders found in the list.
        if input_state.carry:
            for object_holder in self.level.object_holders:
                distance = (object_holder.body.position - self.body.position).length
                if distance <= self.radius * 3:
                    if self.held_object is None:
                        # When picking up we find the root.
                        root = get_root(object_holder)
                        if root.held_object is not None:
                            self.held_object = root.pick_up(self)
                            return
                    else:
                        # When putting down we find the leaf.
                        leaf = get_leaf(object_holder)
                        if leaf.put_down(self.held_object):
                            self.held_object = None
                            return

    def do_interact(self, input_state: InputState) -> None:
        if input_state.interact:
            for interactable in self.level.interactables:
                distance = (interactable.body.position - self.body.position).length
                if distance <= self.radius * 3:
                    interactable.interact()

    def update(self, delta_time: float) -> None:
        input_state = self.agent.update(delta_time, create_game_state(self.level))
        move_x, move_y = self.get_move_direction(input_state)
        self.body.linearVelocity = (move_x * self.move_speed * delta_time, move_y * self.move_speed * delta_time)
        self.pick_up_or_put_down(input_state)
        self.do_interact(input_state)
        if self.held_object is not None:
            # If the player is holding an object, update its position to match the player's position.
            self.held_object.body.position = b2Vec2(self.body.position.x, self.body.position.y)

    def draw(self) -> None:
        player_draw_radius = WORLD_SCALE * self.radius
        world_xy = physics_to_screen(self.body.position)
        player_colors = (rl.PURPLE, rl.DARKPURPLE)
        player_color = player_colors[self.player_num % len(player_colors)]
        rl.draw_circle_v(world_xy, player_draw_radius, player_color)
        rl.draw_circle_lines(int(world_xy.x), int(world_xy.y), player_draw_radius, rl.BLACK)


class Level:
    def __init__(self, level_data: dict):
        self.level_data = level_data
        self.layout_objects: list[dict] = level_data.get("layout", [])
        self.player_starts: list[list] = level_data.get("player_starts", [])
        self.velocity_iterations = 8
        self.position_iterations = 3
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.players: list[Player] = []
        self.game_objects: list[GameObject] = []
        self.interactables: list[Interactable] = []
        self.object_holders: list[ObjectHolder] = []

        self.camera_padding = 100
        self.camera_min_zoom = 0.05
        self.camera_max_zoom = 10.0
        self.camera_zoom_step = 0.1

        self.build_level()
        self.camera = self.build_camera()

    def get_screen_size(self) -> tuple[int, int]:
        width = rl.get_screen_width()
        height = rl.get_screen_height()
        return width, height

    def collect_world_points(self) -> list[rl.Vector2]:
        to_world_xy = lambda position: rl.Vector2(position[0] * WORLD_SCALE, -position[1] * WORLD_SCALE)

        points = [to_world_xy(layout_object.get("position", [0.0, 0.0])) for layout_object in self.layout_objects]
        points.extend(to_world_xy(player_start) for player_start in self.player_starts)
        return points

    def build_camera(self) -> rl.Camera2D:
        screen_width, screen_height = self.get_screen_size()
        points = self.collect_world_points()
        if not points:
            return rl.Camera2D(
                rl.Vector2(screen_width / 2, screen_height / 2),
                rl.Vector2(0.0, 0.0),
                0.0,
                1.0,
            )

        min_x = min(point.x for point in points)
        max_x = max(point.x for point in points)
        min_y = min(point.y for point in points)
        max_y = max(point.y for point in points)

        world_width = max(max_x - min_x, WORLD_SCALE * 2.5)
        world_height = max(max_y - min_y, WORLD_SCALE * 2.5)
        usable_width = max(screen_width - (2 * self.camera_padding), 1)
        usable_height = max(screen_height - (2 * self.camera_padding), 1)
        zoom_x = usable_width / world_width
        zoom_y = usable_height / world_height
        zoom = max(self.camera_min_zoom, min(min(zoom_x, zoom_y), self.camera_max_zoom))

        return rl.Camera2D(
            rl.Vector2(screen_width / 2, screen_height / 2),
            rl.Vector2((min_x + max_x) / 2, (min_y + max_y) / 2),
            0.0,
            zoom,
        )

    def update_camera(self) -> None:
        screen_width, screen_height = self.get_screen_size()
        self.camera.offset = rl.Vector2(screen_width / 2, screen_height / 2)

        if rl.is_key_pressed(rl.KEY_R):
            self.camera = self.build_camera()
            return

        wheel_move = rl.get_mouse_wheel_move()
        if wheel_move != 0:
            zoom_multiplier = 1.0 + (wheel_move * self.camera_zoom_step)
            self.camera.zoom = max(self.camera_min_zoom, min(self.camera.zoom * zoom_multiplier, self.camera_max_zoom))

        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_MIDDLE):
            delta = rl.get_mouse_delta()
            self.camera.target.x -= delta.x / self.camera.zoom
            self.camera.target.y -= delta.y / self.camera.zoom

    def build_level(self) -> None:
        # Create players first so they are first in the draw order.
        for player_num, player_start in enumerate(self.player_starts):

            position = world_to_physics(player_start)
            agent = HumanAgent(player_num) if player_num == 0 else RandomAgent(player_num)
            player = Player(
                player_num=player_num,
                level=self,
                position=position,
                agent=agent,
            )
            self.players.append(player)
            self.game_objects.append(player)

        for layout_object in self.layout_objects:
            object_type = layout_object.get("type")
            position = world_to_physics(layout_object.get("position", [0.0, 0.0]))
            if object_type == "tabletop":
                self.game_objects.append(Tabletop(level=self, position=position))
                continue

            if object_type == "dispenser":
                held_object = None
                if layout_object.get("ingredient") == "onion":
                    held_object = Onion(self)
                self.game_objects.append(Dispenser(level=self, position=position, held_object=held_object))
                continue

            if object_type == "chopping_board":
                self.game_objects.append(ChoppingBoard(level=self, position=position))
                continue

            if object_type == "stove":
                self.game_objects.append(Stove(level=self, position=position))
                continue

            if object_type == "delivery_station":
                self.game_objects.append(DeliveryStation(level=self, position=position))
                continue

            if object_type == "plate_return_station":
                self.game_objects.append(PlateReturnStation(level=self, position=position))
                continue

            if object_type == "drying_rack":
                self.game_objects.append(DryingRack(level=self, position=position))
                continue

            if object_type == "sink":
                self.game_objects.append(Sink(level=self, position=position))
                continue

            if object_type == "rubbish_bin":
                self.game_objects.append(RubbishBin(level=self, position=position))
                continue


        # If a PlateReturn and DryingRack exist at the same position, destroy the PlateReturn.
        for i in range(len(self.game_objects)):
            for j in range(i + 1, len(self.game_objects)):
                obj1 = self.game_objects[i]
                obj2 = self.game_objects[j]
                if isinstance(obj1, PlateReturnStation) and isinstance(obj2, DryingRack) and obj1.body.position == obj2.body.position:
                    self.game_objects.remove(obj1)
                    break
                elif isinstance(obj2, PlateReturnStation) and isinstance(obj1, DryingRack) and obj2.body.position == obj1.body.position:
                    self.game_objects.remove(obj2)
                    break

        # Populate interactables and object holders lists for easy access later.
        for game_object in self.game_objects:
            if isinstance(game_object, Interactable):
                self.interactables.append(game_object)
            if isinstance(game_object, ObjectHolder):
                self.object_holders.append(game_object)

        # Holdable items like plates, pots, and fire extinguishers always start on top of an object holder. Find the one closest to the layout_object's position and put the holdable item on it.
        for layout_object in self.layout_objects:
            object_type = layout_object.get("type")
            object = None
            if object_type == "plate":
                object = Plate(self)
            elif object_type == "pot":
                object = Pot(self)
            elif object_type == "fire_extinguisher":
                object = FireExtinguisher(self)

            if object is not None:
                # Find closest object holder to the layout_object's position.
                position = world_to_physics(layout_object.get("position", [0.0, 0.0]))
                closest_holder = None
                closest_distance = float("inf")
                for object_holder in self.object_holders:
                    distance = (object_holder.body.position - position).length
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_holder = object_holder

                if closest_holder is not None:
                    closest_holder.put_down(object)
                    self.game_objects.append(object)


    def update(self, delta_time: float) -> None:
        self.update_camera()

        if delta_time <= 0.0:
            return

        for game_object in self.game_objects:
            game_object.update(delta_time)

        self.world.Step(delta_time, self.velocity_iterations, self.position_iterations)

    def draw(self) -> None:
        rl.begin_mode_2d(self.camera)
        for game_object in self.game_objects:
            game_object.draw()
        rl.end_mode_2d()

    def remove_game_object(self, obj: GameObject) -> None:
        if obj in self.game_objects:
            self.game_objects.remove(obj)


OBJECT_TO_NAME = {
    Tabletop: "tabletop",
    Dispenser: "dispenser",
    ChoppingBoard: "chopping_board",
    Stove: "stove",
    DeliveryStation: "delivery_station",
    PlateReturnStation: "plate_return_station",
    DryingRack: "drying_rack",
    Sink: "sink",
    RubbishBin: "rubbish_bin",
    FireExtinguisher: "fire_extinguisher",
    Plate: "plate",
    Pot: "pot",
    Soup: "soup",
    Onion: "onion",
}


NAME_TO_OBJECT = {v: k for k, v in OBJECT_TO_NAME.items()}


def create_game_state(level: Level) -> dict:
        state: dict[str, dict[str, float]] = {}
        for player in level.players:
            name = f"player_{player.player_num}"
            position = physics_to_world(player.body.position)
            state[name] = {"position": position}
        for game_object in level.game_objects:
            if not isinstance(game_object, Player):
                name = OBJECT_TO_NAME[type(game_object)]
                position = physics_to_world(game_object.body.position)
                state[name] = {"position": position}
                if isinstance(game_object, Onion) or isinstance(game_object, Soup):
                    state[name]["progress"] = game_object.progress
        return state
