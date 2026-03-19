import math
import pyray as rl

from Box2D import b2CircleShape, b2PolygonShape, b2Vec2, b2World


WORLD_SCALE = 50.0
OBJECT_DRAW_SIZE = WORLD_SCALE * 1.25
TABLETOP_SIZE = b2Vec2(1.25, 1.25)
PLAYER_RADIUS = 0.4
PLAYER_MOVE_SPEED = 7.2
PLAYER_CONTROL_SCHEMES = (
    {"up": rl.KEY_W, "down": rl.KEY_S, "left": rl.KEY_A, "right": rl.KEY_D},
    {"up": rl.KEY_UP, "down": rl.KEY_DOWN, "left": rl.KEY_LEFT, "right": rl.KEY_RIGHT},
)
TABLETOP_COLOR = rl.LIGHTGRAY
PLAYER_COLORS = (rl.MAGENTA, rl.ORANGE)

def world_to_physics(position: list[float]) -> b2Vec2:
    return b2Vec2(position[0], -position[1])


def physics_to_world(position: b2Vec2) -> rl.Vector2:
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
    pass


class ObjectHolder(GameObject):
    """
    Base class for objects that can hold other objects that the player can grab from when they press the grab key.
    For example, Tabletops, Stoves, Chopping Boards, Dispensers, and Sinks are all ObjectHolders.
    """
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.held_object: Holdable | None = None

    def update(self, delta_time: float) -> None:
        super().update(delta_time)
        if self.held_object is not None:
            self.held_object.body.position = b2Vec2(self.body.position.x, self.body.position.y)

    def pick_up(self) -> Holdable | None:
        if self.held_object is not None:
            obj = self.held_object
            self.held_object = None
            return obj
        return None

    def put_down(self, obj: Holdable) -> bool:
        if self.held_object is None:
            self.held_object = obj
            return True
        elif isinstance(self.held_object, ObjectHolder):
            # If the object holder is already holding something, try to put the object down onto that object.
            return self.held_object.put_down(obj)
        return False


class Interactable(ObjectHolder):
    """
    Base class for interactable objects in the game world.
    When a player is within range of an interactable, they can interact with it by pressing the interact button.
    """
    def interact(self) -> None:
        pass


class Onion(Holdable):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.state = "whole"
        self.body = level.world.CreateStaticBody()

    def update(self, delta_time: float) -> None:
        del delta_time

    def draw(self) -> None:
        color = rl.GRAY
        if self.state == "whole":
            color = rl.GRAY
        elif self.state == "chopped":
            color = rl.YELLOW
        elif self.state == "cooked":
            color = rl.BROWN

        draw_centered_circle(center=physics_to_world(self.body.position), radius=OBJECT_DRAW_SIZE * 0.25, color=color)


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
            center=physics_to_world(self.body.position),
            size=WORLD_SCALE * TABLETOP_SIZE.x,
            color=rl.DARKGREEN,
        )

    def pick_up(self) -> Onion | None:
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
        self.color = TABLETOP_COLOR

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_world(self.body.position),
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

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_world(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )

    def interact(self) -> None:
        if isinstance(self.held_object, Pot):
            if isinstance(self.held_object.held_object, Onion) and self.held_object.held_object.state == "chopped":
                self.held_object.held_object.state = "cooked"


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

    def draw(self) -> None:
        draw_centered_square(
            center=physics_to_world(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )

    def interact(self) -> None:
        if isinstance(self.held_object, Onion) and self.held_object.state == "whole":
            self.held_object.state = "chopped"


class Plate(Holdable):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_world(self.body.position), radius=OBJECT_DRAW_SIZE * 0.22, color=rl.WHITE)


class Pot(Holdable, ObjectHolder):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def put_down(self, obj: GameObject) -> bool:
        # Pots can only hold cut onions.
        if isinstance(obj, Onion) and obj.state == "chopped":
            return super().put_down(obj)
        return False

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_world(self.body.position), radius=OBJECT_DRAW_SIZE * 0.28, color=rl.DARKGRAY)


class FireExtinguisher(Holdable):
    def __init__(self, level: "Level"):
        super().__init__(level)
        self.body = level.world.CreateStaticBody()

    def draw(self) -> None:
        draw_centered_circle(center=physics_to_world(self.body.position), radius=OBJECT_DRAW_SIZE * 0.18, color=rl.MAROON)


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
            center=physics_to_world(self.body.position),
            width=WORLD_SCALE * self.size.x,
            height=WORLD_SCALE * self.size.y,
            color=self.color,
        )

    def put_down(self, obj: Plate) -> bool:
        # Only accept plates, and only if they have a cooked onion on them.
        # DeliveryStation can accept an unlimited number of plates.
        if isinstance(obj, Plate) and isinstance(obj.held_object, Onion) and obj.held_object.state == "cooked":
            super().put_down(obj)
            del obj.held_object.held_object
            del obj.held_object
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
            center=physics_to_world(self.body.position),
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
            center=physics_to_world(self.body.position),
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
            center=physics_to_world(self.body.position),
            size=WORLD_SCALE * self.size.x,
            color=self.color,
        )


class Player:
    def __init__(
        self,
        player_num: int,
        level: "Level",
        position: b2Vec2,
        radius: float = PLAYER_RADIUS,
        move_speed: float = PLAYER_MOVE_SPEED,
        controls: dict | None = None,
    ):
        self.player_num = player_num
        self.radius = radius
        self.move_speed = move_speed
        self.controls = controls
        self.level = level
        self.held_object: GameObject | None = None
        self.body = level.world.CreateDynamicBody(
            position=position,
            fixedRotation=True,
            linearDamping=12.0,
            allowSleep=False,
        )
        self.body.CreateFixture(
            shape=b2CircleShape(radius=radius),
            density=1.0,
            friction=0.0,
            restitution=0.0,
        )

    def get_move_direction(self) -> tuple[float, float]:
        if self.controls is None:
            return 0.0, 0.0

        horizontal = float(rl.is_key_down(self.controls["right"])) - float(
            rl.is_key_down(self.controls["left"])
        )
        vertical = float(rl.is_key_down(self.controls["down"])) - float(
            rl.is_key_down(self.controls["up"])
        )

        magnitude = math.hypot(horizontal, vertical)
        if magnitude == 0.0:
            return 0.0, 0.0

        return horizontal / magnitude, vertical / magnitude

    def pick_up_or_put_down(self) -> GameObject | None:
        # Check for nearby object_holders to pick up from or put down onto.
        # Priority is given to putting down over picking up, and to the first object_holders found in the list.
        if rl.is_key_pressed(rl.KEY_SPACE):
            for object_holder in self.level.object_holders:
                distance = (object_holder.body.position - self.body.position).length
                if distance <= self.radius * 3:
                    if self.held_object is not None:
                        if object_holder.put_down(self.held_object):
                            self.held_object = None
                            return None
                    else:
                        obj = object_holder.pick_up()
                        if obj is not None:
                            self.held_object = obj
                            return obj
        return self.held_object

    def do_interact(self) -> None:
        if rl.is_key_pressed(rl.KEY_LEFT_CONTROL):
            for interactable in self.level.interactables:
                distance = (interactable.body.position - self.body.position).length
                if distance <= self.radius * 3:
                    interactable.interact()

    def update(self, delta_time: float) -> None:
        del delta_time
        move_x, move_y = self.get_move_direction()
        self.body.linearVelocity = (move_x * self.move_speed, move_y * self.move_speed)
        self.pick_up_or_put_down()
        self.do_interact()
        if self.held_object is not None:
            # If the player is holding an object, update its position to match the player's position.
            self.held_object.body.position = b2Vec2(self.body.position.x, self.body.position.y)

    def draw(self) -> None:
        player_draw_radius = WORLD_SCALE * self.radius
        world_xy = physics_to_world(self.body.position)
        player_color = PLAYER_COLORS[self.player_num % len(PLAYER_COLORS)]
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
        self.game_objects: list[object] = []
        self.interactables: list[Interactable] = []
        self.object_holders: list[ObjectHolder] = []

        self.build_level()

    def build_level(self) -> None:
        # Create players first so they are first in the draw order.
        for player_num, player_start in enumerate(self.player_starts):
            controls = None
            if player_num < len(PLAYER_CONTROL_SCHEMES):
                controls = PLAYER_CONTROL_SCHEMES[player_num]

            position = world_to_physics(player_start)
            player = Player(
                player_num=player_num,
                level=self,
                position=position,
                controls=controls,
            )
            self.players.append(player)
            self.game_objects.append(player)

        for layout_object in self.layout_objects:
            object_type = layout_object.get("type", layout_object.get("category"))
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
        if delta_time <= 0.0:
            return

        for game_object in self.game_objects:
            game_object.update(delta_time)

        self.world.Step(delta_time, self.velocity_iterations, self.position_iterations)

    def draw(self) -> None:
        for game_object in self.game_objects:
            game_object.draw()
