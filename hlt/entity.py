import abc
from typing import Iterable

from . import commands, constants
from .common import read_input
from .positionals import Direction, Position


class Entity(abc.ABC):
    """
    Base Entity Class from whence Ships, Dropoffs and Shipyards inherit
    """

    def __init__(self, owner, id, position):
        self.owner: int = owner
        self.id = id
        self.position: 'Position' = position

    @staticmethod
    def _generate(player_id):
        """
        Method which creates an entity for a specific player given input from the engine.
        :param player_id: The player id for the player who owns this entity
        :return: An instance of Entity along with its id
        """
        ship_id, x_position, y_position = map(int, read_input().split())
        return ship_id, Entity(player_id, ship_id, Position(x_position, y_position))

    def __repr__(self):
        return "{}(id={}, {})".format(self.__class__.__name__,
                                      self.id,
                                      self.position)


class Dropoff(Entity):
    """
    Dropoff class for housing dropoffs
    """
    pass


class Shipyard(Entity):
    """
    Shipyard class to house shipyards
    """

    def spawn(self):
        """Return a move to spawn a new ship."""
        return commands.GENERATE


class Ship(Entity):
    """
    Ship class to house ship entities
    """
    __ships = {}

    def __init__(self, owner, id, position, halite_amount):
        super().__init__(owner, id, position)
        self.halite_amount = halite_amount
        self.exists = True

    @property
    def is_full(self):
        """Is this ship at max halite capacity?"""
        return self.halite_amount >= constants.MAX_HALITE

    def make_dropoff(self):
        """Return a move to transform this ship into a dropoff."""
        return "{} {}".format(commands.CONSTRUCT, self.id)

    def move(self, direction):
        """
        Return a move to move this ship in a direction without
        checking for collisions.
        """
        raw_direction = Direction.convert(direction) if isinstance(direction, tuple) else direction
        return "{} {} {}".format(commands.MOVE, self.id, raw_direction)

    def stay_still(self):
        """
        Don't move this ship.
        """
        return "{} {} {}".format(commands.MOVE, self.id, commands.STAY_STILL)

    @classmethod
    def _generate(cls, player_id):
        """
        Creates an instance of a ship for a given player given the engine's input.
        If an instance with the same ship.id has previously been generated, that instance will be returned.
        :param player_id: The id of the player who owns this ship
        :return: The ship id and ship object
        """
        # Read game engine input
        ship_id, x_position, y_position, halite = map(int, read_input().split())

        # Check storage to see if ship already exists
        # If the ship exists, update its position and halite
        if ship_id in cls.__ships.keys():
            old_ship = cls.__ships[ship_id]
            old_ship.position.x, old_ship.position.y = x_position, y_position
            old_ship.halite_amount = halite
            return ship_id, old_ship
        else:
            # Otherwise, create and return a new instance
            new_ship = cls(player_id, ship_id, Position(x_position, y_position), halite)
            cls.__ships[ship_id] = new_ship
            return ship_id, new_ship

    @classmethod
    def _mark_destroyed(cls, exists_ids: Iterable[int]):
        exists_ids = set(exists_ids)
        for shid, ship in cls.__ships.items():
            if shid not in exists_ids:
                ship.exists = False

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, {self.position}, cargo={self.halite_amount} halite)"

    def __hash__(self):
        return hash(self.id) ^ id(type(self))

    def __eq__(self, other):
        return other and self.id == other.id
