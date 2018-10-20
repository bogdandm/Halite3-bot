from typing import Tuple, Union

from . import commands


class Direction:
    """
    Holds positional tuples in relation to cardinal directions
    """
    North = (0, -1)
    South = (0, 1)
    East = (1, 0)
    West = (-1, 0)

    All = (North, South, East, West)

    Still = (0, 0)

    @staticmethod
    def convert(direction):
        """
        Converts from this direction tuple notation to the engine's string notation
        :param direction: the direction in this notation
        :return: The character equivalent for the game engine
        """
        if direction == Direction.North:
            return commands.NORTH
        if direction == Direction.South:
            return commands.SOUTH
        if direction == Direction.East:
            return commands.EAST
        if direction == Direction.West:
            return commands.WEST
        if direction == Direction.Still:
            return commands.STAY_STILL
        else:
            raise IndexError

    @staticmethod
    def invert(direction):
        """
        Returns the opposite cardinal direction given a direction
        :param direction: The input direction
        :return: The opposite direction
        """
        if direction == Direction.North:
            return Direction.South
        if direction == Direction.South:
            return Direction.North
        if direction == Direction.East:
            return Direction.West
        if direction == Direction.West:
            return Direction.East
        if direction == Direction.Still:
            return Direction.Still
        else:
            raise IndexError

    @staticmethod
    def nearby(direction: Tuple[int, int]):
        d = (0, 1) if direction[0] else (1, 0)
        return [d, (d[0] * -1, d[1] * -1)]


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def directional_offset(self, direction):
        """
        Returns the position considering a Direction cardinal tuple
        :param direction: the direction cardinal tuple
        :return: a new position moved in that direction
        """
        return self + Position(*direction)

    def get_surrounding_cardinals(self):
        """
        :return: Returns a list of all positions around this specific position in each cardinal direction
        """
        return [self.directional_offset(current_direction) for current_direction in Direction.All]

    def __add__(self, other: Union['Position', Tuple[int, int]]):
        if isinstance(other, Position):
            return Position(self.x + other.x, self.y + other.y)
        else:
            return Position(self.x + other[0], self.y + other[1])

    def __sub__(self, other: Union['Position', Tuple[int, int]]):
        if isinstance(other, Position):
            return Position(self.x - other.x, self.y - other.y)
        else:
            return Position(self.x - other[0], self.y - other[1])

    def __iadd__(self, other: Union['Position', Tuple[int, int]]):
        is_pos = isinstance(other, Position)
        self.x += other.x if is_pos else other[0]
        self.y += other.y if is_pos else other[1]
        return self

    def __isub__(self, other: Union['Position', Tuple[int, int]]):
        is_pos = isinstance(other, Position)
        self.x -= other.x if is_pos else other[0]
        self.y -= other.y if is_pos else other[1]
        return self

    def __abs__(self):
        return Position(abs(self.x), abs(self.y))

    def __eq__(self, other: 'Position'):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__,
                                   self.x,
                                   self.y)

    def __mul__(self, other) -> 'Position':
        return Position(self.x * other, self.y * other)

    def __hash__(self):
        return hash((self.x, self.y)) ^ id(type(self))

    def __iter__(self):
        yield self.x
        yield self.y
