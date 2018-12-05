from itertools import chain
from queue import PriorityQueue
from typing import Iterable, List, Optional, Union

import numpy as np

from bot.utils import extend_grid, memoized_method
from hlt import constants
from .common import read_input
from .entity import Dropoff, Entity, Ship, Shipyard
from .positionals import Direction, Position


class Player:
    """
    Player object containing all items/metadata pertinent to the player.
    """

    def __init__(self, player_id, shipyard, halite=0):
        self.id = player_id
        self.shipyard: 'Shipyard' = shipyard
        self.halite_amount = halite
        self._ships = {}
        self._dropoffs = {}

    def get_ship(self, ship_id):
        """
        Returns a singular ship mapped by the ship id
        :param ship_id: The ship id of the ship you wish to return
        :return: the ship object.
        """
        return self._ships[ship_id]

    def get_ships(self) -> List['Ship']:
        """
        :return: Returns all ship objects in a list
        """
        return list(self._ships.values())

    def get_dropoff(self, dropoff_id):
        """
        Returns a singular dropoff mapped by its id
        :param dropoff_id: The dropoff id to return
        :return: The dropoff object
        """
        return self._dropoffs[dropoff_id]

    def get_dropoffs(self):
        """
        :return: Returns all dropoff objects in a list
        """
        return list(self._dropoffs.values())

    def has_ship(self, ship_id):
        """
        Check whether the player has a ship with a given ID.

        Useful if you track ships via IDs elsewhere and want to make
        sure the ship still exists.

        :param ship_id: The ID to check.
        :return: True if and only if the ship exists.
        """
        return ship_id in self._ships

    @staticmethod
    def _generate():
        """
        Creates a player object from the input given by the game engine
        :return: The player object
        """
        player, shipyard_x, shipyard_y = map(int, read_input().split())
        return Player(player, Shipyard(player, -1, Position(shipyard_x, shipyard_y)))

    def _update(self, num_ships, num_dropoffs, halite):
        """
        Updates this player object considering the input from the game engine for the current specific turn.
        :param num_ships: The number of ships this player has this turn
        :param num_dropoffs: The number of dropoffs this player has this turn
        :param halite: How much halite the player has in total
        :return: nothing.
        """
        self.halite_amount = halite
        self._ships = {id: ship for (id, ship) in [Ship._generate(self.id) for _ in range(num_ships)]}
        Ship._mark_destroyed(self._ships.keys())
        self._dropoffs = {id: dropoff for (id, dropoff) in [Dropoff._generate(self.id) for _ in range(num_dropoffs)]}


class MapCell:
    """A cell on the game map."""

    def __init__(self, position, halite_amount):
        self.position = position
        self.halite_amount = halite_amount
        self.ship: Ship = None
        self.structure = None

    @property
    def is_empty(self):
        """
        :return: Whether this cell has no ships or structures
        """
        return self.ship is None and self.structure is None

    @property
    def is_occupied(self):
        """
        :return: Whether this cell has any ships
        """
        return self.ship is not None

    def is_occupied_base(self, me: Player):
        return self.structure and self.ship and self.structure.owner == me.id and self.ship.owner != me.id

    @property
    def has_structure(self):
        """
        :return: Whether this cell has any structures
        """
        return self.structure is not None

    @property
    def structure_type(self):
        """
        :return: What is the structure type in this cell
        """
        return None if not self.structure else type(self.structure)

    def mark_unsafe(self, ship):
        """
        Mark this cell as unsafe (occupied) for navigation.

        Use in conjunction with GameMap.naive_navigate.
        """
        self.ship = ship

    def mark_safe(self):
        """
        Mark this cell as safe for navigation.

        Use in conjunction with GameMap.naive_navigate.
        """
        self.ship = None

    def __eq__(self, other):
        return self.position == other.position

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f'MapCell({self.position}, halite={self.halite_amount})'


class PrioritizedItem:
    def __init__(self, position, priority):
        self.position = position
        self.priority = priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __gte__(self, other):
        return self.priority >= other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __lte__(self, other):
        return self.priority <= other.priority


class GameMap:
    """
    The game map.

    Can be indexed by a position, or by a contained entity.
    Coordinates start at 0. Coordinates are normalized for you
    """

    def __init__(self, cells: List[List[MapCell]], width, height):
        self.width = width
        self.height = height
        self._cells = cells
        self.halite = np.empty((height, width), dtype=float)
        for y, row in enumerate(cells):
            for x, cell in enumerate(row):
                self.halite[y, x] = cell.halite_amount
        self.initial_halite = self.total_halite
        self.halite_extended = extend_grid(self.halite)

    def __getitem__(self, location) -> Optional[Union['MapCell', Iterable['MapCell']]]:
        """
        Getter for position object or entity objects within the game map
        :param location: the position or entity to access in this map
        :return: the contents housing that cell or entity
        """
        if isinstance(location, Position):
            location = self.normalize(location)
            return self._cells[location.y][location.x]
        elif isinstance(location, tuple):
            location = self.normalize(Position(*location))
            return self._cells[location.y][location.x]
        elif isinstance(location, Entity):
            return self._cells[location.position.y][location.position.x]
        elif hasattr(location, '__iter__'):
            return (self._cells[position.y][position.x] for position in location)
        return None

    def __iter__(self):
        for row in self._cells:
            yield from row

    @property
    def cells(self) -> List[List[MapCell]]:
        return self._cells

    @property
    def total_halite(self):
        return self.halite.sum()

    @memoized_method
    def distance(self, source: Position, target: Position):
        """
        Compute the Manhattan distance between two locations.
        Accounts for wrap-around.
        :param source: The source from where to calculate
        :param target: The target to where calculate
        :return: The distance between these items
        """
        source = self.normalize(source)
        target = self.normalize(target)
        resulting_position = abs(source - target)
        return min(resulting_position.x, self.width - resulting_position.x) + \
               min(resulting_position.y, self.height - resulting_position.y)

    # @memoized_method
    def normalize(self, position):
        """
        Normalized the position within the bounds of the toroidal map.
        i.e.: Takes a point which may or may not be within width and
        height bounds, and places it within those bounds considering
        wraparound.
        :param position: A position object.
        :return: A normalized position object fitting within the bounds of the map
        """
        if not hasattr(position, 'x'):
            return Position(position[0] % self.width, position[1] % self.height)
        else:
            return Position(position.x % self.width, position.y % self.height)

    def normalize_direction(self, direction):
        return tuple(map(lambda x: x if abs(x) == 1 or x == 0 else -abs(x) // x, direction))

    @staticmethod
    def _get_target_direction(source, target):
        """
        Returns where in the cardinality spectrum the target is from source. e.g.: North, East; South, West; etc.
        NOTE: Ignores toroid
        :param source: The source position
        :param target: The target position
        :return: A tuple containing the target Direction. A tuple item (or both) could be None if within same coords
        """
        return (Direction.South if target.y > source.y else Direction.North if target.y < source.y else None,
                Direction.East if target.x > source.x else Direction.West if target.x < source.x else None)

    def get_unsafe_moves(self, source, destination):
        """
        Return the Direction(s) to move closer to the target point, or empty if the points are the same.
        This move mechanic does not account for collisions. The multiple directions are if both directional movements
        are viable.
        :param source: The starting position
        :param destination: The destination towards which you wish to move your object.
        :return: A list of valid (closest) Directions towards your target.
        """
        source = self.normalize(source)
        destination = self.normalize(destination)
        distance = abs(destination - source)
        y_cardinality, x_cardinality = self._get_target_direction(source, destination)

        if distance.x != 0:
            yield x_cardinality if distance.x < (self.width / 2) else Direction.invert(x_cardinality)
        if distance.y != 0:
            yield y_cardinality if distance.y < (self.height / 2) else Direction.invert(y_cardinality)

    def update_ship_position(self, ship: 'Ship', direction: tuple):
        self[ship.position + direction].mark_unsafe(ship)
        self[ship.position].mark_safe()
        return ship.move(direction)

    def swap_ships(self, ship1: 'Ship', ship2: 'Ship'):
        self[ship1.position].mark_unsafe(ship2)
        self[ship2.position].mark_unsafe(ship1)
        yield ship1.move(self.normalize_direction(ship2.position - ship1.position))
        yield ship2.move(self.normalize_direction(ship1.position - ship2.position))

    def naive_navigate(self, ship, destination):
        """
        Returns a singular safe move towards the destination.

        :param ship: The ship to move.
        :param destination: Ending position
        :return: A direction.
        """
        moves = list(self.get_unsafe_moves(ship.position, destination))
        for direction in chain(moves, *map(Direction.nearby, moves)):
            if not self[ship.position + direction].is_occupied:
                self.update_ship_position(ship, direction)
                return direction

        return Direction.Still

    def a_star_path_search(self, start: 'Position', target: 'Position', ignore_ships=True, move_penalty=10):
        total_halite = self.total_halite
        halite_estimated_per_cell = total_halite / self.width / self.height / constants.MOVE_COST_RATIO / 2

        queue = PriorityQueue()
        queue.put(PrioritizedItem(start, 0))
        closed = {start: 0}
        came_from = {start: None}

        while not queue.empty():
            current: Position = queue.get().position
            if current == target:
                break

            for direction in Direction.All:
                next_node = self.normalize(current + direction)
                new_cost = closed[current] + self[next_node].halite_amount / constants.MOVE_COST_RATIO + move_penalty
                if ignore_ships is False and self[next_node].is_occupied:
                    new_cost += constants.MAX_HALITE / constants.MOVE_COST_RATIO / 2
                if next_node not in closed or new_cost < closed[next_node]:
                    closed[next_node] = new_cost
                    came_from[next_node] = current
                    estimated_cost = new_cost + self.distance(next_node, target) * halite_estimated_per_cell
                    queue.put(PrioritizedItem(next_node, estimated_cost))

        result = [target]
        current = came_from[target]
        while current != start:
            result.append(current)
            current = came_from[current]
        result.reverse()
        return result

    @staticmethod
    def _generate():
        """
        Creates a map object from the input given by the game engine
        :return: The map object
        """
        map_width, map_height = map(int, read_input().split())
        game_map: List[List[MapCell]] = [[None for _ in range(map_width)] for _ in range(map_height)]
        for y_position in range(map_height):
            cells = read_input().split()
            for x_position in range(map_width):
                game_map[y_position][x_position] = MapCell(
                    Position(x_position, y_position),
                    int(cells[x_position])
                )
        return GameMap(game_map, map_width, map_height)

    def _update(self):
        """
        Updates this map object from the input given by the game engine
        :return: nothing
        """
        # Mark cells as safe for navigation (will re-mark unsafe cells
        # later)
        for row in self.cells:
            for cell in row:
                cell.ship = None

        for _ in range(int(read_input())):
            cell_x, cell_y, cell_energy = map(int, read_input().split())
            self.cells[cell_y][cell_x].halite_amount = cell_energy
            self.halite[cell_y, cell_x] = cell_energy

            half = self.height // 2
            cell_x_2 = cell_x + self.width if cell_x < half else cell_x - self.width
            cell_y_2 = cell_y + self.height if cell_y < half else cell_y - self.height
            for x in (cell_x, cell_x_2):
                for y in (cell_y, cell_y_2):
                    self.halite_extended[y + half, x + half] = cell_energy
