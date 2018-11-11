import logging
import operator
import time
from itertools import combinations
from random import shuffle
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

import hlt
from hlt import Direction, Position, constants
from hlt.entity import Ship
from .const import LOCAL, V2


def ship_collecting_halite_coefficient(ship, gmap):
    ship_cargo_free = constants.MAX_HALITE - ship.halite_amount
    cell = gmap[ship.position].halite_amount or 10 ** -10
    return max((cell - ship_cargo_free) / cell, .1)


class Bot:
    def __init__(
            self,
            distance_penalty_k=1.3,
            ship_fill_k=.7,
            ship_limit=30,
            ship_spawn_stop_turn=.5,
            enemy_ship_penalty=.1,
            enemy_ship_nearby_penalty=.3,
            same_target_penalty=.7,
            turn_time_warning=1.8,
            ship_limit_scaling=1  # ship_limit_scaling + 1 multiplier on large map
    ):
        self.distance_penalty_k = distance_penalty_k
        self.ship_fill_k_base = ship_fill_k
        self.ship_limit_base = ship_limit
        self.ship_limit = self.ship_limit_base
        self.ship_turns_stop = ship_spawn_stop_turn
        self.enemy_ship_penalty = enemy_ship_penalty
        self.enemy_ship_nearby_penalty = enemy_ship_nearby_penalty
        self.same_target_penalty = same_target_penalty
        self.stay_still_bonus = None
        self.turn_time_warning = turn_time_warning
        self.ship_limit_scaling = ship_limit_scaling

        self.game = hlt.Game()
        self.ships_targets: Dict[Ship, Tuple[int, int]] = {}
        self.ships_to_home: Set[int] = set()
        self.max_ships_reached = 0
        self.collect_ships_stage_started = False
        self.fast_mode = False

        self.callbacks = []
        self.debug_maps = {}

    def run(self):
        map_creator = lambda: np.empty(shape=(self.game.map.height, self.game.map.width), dtype=float)
        self.mask = map_creator()
        self.per_ship_mask = map_creator()
        self.blurred_halite = map_creator()
        self.weights = map_creator()

        self.ship_limit = round(
            self.ship_limit_base * (1 + (self.game.map.width - 32) / (64 - 32) * self.ship_limit_scaling)
        )

        self.distances = np.zeros((self.game.map.width, self.game.map.height))
        #       lambda d: 3.0 * (1 - math.e ** -((d / 2.0) ** 2)) + 1.0 + d
        d_map = lambda d: (d + 1) ** self.distance_penalty_k
        for x in range(self.game.map.width):
            for y in range(self.game.map.height):
                self.distances[y][x] = d_map(self.game.map.distance((0, 0), (x, y)))

        self.game.ready("BogdanDm" + ("_V2" if V2 else ""))
        logging.info("Player ID: {}.".format(self.game.my_id))

        self.stay_still_bonus = 1 + 1 / constants.MOVE_COST_RATIO
        if len(self.game.players) == 4:
            self.ship_limit //= 1.2

        while True:
            time1 = time.time()
            self.game.update_frame()

            my_ships = {ship.id: ship for ship in self.game.me.get_ships()}
            # Update ships and delete destroyed ones
            self.ships_targets = {
                my_ships[ship.id]: target
                for ship, target in self.ships_targets.items()
                if ship.id in my_ships
            }

            commands = self.loop() if not self.collect_ships_stage else self.collect_ships()
            self.game.end_turn(commands)

            for fn in self.callbacks:
                while not fn():
                    pass

            turn_time = time.time() - time1
            if not LOCAL and not self.fast_mode and turn_time > self.turn_time_warning:
                self.fast_mode = True
                logging.info("Fast mod: ENABLED")
            elif self.fast_mode and turn_time < self.turn_time_warning - .5:
                self.fast_mode = False
                logging.info("Fast mod: DISABLED")

    def add_callback(self, fn):
        self.callbacks.append(fn)

    @property
    def ship_fill_k(self):
        return self.ship_fill_k_base * max(2 / 3, min(1, (1 - self.game.turn_number / constants.MAX_TURNS) * 2))

    def loop(self):
        me = self.game.me
        my_ships = me.get_ships()
        gmap = self.game.map
        home = self.game.me.shipyard.position

        # General mask
        # Contains:
        # * enemy ship penalty
        # * penalty for cells around enemy ships (4 directions)
        # * friendly dynamic penalty based on ship cargo / cell halite ratio
        self.mask.fill(1.)
        for player in self.game.players.values():
            for ship in player.get_ships():
                if player is not me:
                    self.mask[ship.position.y][ship.position.x] *= self.enemy_ship_penalty
                    for dir in Direction.All:
                        pos = gmap.normalize(ship.position + dir)
                        self.mask[pos.y][pos.x] *= self.enemy_ship_nearby_penalty
                else:
                    self.mask[ship.position.y][ship.position.x] *= ship_collecting_halite_coefficient(ship, gmap)

        if not self.fast_mode:
            for y, row in enumerate(gmap.cells):
                for x, cell in enumerate(row):
                    self.blurred_halite[y][x] = cell.halite_amount
            self.blurred_halite = gaussian_filter(self.blurred_halite, sigma=2, truncate=2.0, mode='wrap')
            gbh_min, gbh_max = self.blurred_halite.min(), self.blurred_halite.max()
            self.blurred_halite -= gbh_min
            self.blurred_halite /= gbh_max - gbh_min
            if self.callbacks:
                self.debug_maps["gbh_norm"] = self.blurred_halite

        # Generate moves for ships from mask and halite field
        moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]] = []
        for ship in my_ships:
            # Ship unloading
            to_home = ship in self.ships_to_home
            if to_home and ship.position == home:
                logging.info(f"Ship#{ship.id} unload")
                self.ships_to_home.remove(ship)
                to_home = False

            if to_home or ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                self.ships_to_home.add(ship)
                self.ships_targets[ship] = path = gmap.a_star_path_search(ship.position, home)
                moves.append((
                    ship,
                    (gmap.normalize_direction(path[0] - ship.position),)
                ))

            else:
                self.per_ship_mask.fill(1.0)
                for another_ship, target in self.ships_targets.items():
                    if isinstance(target, list):
                        target = target[0]
                    if another_ship is ship or target is None or target == home:
                        continue
                    self.per_ship_mask[target[1]][target[0]] *= self.same_target_penalty

                # Ship has too low halite to move
                if gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                    # self.ships_targets[ship] = None
                    yield ship.stay_still()
                    continue

                position = None
                for coord in map(operator.attrgetter('position'), gmap):
                    self.weights[coord.y][coord.x] = gmap[coord].halite_amount

                # Apply masks
                if not self.fast_mode:
                    self.weights *= self.blurred_halite / 2 + .5
                self.weights *= self.mask
                self.weights *= self.per_ship_mask
                distances = np.roll(self.distances, ship.position.y, axis=0)
                distances = np.roll(distances, ship.position.x, axis=1)
                self.weights /= distances
                self.weights[ship.position.y][ship.position.x] /= ship_collecting_halite_coefficient(ship, gmap)

                if self.callbacks:
                    self.debug_maps[ship.id] = self.weights.copy()

                # Found max point
                y, x = np.unravel_index(self.weights.argmax(), self.weights.shape)
                max_halite: Tuple[Position, int] = (Position(x, y), self.weights[y][x])
                if max_halite[1] > 0.1:
                    position = max_halite[0]

                if position:
                    if position == ship.position:
                        logging.info(f"Ship#{ship.id} collecting halite")
                        self.ships_targets[ship] = None
                        yield ship.stay_still()
                    else:
                        logging.info(f"Ship#{ship.id} {ship.position} moving towards {position}")
                        self.ships_targets[ship] = position
                        moves.append((ship, list(self.game.map.get_unsafe_moves(ship.position, position))))
                else:
                    logging.info(f"Ship#{ship.id} does not found good halite deposit")
                    self.ships_targets[ship] = None
                    yield ship.stay_still()

        # Resolve moves into commands in 2 iterations
        for i in range(2):
            commands, moves = self.resolve_moves(moves)
            yield from commands

        # Remaining ship has not any safe moves
        for ship, d in moves:
            logging.debug(f"Unsafe move: {ship} -> {d}")
            yield ship.stay_still()

        # Max ships: base at 32 map size, 2*base at 64 map size
        if (
                self.game.turn_number <= constants.MAX_TURNS * self.ship_turns_stop
                and me.halite_amount >= constants.SHIP_COST
                and not gmap[me.shipyard].is_occupied
                and self.max_ships_reached <= 2
        ):
            if len(my_ships) < self.ship_limit:
                yield me.shipyard.spawn()
            else:
                self.max_ships_reached += 1

    @property
    def collect_ships_stage(self):
        if self.collect_ships_stage_started:
            return True
        if self.game.turn_number + 5 < constants.MAX_TURNS - self.game.map.width / 2:
            return False
        me = self.game.me
        gmap = self.game.map
        distances = [gmap.distance(ship.position, me.shipyard.position)
                     for ship in me.get_ships()]
        distances = sorted(distances)
        if len(distances) > 10:
            distances = distances[:-4]
        if constants.MAX_TURNS - self.game.turn_number + 5 >= distances[-1]:
            self.collect_ships_stage_started = True
            return True

    def get_ship_parking_spot(self, ship: 'Ship'):
        gmap = self.game.map
        home = self.game.me.shipyard.position
        s = gmap.width // 2
        x, y = gmap.normalize(ship.position - home + (s, s))
        if 0 <= x < s:
            hor = 0
        elif x == s:
            hor = 0 if y > s else 1
        else:
            hor = 1

        if 0 <= y < s:
            vert = 0
        elif y == s:
            vert = 0 if x > s else 1
        else:
            vert = 1

        return {
            (0, 0): (0, -1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
            (0, 1): (-1, 0),
        }[hor, vert]

    def collect_ships(self):
        gmap = self.game.map
        me = self.game.me
        home = me.shipyard.position

        collect_points = [home + d for d in Direction.All]
        # points = {p: [] for p in collect_points}
        moves = []
        for ship in me.get_ships():
            if ship.position not in collect_points and ship.position != home:
                point = home + self.get_ship_parking_spot(ship)
                # points[point].append(ship)
                ship_moves = list(gmap.get_unsafe_moves(ship.position, point))
                shuffle(ship_moves)
                moves.append((ship, ship_moves))
            else:
                direction = next(gmap.get_unsafe_moves(ship.position, home), None)
                if direction:
                    yield gmap.update_ship_position(ship, direction)
                else:
                    yield ship.stay_still()

        for i in range(4):
            commands, moves = self.resolve_moves(moves)
            yield from commands

        for ship, d in moves:
            logging.debug(f"Unsafe move: {ship} -> {d}")
            yield ship.stay_still()

    def resolve_moves(self, moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]]):
        gmap = self.game.map
        me = self.game.me

        tmp_moves = moves[:]
        unsafe_moves = []
        swapped_ships: Set[Ship] = set()
        commands = []

        # Resolve all regular movements until all remaining moves are not safe
        changing = True
        while changing and tmp_moves:
            changing = False
            while tmp_moves:
                ship, directions = tmp_moves.pop()
                for d in directions:
                    cell = gmap[ship.position + d]
                    if not cell.is_occupied or cell.is_occupied_base(me):
                        logging.info(f"Ship#{ship.id} moves {Direction.Names[d]}")
                        commands.append(gmap.update_ship_position(ship, d))
                        changing = True
                        break
                else:
                    unsafe_moves.append((ship, directions))

            tmp_moves, unsafe_moves = unsafe_moves, []

        # Perform "ship swap" movements
        for a, b in combinations(tmp_moves, 2):
            resolved = False
            ship1, ld1 = a
            ship2, ld2 = b
            if ship1 in swapped_ships or ship2 in swapped_ships:
                continue
            for d1 in ld1:
                if resolved:
                    break
                for d2 in ld2:
                    if (
                            gmap.normalize(ship1.position + d1) == ship2.position
                            and ship1.position == gmap.normalize(ship2.position + d2)
                    ):
                        resolved = True
                        swapped_ships.add(ship1)
                        swapped_ships.add(ship2)
                        commands.extend(gmap.swap_ships(ship1, ship2))

        tmp_moves = [(ship, d) for ship, d in tmp_moves if ship not in swapped_ships]

        return commands, tmp_moves
