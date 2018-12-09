import logging
import operator
import time
from itertools import combinations
from random import shuffle
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import skimage.measure as msr
from scipy.ndimage import gaussian_filter
from skimage import draw

from hlt import Direction, Game, Position, constants
from hlt.entity import Ship
from .const import LOCAL, V2


def ship_collecting_halite_coefficient(ship, gmap):
    ship_cargo_free = constants.MAX_HALITE - ship.halite_amount
    cell = gmap[ship.position].halite_amount or 10 ** -10
    return max((cell - ship_cargo_free) / cell, .1)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


class ShipManager:
    def __init__(self, game: Game):
        self.game = game

        self.ships: List[Ship] = []
        self.distances: Dict[Ship, int] = {}
        self.targets: Dict[Ship, Optional[Union[Position, Tuple[int, int], List[Position]]]] = {}
        self._commands: Dict[Ship, Optional[str]] = {}

    def update(self):
        self.ships = self.game.me.get_ships()

        gmap = self.game.map
        me = self.game.me
        bases = {me.shipyard.position, *(base.position for base in me.get_dropoffs())}

        self.distances = {
            ship: min(gmap.distance(ship.position, base) for base in bases)
            for ship in self.ships
        }
        self.ships = [ship for ship, distance in sorted(self.distances.items(), key=operator.itemgetter(1))]
        self.targets = {ship: None for ship in self.ships}

    def add_target(self, ship: Ship, target: Optional[Union[Position, Tuple[int, int], List[Position]]]):
        self.targets[ship] = target

    # def get_ships_lt(self, ship: Ship):
    #     ix = self.ships.index(ship)
    #     return self.ships[:ix]
    #
    # def get_ships_gt(self, ship: Ship):
    #     ix = self.ships.index(ship)
    #     return self.ships[ix:]


class Bot:
    def __init__(
            self,
            distance_penalty_k=1.3,
            ship_fill_k=.7,
            ship_limit=45,
            ship_spawn_stop_turn=.5,
            dropoff_spawn_stop_turn=.7,
            enemy_ship_penalty=.3,
            enemy_ship_nearby_penalty=.6,
            same_target_penalty=.7,
            turn_time_warning=1.8,
            ship_limit_scaling=1.2,  # ship_limit_scaling + 1 multiplier on large map
            halite_threshold=1 / 20,

            potential_gauss_sigma=6.,
            contour_k=.4,
            dropoff_threshold=.02,
            dropoff_my_ship=1,
            dropoff_enemy_ship=-.1,
            dropoff_my_base=-75,
            dropoff_enemy_base=-30
    ):
        self.distance_penalty_k = distance_penalty_k
        self.ship_fill_k_base = ship_fill_k
        self.ship_limit_base = ship_limit
        self.ship_limit = self.ship_limit_base
        self.ship_turns_stop = ship_spawn_stop_turn
        self.dropoff_spawn_stop_turn = dropoff_spawn_stop_turn
        self.enemy_ship_penalty = enemy_ship_penalty
        self.enemy_ship_nearby_penalty = enemy_ship_nearby_penalty
        self.same_target_penalty = same_target_penalty
        self.turn_time_warning = turn_time_warning
        self.ship_limit_scaling = ship_limit_scaling
        self.halite_threshold = halite_threshold
        self.stay_still_bonus = None

        self.potential_gauss_sigma = potential_gauss_sigma
        self.contour_k = contour_k
        self.dropoff_threshold = dropoff_threshold
        self.dropoff_my_ship = dropoff_my_ship
        self.dropoff_enemy_ship = dropoff_enemy_ship
        self.dropoff_my_base = dropoff_my_base
        self.dropoff_enemy_base = dropoff_enemy_base

        self.game = Game()
        self.ship_manager = ShipManager(self.game)
        self.building_dropoff: Optional[Tuple[Position, Ship]] = None
        self.ships_to_home: Set[int] = set()
        self.max_ships_reached = 0
        self.collect_ships_stage_started = False
        self.fast_mode = False

        self.callbacks = []
        self.debug_maps = {}

    def run(self):
        map_creator = lambda: np.empty(shape=(self.game.map.height, self.game.map.width), dtype=float)
        self.mask = map_creator()
        self.blurred_halite = map_creator()
        self.filtered_halite = map_creator()
        self.weighted_halite = map_creator()
        self.per_ship_mask = map_creator()

        self.ship_limit = round(
            self.ship_limit_base * (1 + (self.game.map.width - 32) / (64 - 32) * self.ship_limit_scaling)
        )

        self.distances = np.zeros((self.game.map.width, self.game.map.height), dtype=float)
        #       lambda d: 3.0 * (1 - math.e ** -((d / 2.0) ** 2)) + 1.0 + d
        d_map = lambda d: (d + 1) ** self.distance_penalty_k
        for x in range(self.game.map.width):
            for y in range(self.game.map.height):
                self.distances[y, x] = d_map(self.game.map.distance((0, 0), (x, y)))

        self.game.ready("BogdanDm_V21" + ("_V2" if V2 else ""))
        logging.info("Player ID: {}.".format(self.game.my_id))

        self.stay_still_bonus = 1 + 1 / constants.MOVE_COST_RATIO
        if len(self.game.players) == 4:
            self.ship_limit //= 1.2

        while True:
            time1 = time.time()
            self.game.update_frame()
            self.ship_manager.update()

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
        moves: Dict[Ship, Iterable[Tuple[int, int]]] = {}
        bases = {home, *(base.position for base in me.get_dropoffs())}

        # Check whether new dropoff can be created or not
        if self.building_dropoff is None:
            if 10 < self.game.turn_number < constants.MAX_TURNS * self.dropoff_spawn_stop_turn:
                dropoff_position, dropoff_ship = self.dropoff_builder()
            else:
                dropoff_position, dropoff_ship = None, None
        else:
            dropoff_position, dropoff_ship = self.building_dropoff
            if not dropoff_ship.exists:
                dropoff_position, dropoff_ship = None, None
                self.building_dropoff = None

        if dropoff_ship is not None and dropoff_position is not None:
            if me.halite_amount > constants.DROPOFF_COST:
                if dropoff_ship.position == dropoff_position:
                    # If ship in place wait until we have enough halite to build dropoff
                    if me.halite_amount + dropoff_ship.halite_amount > constants.DROPOFF_COST:
                        yield dropoff_ship.make_dropoff()
                        me.halite_amount -= constants.DROPOFF_COST - dropoff_ship.halite_amount
                        logging.info(f"Building dropoff {dropoff_position} from Ship#{dropoff_ship.id}")
                        self.building_dropoff = None
                    else:
                        yield dropoff_ship.stay_still()
                else:
                    # Move to new dropoff position
                    self.ship_manager.add_target(dropoff_ship, dropoff_position)
                    self.building_dropoff = dropoff_position, dropoff_ship
                    logging.info(f"Ship#{dropoff_ship.id} moving to dropoff position {dropoff_position}")
                    path = gmap.a_star_path_search(dropoff_ship.position, dropoff_position)
                    moves[dropoff_ship] = (gmap.normalize_direction(path[0] - dropoff_ship.position),)
            else:
                yield dropoff_ship.stay_still()

        # General mask
        # Contains:
        # * enemy ship penalty
        # * penalty for cells around enemy ships (4 directions)
        # * friendly dynamic penalty based on ship cargo / cell halite ratio
        self.mask.fill(1.)
        for player in self.game.players.values():
            for ship in player.get_ships():
                if player is not me:
                    self.mask[ship.position.y, ship.position.x] *= self.enemy_ship_penalty
                    for dir in Direction.All:
                        pos = gmap.normalize(ship.position + dir)
                        self.mask[pos.y, pos.x] *= self.enemy_ship_nearby_penalty
                else:
                    self.mask[ship.position.y, ship.position.x] *= ship_collecting_halite_coefficient(ship, gmap)

        if not self.fast_mode:
            self.blurred_halite = gaussian_filter(gmap.halite, sigma=2, truncate=2.0, mode='wrap')
            gbh_min, gbh_max = self.blurred_halite.min(), self.blurred_halite.max()
            self.blurred_halite -= gbh_min
            self.blurred_halite /= gbh_max - gbh_min
            if self.callbacks:
                self.debug_maps["gbh_norm"] = self.blurred_halite
            self.filtered_halite *= self.blurred_halite / 2 + .5

        max_halite: float = gmap.halite.max()
        self.filtered_halite[:, :] = gmap.halite[:, :]
        self.filtered_halite[self.filtered_halite < max_halite * self.halite_threshold] = 0
        self.filtered_halite *= self.mask

        # Generate moves for ships from mask and halite field
        for ship in reversed(self.ship_manager.ships):
            if ship == dropoff_ship:
                continue

            # Ship unloading
            to_home = ship in self.ships_to_home
            if to_home and ship.position in bases:
                logging.info(f"Ship#{ship.id} unload")
                self.ships_to_home.remove(ship)
                to_home = False

            if to_home or ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                self.ships_to_home.add(ship)
                ship_target = min(
                    bases,
                    key=lambda base: gmap.distance(ship.position, base)
                )
                path = gmap.a_star_path_search(ship.position, ship_target)
                self.ship_manager.add_target(ship, path)
                moves[ship] = (gmap.normalize_direction(path[0] - ship.position),)

            # Ship has too low halite to move
            elif gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                yield ship.stay_still()
                continue

            else:
                self.per_ship_mask.fill(1.0)
                for another_ship, target in self.ship_manager.targets.items():
                    if isinstance(target, list):
                        target = target[0]
                    if another_ship is ship or target is None or target == home:
                        continue
                    self.per_ship_mask[target[1], target[0]] *= self.same_target_penalty
                self.weighted_halite[:, :] = self.filtered_halite[:, :]

                # Apply masks
                distances = np.roll(self.distances, ship.position.y, axis=0)
                distances = np.roll(distances, ship.position.x, axis=1)
                self.weighted_halite /= distances
                self.weighted_halite *= self.per_ship_mask
                self.weighted_halite[ship.position.y, ship.position.x] /= ship_collecting_halite_coefficient(ship, gmap)

                if self.callbacks:
                    self.debug_maps[ship.id] = self.weighted_halite.copy()

                # Found max point
                y, x = np.unravel_index(self.weighted_halite.argmax(), self.weighted_halite.shape)
                max_halite = self.weighted_halite[y, x]
                if max_halite > 0.1:
                    target = Position(x, y)
                    if target is dropoff_position:
                        continue
                    if target == ship.position:
                        logging.info(f"Ship#{ship.id} collecting halite")
                        self.ship_manager.add_target(ship, None)
                        yield ship.stay_still()
                    else:
                        logging.info(f"Ship#{ship.id} {ship.position} moving towards {target}")
                        self.ship_manager.add_target(ship, target)
                        moves[ship] = list(self.game.map.get_unsafe_moves(ship.position, target))
                else:
                    logging.info(f"Ship#{ship.id} does not found good halite deposit")
                    self.ship_manager.add_target(ship, None)
                    yield ship.stay_still()

        # Resolve moves into commands in 2 iterations
        moves: List[Tuple[Ship, Iterable[Tuple[int, int]]]] = list(moves.items())
        for i in range(2):
            commands, moves = self.resolve_moves(moves)
            yield from commands

        # Remaining ship has not any safe moves
        for ship, d in moves:
            logging.debug(f"Unsafe move: {ship} -> {d}")
            yield ship.stay_still()

        # Max ships: base at 32 map size, 2*base at 64 map size
        if (
                (dropoff_position is None or me.halite_amount >= constants.SHIP_COST + constants.DROPOFF_COST)
                and self.game.turn_number <= constants.MAX_TURNS * self.ship_turns_stop
                and me.halite_amount >= constants.SHIP_COST
                and not gmap[me.shipyard].is_occupied
                and self.max_ships_reached <= 2
        ):
            if len(my_ships) < self.ship_limit:
                yield me.shipyard.spawn()
            else:
                self.max_ships_reached += 1

    def dropoff_builder(self) -> Tuple[Optional[Position], Optional[Ship]]:
        GAUSS_SIGMA = 2.
        MIN_HALITE_PER_REGION = 16000

        gmap = self.game.map
        bases = {self.game.me.shipyard.position, *(base.position for base in self.game.me.get_dropoffs())}
        halite_ex = gmap.halite_extended.copy()
        h, w, d = gmap.height, gmap.width, 2

        # Fill map border with zeros
        halite_ex[:d, 0:2 * w] = 0
        halite_ex[-d:, 0:2 * w] = 0
        halite_ex[0:2 * h, :d] = 0
        halite_ex[0:2 * h, -d:] = 0
        halite_ex_gb = gaussian_filter(halite_ex, sigma=GAUSS_SIGMA, truncate=2.0, mode='wrap')

        # Find contours of high halite deposits
        contours = msr.find_contours(halite_ex_gb, halite_ex_gb.max() * self.contour_k, fully_connected="high")
        masks = [poly2mask(c[:, 0], c[:, 1], halite_ex.shape) for c in contours]

        # Remove small halite chunks
        good_contours = []
        for c, m in zip(contours, masks):
            if np.sum(np.multiply(gmap.halite_extended, m)) > MIN_HALITE_PER_REGION:
                good_contours.append(c)

        # Remove offset of contour points
        contours = [contour - h // 2 for contour in good_contours]

        # Set up possible dropoff points
        points = {tuple(float(round(x)) for x in point)
                  for contour in contours
                  for point in contour}
        points = [list(point) for point in points
                  if all(0 <= x < h for x in point) and Position(*map(int, point)) not in bases]
        points = np.array(points, dtype=np.int)
        if points.shape[0] == 0:
            return None, None

        # Create potential field
        potential_field = np.zeros(gmap.halite.shape, dtype=np.float)
        for player in self.game.players.values():
            me = player is self.game.me
            for ship in player.get_ships():
                if me:
                    potential_field[ship.position.y, ship.position.x] += self.dropoff_my_ship
                else:
                    potential_field[ship.position.y, ship.position.x] += self.dropoff_enemy_ship
            for base in player.get_dropoffs():
                if me:
                    potential_field[base.position.y, base.position.x] += self.dropoff_my_base
                else:
                    potential_field[base.position.y, base.position.x] += self.dropoff_enemy_base
            pos = player.shipyard.position
            potential_field[pos.y, pos.x] += self.dropoff_my_base if me else self.dropoff_enemy_base
        potential_field = gaussian_filter(potential_field, sigma=self.potential_gauss_sigma, truncate=3.0, mode='wrap')

        if self.callbacks:
            self.debug_maps['dropoff'] = potential_field

        # Overlay points and potential_field
        potential_field_filtered = np.full(potential_field.shape, 0, dtype=np.float)
        x, y = points[:, 0], points[:, 1]
        potential_field_filtered[x, y] = potential_field[x, y]

        self.debug_maps['dropoff_2'] = potential_field_filtered
        y, x = np.unravel_index(potential_field_filtered.argmax(), potential_field_filtered.shape)
        value = potential_field_filtered[y, x]
        logging.debug(f"DROPOFF -> {value:.6f}")
        if value < self.dropoff_threshold:
            return None, None

        ships = self.game.me.get_ships()
        target = Position(x, y)
        distance = lambda ship: gmap.distance(ship.position, target)
        # ship_filter= lambda ship: ship.halite_amount / constants.MAX_HALITE < self.ship_fill_k * .75
        ship = min(ships, key=distance)
        return target, ship

    @property
    def collect_ships_stage(self):
        if self.collect_ships_stage_started:
            return True
        if self.game.turn_number + 3 < constants.MAX_TURNS - self.game.map.width:
            return False
        me = self.game.me
        gmap = self.game.map
        distances = [gmap.distance(ship.position, me.shipyard.position)
                     for ship in me.get_ships()]
        distances = sorted(distances)
        if not distances:
            return False
        if constants.MAX_TURNS - self.game.turn_number - 5 <= distances[-1]:
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
