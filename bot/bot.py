import logging
import operator
from typing import Tuple

import hlt
from hlt import Position, constants
from .const import LOCAL


class Bot:
    def __init__(
            self,
            ship_fill_k=.75,
            distance_penalty_k=1.3,
            ship_limit=30,
            ship_turns_stop=100
    ):
        self.ship_fill_k = ship_fill_k
        self.distance_penalty_k = distance_penalty_k
        self.ship_limit_base = ship_limit
        self.ship_turns_stop = ship_turns_stop

        self.game = hlt.Game()
        self.callbacks = []
        self.debug_maps = {}
        self.ships_targets = {}

    def run(self):
        self.game.ready("BogdanDm")
        logging.info("Player ID: {}.".format(self.game.my_id))
        while True:
            self.game.update_frame()
            commands = self.loop()
            self.game.end_turn(commands)
            for fn in self.callbacks:
                while not fn():
                    pass

    def add_callback(self, fn):
        self.callbacks.append(fn)

    def loop(self):
        me = self.game.me
        gmap = self.game.map
        home = self.game.me.shipyard.position

        for ship in me.get_ships():
            if ship.halite_amount >= constants.MAX_HALITE * self.ship_fill_k:
                logging.info(f"Ship#{ship.id} moving home")
                ship.target = home
                yield ship.move(self.game.map.naive_navigate(ship, home))

            else:
                if gmap[ship].halite_amount / constants.MOVE_COST_RATIO > ship.halite_amount:
                    ship.target = None
                    yield ship.stay_still()
                    continue

                position = None
                radius = 8 if LOCAL else None
                surrounding = {}
                if radius:
                    field = ship.position.get_surrounding_cardinals(radius, center=True)
                else:
                    field = map(operator.attrgetter('position'), gmap)
                for coord in field:
                    if (gmap[coord].ship is None or gmap[coord].ship is ship):
                        surrounding[Position(*coord)] = gmap[coord].halite_amount
                for coord in surrounding:
                    # surrounding[coord] /= 1 + 1 / constants.MOVE_COST_RATIO
                    surrounding[coord] /= (gmap.calculate_distance(ship.position, coord) + 1) ** self.distance_penalty_k

                if surrounding:
                    max_halite: Tuple[Position, int] = max(surrounding.items(), key=operator.itemgetter(1))
                    if max_halite[1] > 0.1:
                        position = max_halite[0]

                self.debug_maps[ship.id] = surrounding

                if position:
                    if position == ship.position:
                        logging.info(f"Ship#{ship.id} collecting halite")
                        ship.target = None
                        yield ship.stay_still()
                    else:
                        logging.info(f"Ship#{ship.id} {ship.position} moving towards {position}")
                        ship.target = position
                        yield ship.move(self.game.map.naive_navigate(ship, position))
                else:
                    logging.info(f"Ship#{ship.id} does not found good halite deposit")
                    ship.target = None
                    yield ship.stay_still()

            # else:
            #     logging.info(f"Ship#{ship.id} collecting halite")
            #     yield ship.stay_still()

        ship_limit = round(self.ship_limit_base * (1 + (self.game.map.width - 32) / (64 - 32)))
        if (
                self.game.turn_number <= constants.MAX_TURNS - self.ship_turns_stop
                and len(me.get_ships()) < ship_limit
                and me.halite_amount >= constants.SHIP_COST
                and not gmap[me.shipyard].is_occupied
        ):
            for pos in gmap[me.shipyard].position.get_surrounding_cardinals():
                if gmap[pos].is_occupied:
                    break
            else:
                yield me.shipyard.spawn()
