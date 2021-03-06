import base64
import json
import sys
import time
from random import randint, random

import scipy.stats


class FloatArgument:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def generate(self):
        return random() * (self.max - self.min)

    def mutate(self, value, mutation_rate):
        percent_value = (value - self.min) / (self.max - self.min)
        new_percent_value = scipy.stats.norm(loc=percent_value, scale=mutation_rate).rvs()
        new_percent_value = max(new_percent_value, 0)
        new_percent_value = min(new_percent_value, 1)
        return new_percent_value * (self.max - self.min) + self.min

    def breed(self, x, y):
        return self.mutate((x + y) / 2, abs(x - y) / (self.max - self.min) / 2)


class IntegerArgument(FloatArgument):
    def generate(self):
        return randint(self.min, self.max)

    def mutate(self, value, mutation_rate):
        value = super().mutate(value, mutation_rate)
        return round(value)


def compile_args(args: dict) -> str:
    s = json.dumps(args, sort_keys=True)
    result = base64.b64encode(s.encode()).decode()
    if sys.platform.startswith("win"):
        return result
    filename = f"/tmp/{round(time.time() * 100000)}-{randint(0, 1000)}.args"
    with open(filename, "w") as f:
        f.write(result)
    return filename


class GenericBotArguments:
    AMP = "&" if sys.platform.startswith("win") else "&&"
    command = None
    version = 0

    def __iter__(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if attr_name.startswith("_") or not isinstance(attr, FloatArgument):
                continue
            yield attr_name, attr

    def generate(self) -> dict:
        return {name: arg.generate() for name, arg in self}

    def mutate(self, args: dict, mutation_rate) -> dict:
        return {name: arg.mutate(args[name], mutation_rate) for name, arg in self}

    def breed(self, args1: dict, args2: dict) -> dict:
        result = {}
        for key in args1.keys():
            result[key] = getattr(self, key).breed(args1[key], args2[key])
        return result


class BotArgumentsV3(GenericBotArguments):
    command = "cd backups/v3 %s python MyBot.py --args {args}" % GenericBotArguments.AMP
    version = 3

    ship_fill_k = FloatArgument(0.2, 1.0)


class BotArgumentsV11(BotArgumentsV3):
    command = "cd backups/v11 %s python MyBot.py --args {args}" % GenericBotArguments.AMP
    version = 11

    distance_penalty_k = FloatArgument(0.0, 2.0)
    ship_limit = IntegerArgument(10, 60)
    ship_spawn_stop_turn = FloatArgument(0.0, 1.0)
    enemy_ship_penalty = FloatArgument(0.0, 1.0)
    enemy_ship_nearby_penalty = FloatArgument(0.0, 1.0)
    same_target_penalty = FloatArgument(0.0, 1.0)
    lookup_radius = IntegerArgument(5, 25)
    ship_limit_scaling = FloatArgument(0.0, 2.0)


class BotArgumentsV12(GenericBotArguments):
    command = "cd backups/v12 %s python MyBot.py --args {args}"
    version = 12

    distance_penalty_k = FloatArgument(0.0, 2.0)
    ship_fill_k = FloatArgument(0.2, 1.0)
    ship_limit = IntegerArgument(10, 100)
    ship_spawn_stop_turn = FloatArgument(0.3, 1.0)
    enemy_ship_penalty = FloatArgument(0.0, 1.0)
    enemy_ship_nearby_penalty = FloatArgument(0.0, 1.0)
    same_target_penalty = FloatArgument(0.0, 1.0)
    ship_limit_scaling = FloatArgument(0.0, 2.0)


class BotArguments(GenericBotArguments):
    command = "python MyBot.py --local --args {args}"
    version = 13

    potential_gauss_sigma = FloatArgument(2.0, 10.0)
    contour_k = FloatArgument(0.2, 0.9)
    dropoff_my_ship = FloatArgument(0.5, 10.0)
    dropoff_enemy_ship = FloatArgument(-10.0, -0.5)
    dropoff_my_base = FloatArgument(-150.0, -10.0)
    dropoff_enemy_base = FloatArgument(-150.0, -10.0)


class BotArgumentsStayStill(GenericBotArguments):
    command = "python MyBot.py --local --args {args}"
    version = 22

    stay_still_bonus = FloatArgument(0.8, 4.0)
