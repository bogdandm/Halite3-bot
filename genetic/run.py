import argparse
import sys

from .bot_model import BotArguments, BotArgumentsV3
from .core import GeneticOptimizer, GeneticOptimizerCore

VERSION = {
    "latest": BotArguments,
    "v3": BotArgumentsV3
}


# python -m genetic.run -n 16 -N 20 10 -g 5
def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mutation-rate-base", default=.01, type=float, help="Mutation std sigma (percent)")
    parser.add_argument("-n", "--bots-per-generation", default=24, type=int)
    parser.add_argument("-N", "--number-games", nargs=2, default=(8, 4), type=int,
                        metavar=("<2 players>", "<4 players>"), help="Number of games for each game type")
    parser.add_argument("-v", "--version", default="latest", choices=VERSION)
    parser.add_argument("-g", "--generations", default=4, type=int)
    parser.add_argument("-p", "--print", action="store_true", help="Print result stats")

    args = parser.parse_args(args)
    return args.mutation_rate_base, args.bots_per_generation, args.number_games, VERSION[args.version], \
           args.generations, args.print


mutation_rate_base, bots_per_generation, (count_2, count_4), version, generations, print_data \
    = parse_args(*sys.argv[1:])

if print_data:
    GeneticOptimizer(GeneticOptimizerCore(version())).print()
else:
    go = GeneticOptimizer(GeneticOptimizerCore(
        version(),
        mutation_rate=mutation_rate_base,  # lambda generation: mutation_rate_base / (generation + 1),
        bots_per_generation=bots_per_generation,
        count_2=count_2,
        count_4=count_4
    ))
    go.run(generations=generations)
