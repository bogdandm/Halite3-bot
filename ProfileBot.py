#!/usr/bin/env python3
# Python 3.7

import cProfile
import logging

from bot.bot import Bot

pr = cProfile.Profile()
pr.enable()

bot = Bot()
logging.info("Initialization finished")

disabled = False


def disable_pr():
    global disabled, pr
    if bot.game.turn_number == 90:
        disabled = True
        pr.disable()
        pr.dump_stats("profile.pstat")
    return True


bot.add_callback(disable_pr)
bot.run()
