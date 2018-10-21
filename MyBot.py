#!/usr/bin/env python3
# Python 3.7

import logging
import sys

from bot.bot import Bot

bot = Bot()
if "--plot" in sys.argv:
    from bot.plotter import Plotter
    plotter = Plotter(bot)
logging.info("Initialization finished")
bot.run()
