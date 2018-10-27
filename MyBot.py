#!/usr/bin/env python3
# Python 3.7

import logging
import sys

from bot.bot import Bot

if "--args" in sys.argv:
    import base64, json

    i = sys.argv.index("--args")
    args = sys.argv[i + 1]
    args = json.loads(base64.b64decode(args).decode())
else:
    args = {}
bot = Bot(**args)
if "--plot" in sys.argv:
    from bot.plotter import Plotter
    plotter = Plotter(bot)
logging.info("Initialization finished")
bot.run()
