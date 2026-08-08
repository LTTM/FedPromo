"""
Init file for the modules package.

Author: Matteo Caligiuri
"""

## Import general modules
import os
import logging


## Set the necessary environment variables
# Matplotlib
# Set the backend for matplotlib to TkAgg
# This is needed to avoid that the debugger over the ssh connection crashes
# os.environ["MPLBACKEND"] = "TkAgg"  # pylint: disable=wrong-import-position

# RAY
# Allow to print the same log multiple times
os.environ["RAY_DEDUP_LOGS"] = "0"  # pylint: disable=wrong-import-position
os.environ["RAY_COLOR_PREFIX"] = "1"  # pylint: disable=wrong-import-position
# Enable ray debug mode
ray_debug = os.environ.get("RAY_DEBUG", "0")  # pylint: disable=wrong-import-position
if ray_debug == "1":
    os.environ["RAY_DEBUG"] = "1"  # pylint: disable=wrong-import-position
    os.environ["RAY_DEBUG_POST_MORTEM"] = "1"  # pylint: disable=wrong-import-position
else:
    os.environ["RAY_DEBUG"] = "0"
    os.environ["RAY_DEBUG_POST_MORTEM"] = "0"


## Set custom backend for matplotlib
import matplotlib

matplotlib.use("Agg", force=True)  # pylint: disable=wrong-import-position


## Import the necessary modules
from .datasets import *
from .federated import *
from .losses import *
from .models import *
from .trainers import *
from .common import *


## Uniform flwr logging
# Remove the console handler of the flwr logger
# Required to have proper logging with Hydra
flwr_logger = logging.getLogger("flwr")
for handler in flwr_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        flwr_logger.removeHandler(handler)
