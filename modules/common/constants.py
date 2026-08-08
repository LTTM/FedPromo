"""
This file contains the constants that are used in the project

Author: Matteo Caligiuri
"""

# Section of the whole model that could be trained in the clients

# During the FED training, the client will train only the specified module
ALLOWED_MODULE2TRAIN = ["all", "main", "head"]

# Accepted logger types for the data logger
ACCEPTED_DATA_LOGGER = ["mlflow", "tensorboard"]
