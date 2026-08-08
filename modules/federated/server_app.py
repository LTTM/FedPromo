"""foudation-fed: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy


# Define what to export
__all__ = ["generate_server_fn"]


def generate_server_fn(strategy: Strategy, num_rounds: int) -> callable:
    """
    Generate a Flower server.

    Args:
        strategy (flwr.server.strategy.Strategy): The strategy to use.
        num_rounds (int): The number of rounds to run.

    Returns:
        callable: A function which defines a Flower server.
    """

    def server_fn(context: Context):  # pylint: disable=unused-argument
        """
        Define a Flower server.
        """

        # Define strategy
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
