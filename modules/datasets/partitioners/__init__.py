"""
Init file for partitioners modules.

Author: Matteo Caligiuri
"""

from .partitioner import Partitioner
from .dirichlet import DirichletPartitioner

__all__ = ["Partitioner", "DirichletPartitioner"]
