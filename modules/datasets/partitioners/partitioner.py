# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This module contains the basic partitioner class and its implementations.
The class is a modified version of the partitioner class from the Flower library.
(license above)
The code has been modified to work with Pytorch style datasets.
removing the requirements to use Hugging Face datasets.

Classes:
    - Partitioner: Base class for partitioners.

Functions:
    None

Constants:
    None

Exceptions:
    AttributeError: Raised when the dataset field should be set before using it.
    ValueError: Raised when the dataset is already set.
    TypeError: Raised when the dataset is not of type `datasets.Dataset`.
    TypeError: Raised when the labels object is not of type `List[int]` or
        `Dict[Union[str, int], Union[str, int]]`.

Author: Matteo Caligiuri
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict

from torch.utils.data import Dataset


# Define what to export
__all__ = ["Partitioner"]


class Partitioner(ABC):
    """
    The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be used in the same way for all partitioners.

    Args:
        None
    """

    def __init__(self) -> None:
        self._dataset: Optional[Dataset] = None
        self._labels: Optional[List[int]] = None

    @property
    def dataset(self) -> Dataset:
        """
        Dataset property.

        Args:
            None

        Raises:
            AttributeError: If the dataset field should be set before using it (directly, via

        Returns:
            Dataset: The dataset property.
        """

        if self._dataset is None:
            raise AttributeError(
                "The dataset field should be set before using it (directly, via "
                "`load_partition` or some other method). "
            )
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        """
        Set the dataset property.

        Args:
            value (Dataset): The dataset to set.

        Raises:
            ValueError: If the dataset is already set.
            TypeError: If the dataset is not of type `datasets.Dataset`.

        Returns:
            None
        """

        if self._dataset is not None:
            raise ValueError(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )
        if not isinstance(value, Dataset):
            raise TypeError(
                f"The dataset object you want to assign to the partitioner should be "
                f"of type `datasets.Dataset` but given {type(value)}."
            )
        self._dataset = value

    @property
    def labels(self) -> List[int]:
        """
        Labels property.

        Args:
            None

        Returns:
            List[int]: The labels property
        """

        return self._labels

    @labels.setter
    def labels(
        self, value: Union[Dict[Union[str, int], Union[str, int]], List[int], List[str]]
    ) -> None:
        """
        Set the labels property.

        Args:
            value (Union[Dict[Union[str, int], Union[str, int]], List[int], List[str]]):
                The labels to set. The labels can be a list of integers or strings or a
                dictionary where the keys are integers or strings and the values are
                integers or strings. The last case represents a mapping from the used label
                keys to the original labels.

        Raises:
            TypeError: If the labels object is not of type `List[int]` or
                `Dict[Union[str, int], Union[str, int]]`.

        Returns:
            None
        """

        # If the labels are a dictionary, we need to extract the keys
        if isinstance(value, dict):
            self._labels = [int(k) for k in value]
        elif isinstance(value, list):
            self._labels = [int(l) for l in value]
        else:
            raise TypeError(
                f"The labels object you want to assign to the partitioner should be "
                f"of type `List[int]` or `Dict[Union[str, int], Union[str, int]]` but "
                f"given {type(value)}."
            )

    @abstractmethod
    def load_partition(self, partition_id: int) -> Dataset:
        """
        Load a single partition based on the partition index.

        Args:
            partition_id (int): the index that corresponds to the requested partition

        Returns:
            dataset_partition (Dataset): single dataset partition
        """

    @abstractmethod
    def create_partitions(self) -> List[Dataset]:
        """
        Create all partitions.

        Args:
            None

        Returns:
            dataset_partitions (List[Dataset]): list of dataset partitions
        """

    def is_dataset_assigned(self) -> bool:
        """
        Check if a dataset has been assigned to the partitioner.

        This method returns True if a dataset is already set for the partitioner,
        otherwise, it returns False.

        Args:
            None

        Returns:
            dataset_assigned (bool): True if a dataset is assigned, otherwise False.
        """

        return self._dataset is not None

    @property
    @abstractmethod
    def num_partitions(self) -> int:
        """
        Total number of partitions.

        Args:
            None

        Returns:
            int: The number of partitions.
        """
