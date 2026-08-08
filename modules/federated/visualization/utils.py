# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
This module contains the utils for metrics computtion.
The class is a modified version of the one from the Flower library (license above).
The code has been modified to work with Pytorch style datasets.
removing the requirements to use Hugging Face datasets.

Classes:
    None

Functions:
    - compute_counts: Compute the counts of unique values in a given column 
      in the partitions.
    - compute_frequencies: Compute the frequencies of unique values in a given 
      column in the partitions.

Constants:
    None

Exceptions:
    ValueError: Raised when the dataset or the labels are None.
    ValueError: Raised when the column_name is not present in the dataset.
    ValueError: Raised when the unique_labels contain non-unique elements.
    Warning: Raised when the verbose names can not be established.
    ValueError: Raised when the plot_type is not in PLOT_TYPES.
    ValueError: Raised when the size_unit is not in SIZE_UNITS.
    ValueError: Raised when the partition_id_axis is not in AXIS_TYPES.
    ValueError: Raised when the data is not a Partitioner or a pd.DataFrame.

Author: Matteo Caligiuri
"""


__all__ = [
    "compute_counts",
    "compute_frequencies",
]


import warnings
from typing import Optional, Union, List

import torch
import pandas as pd

from modules.datasets.partitioners import Partitioner
from modules.federated.visualization.constants import AXIS_TYPES, PLOT_TYPES, SIZE_UNITS


def compute_counts(
    data: Union[Partitioner, pd.DataFrame],
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the counts of unique values in a given column in the partitions.

    Take into account all possible labels in dataset when computing count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Args:
        data (Union[Partitioner, pd.DataFrame]): The data to compute the counts from.
            If a Partitioner is provided, the data will be loaded from the partitions.
            If a pd.DataFrame is provided, the data will be used as is.
        column_name (str): Column name identifying label based on which the count will
            be calculated.
        verbose_names (bool): Whether to use verbose versions of the values in the
            column specified by `column_name`. The verbose values are possible to
            extract if the column is a feature of type `ClassLabel`.
        max_num_partitions (Optional[int]): The maximum number of partitions that will
            be used. If greater than the total number of partitions in a partitioner,
            it won't have an effect. If left as None, then all partitions will be used.

    Returns:
        pd.DataFrame: DataFrame where the row index represent the partition id and the

    Examples:
        Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

        >>> from flwr_datasets import FederatedDataset
        >>> from flwr_datasets.partitioner import DirichletPartitioner
        >>> from flwr_datasets.metrics import compute_counts
        >>>
        >>> fds = FederatedDataset(
        >>>     dataset="cifar10",
        >>>     partitioners={
        >>>         "train": DirichletPartitioner(
        >>>             num_partitions=20,
        >>>             partition_by="label",
        >>>             alpha=0.3,
        >>>             min_partition_size=0,
        >>>         ),
        >>>     },
        >>> )
        >>> partitioner = fds.partitioners["train"]
        >>> counts_dataframe = compute_counts(
        >>>     partitioner=partitioner,
        >>>     column_name="label"
        >>> )
    """

    if isinstance(data, Partitioner):
        partitioner = data
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise ValueError(
            "The data must be either a Partitioner or a pd.DataFrame, "
            f"but given {type(data)}."
        )

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partition = partitioner.load_partition(0)

    try:
        # Unique labels are needed to represent the correct count of each class
        # (some of the classes can have zero samples that's why this
        # adjustment is needed)
        unique_labels = partition.features[column_name].str2int(
            partition.features[column_name].names
        )
    except AttributeError:  # If the column_name is not formally a Label
        unique_labels = _get_unique_classes(
            partition_by=column_name, dataset=partition, labels=None
        )

    partition_id_to_label_absolute_size = {}
    for partition_id in range(max_num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_id_to_label_absolute_size[partition_id] = _compute_counts(
            [partition.dataset.targets[index] for index in partition.indices],
            unique_labels,
        )

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_label_absolute_size, orient="index"
    )
    dataframe.index.name = "Partition ID"
    dataframe.fillna(0, inplace=True)

    if verbose_names:
        # Adjust the column name values of the dataframe
        current_labels = dataframe.columns
        try:
            legend_names = partitioner.dataset.idx_to_class(
                [int(v) for v in current_labels]
            )
            dataframe.columns = legend_names
        except AttributeError:
            warnings.warn(
                "The verbose names can not be established. "
                "The dataset does not have the idx_to_class method.",
                stacklevel=1,
            )
    return dataframe


def _get_unique_classes(
    partition_by: int,
    dataset: Optional[torch.utils.data.Dataset] = None,
    labels: Optional[List[int]] = None,
) -> List[int]:
    """
    Get the unique classes in the dataset.

    Args:
        None

    Returns:
        List[int]: The unique classes in the dataset.
    """

    # Check that either the dataset or the labels are not None
    if dataset is None and labels is None:
        raise ValueError("Either the dataset or the labels must be provided.")

    if labels is not None and partition_by == 1:
        return labels
    elif hasattr(dataset, "targets"):
        return list(set(dataset.targets))
    else:
        # Cycle through the dataset to get all the
        # unique classes based on the partition_by index
        unique_classes = []
        for sample in dataset:
            if sample[partition_by] not in unique_classes:
                unique_classes.append(sample[partition_by])
        return unique_classes


def compute_frequencies(
    data: Union[Partitioner, pd.DataFrame],
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the frequencies of unique values in a given column in the partitions.

    The frequencies sum up to 1 for a given partition id. This function takes into
    account all possible labels in the dataset when computing the count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Args:
        data (Union[Partitioner, pd.DataFrame]): The data to compute the frequencies
            from. If a Partitioner is provided, the data will be loaded from the
            partitions. If a pd.DataFrame is provided, the data will be used as is.
        column_name (str): Column name identifying label based on which the count will
            be calculated.
        verbose_names (bool): Whether to use verbose versions of the values in the
            column specified by `column_name`. The verbose values are possible to
            extract if the column is a feature of type `ClassLabel`.
        max_num_partitions (Optional[int]): The maximum number of partitions that will
            be used. If greater than the total number of partitions in a partitioner,
            it won't have an effect. If left as None, then all partitions will be used.

    Returns:
        pd.DataFrame: DataFrame where the row index represent the partition id and the
            column index represent the unique values found in column specified by
            `column_name` (e.g. representing the labels). The value of the
            dataframe.loc[i, j] represents the ratio of the label j to the total
            number of sample of in partition i.

    Examples:
        Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

        >>> from flwr_datasets import FederatedDataset
        >>> from flwr_datasets.partitioner import DirichletPartitioner
        >>> from flwr_datasets.metrics import compute_frequencies
        >>>
        >>> fds = FederatedDataset(
        >>>     dataset="cifar10",
        >>>     partitioners={
        >>>         "train": DirichletPartitioner(
        >>>             num_partitions=20,
        >>>             partition_by="label",
        >>>             alpha=0.3,
        >>>             min_partition_size=0,
        >>>         ),
        >>>     },
        >>> )
        >>> partitioner = fds.partitioners["train"]
        >>> counts_dataframe = compute_frequencies(
        >>>     partitioner=partitioner,
        >>>     column_name="label"
        >>> )
    """

    if isinstance(data, Partitioner):
        dataframe = compute_counts(data, column_name, verbose_names, max_num_partitions)
    elif isinstance(data, pd.DataFrame):
        dataframe = data
    else:
        raise ValueError(
            "The data must be either a Partitioner or a pd.DataFrame, "
            f"but given {type(data)}."
        )
    dataframe = dataframe.div(dataframe.sum(axis=1), axis=0)
    return dataframe


def _compute_counts(
    labels: Union[list[int], list[str]], unique_labels: Union[list[int], list[str]]
) -> pd.Series:
    """
    Compute the count of labels when taking into account all possible labels.

    Also known as absolute frequency.

    Args:
        labels (Union[List[int], List[str]]): The labels from the datasets.
        unique_labels (Union[List[int], List[str]]): The reference all unique label.
            Needed to avoid missing any label, instead having the value equal to zero
            for them.

    Returns:
        pd.Series: The pd.Series with label as indices and counts as values.
    """

    if len(unique_labels) != len(set(unique_labels)):
        raise ValueError("unique_labels must contain unique elements only.")
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros


def _compute_frequencies(
    labels: Union[list[int], list[str]], unique_labels: Union[list[int], list[str]]
) -> pd.Series:
    """
    Compute the distribution of labels when taking into account all possible labels.

    Also known as relative frequency.

    Args:
        labels (Union[List[int], List[str]]): The labels from the datasets.
        unique_labels (Union[List[int], List[str]]): The reference all unique label.
            Needed to avoid missing any label, instead having the value equal to zero
            for them.

    Returns:
        pd.Series: The pd.Series with label as indices and probabilities as values.
    """

    counts = _compute_counts(labels, unique_labels)
    if len(labels) == 0:
        frequencies = counts.astype(float)
        return frequencies
    frequencies = counts.divide(len(labels))
    return frequencies


def _validate_parameters(
    plot_type: str, size_unit: str, partition_id_axis: str
) -> None:
    """
    Validate the parameters for the plot.

    Args:
        plot_type (str): The type of the plot.
        size_unit (str): The unit of the size.
        partition_id_axis (str): The axis of the partition id.

    Returns:
        None
    """

    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )
    if size_unit not in SIZE_UNITS:
        raise ValueError(
            f"Invalid size_unit: {size_unit}. Must be one of {SIZE_UNITS}."
        )
    if partition_id_axis not in AXIS_TYPES:
        raise ValueError(
            f"Invalid partition_id_axis: {partition_id_axis}. "
            f"Must be one of {AXIS_TYPES}."
        )
