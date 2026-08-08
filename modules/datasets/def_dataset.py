"""
This module contains the basic dataset class and its implementations.

Classes:
    - DefDataset: Base class for datasets.

Functions:
    None

Constants:
    None

Exceptions:
    AttributeError: Raised when a field is not set before using it.
    TypeError: Raised when the input is not of the expected type.
"""

from abc import ABC
from typing import Optional, List, Union, Dict, Tuple


import torch


# Define what to export
__all__ = ["DefDataset"]


class DefDataset(ABC):
    """
    The base dataset class that enables obtaining federated datasets.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be used in the same way for all partitioners.

    Args:
        None
    """

    # *args and **kwargs are required in order that the child classes
    # can have their own __init__ method with custom arguments.
    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        self._targets: Optional[List[int]] = None
        self._classes: Optional[List[str]] = None
        self._class_to_idx: Optional[Dict[str, int]] = None
        self._split: str
        self._image_paths: List[str]
        self._embeddings: Optional[Dict[str, Tuple[int, torch.Tensor]]] = None

    @property
    def targets(self) -> List[int]:
        """
        Targets property.

        Args:
            None

        Raises:
            AttributeError: The targets field should be set before using it.

        Returns:
            List[int]: The targets property.
        """

        if self._targets is None:
            raise AttributeError("The targets field should be set before using it.")

        return self._targets

    @targets.setter
    def targets(self, targets: List[int]) -> None:
        """
        Set the targets field.

        Args:
            targets (List[int]): The targets to set.

        Raises:
            AttributeError: The targets field should not be set again.
            TypeError: The targets are not a list.

        Returns:
            None
        """

        if not isinstance(targets, list) and not isinstance(targets, torch.Tensor):
            raise TypeError("The targets should be a list.")
        elif isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        self._targets = targets

    @property
    def classes(self) -> List[str]:
        """
        Classes property.

        Args:
            None

        Raises:
            AttributeError: The classes field should be set before using it.

        Returns:
            List[str]: The classes property.
        """

        if self._classes is None:
            raise AttributeError("The classes field should be set before using it.")

        return self._classes

    @classes.setter
    def classes(self, classes: List[str]) -> None:
        """
        Set the classes field.

        Args:
            classes (List[str]): The classes to set.

        Raises:
            AttributeError: The classes field should not be set again.
            TypeError: The classes are not a list.

        Returns:
            None
        """

        if not isinstance(classes, list):
            raise TypeError("The classes should be a list.")

        self._classes = classes

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """
        Class to index property.

        Args:
            None

        Raises:
            AttributeError: The class_to_idx field should be set before using it.

        Returns:
            Dict[str, int]: The class_to_idx property.
        """

        if self._class_to_idx is None:
            raise AttributeError(
                "The class_to_idx field should be set before using it."
            )

        return self._class_to_idx

    @class_to_idx.setter
    def class_to_idx(self, class_to_idx: Dict[str, int]) -> None:
        """
        Set the class_to_idx field.

        Args:
            class_to_idx (Dict[str, int]): The class_to_idx to set.

        Raises:
            AttributeError: The class_to_idx field should not be set again.
            TypeError: The class_to_idx is not a dictionary.

        Returns:
            None
        """

        if not isinstance(class_to_idx, dict):
            raise TypeError("The class_to_idx should be a dictionary.")

        self._class_to_idx = class_to_idx

    def idx_to_class(self, idx: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Get the class name(s) from the index(es).

        Args:
            idx (Union[int, List[int]]): The index(es) to get the class name(s) from.

        Returns:
            Union[str, List[str]]: The class name(s) corresponding to the index(es).
        """

        if isinstance(idx, int):
            return self.classes[idx]
        elif isinstance(idx, list):
            return [self.classes[i] for i in idx]
        else:
            raise TypeError("The index should be an integer or a list of integers.")

    @property
    def split(self) -> str:
        """
        Split property.

        Args:
            None

        Raises:
            AttributeError: The split field should be set before using it.

        Returns:
            str: The split property.
        """

        if self._split is None:
            raise AttributeError("The split field should be set before using it.")

        return self._split

    @split.setter
    def split(self, split: str) -> None:
        """
        Set the split field.

        Args:
            split (str): The split to set.

        Raises:
            AttributeError: The split field should not be set again.
            TypeError: The split are not a str.

        Returns:
            None
        """

        if not isinstance(split, str):
            raise TypeError("The split should be a str.")

        self._split = split

    @property
    def image_paths(self) -> List[str]:
        """
        Image_path property.

        Args:
            None

        Raises:
            AttributeError: The image_paths field should be set before using it.

        Returns:
            List[str]: The image_paths property.
        """

        if self._image_paths is None:
            raise AttributeError("The image_paths field should be set before using it.")

        return self._image_paths

    @image_paths.setter
    def image_paths(self, image_paths: List[str]) -> None:
        """
        Set the image_paths field.

        Args:
            image_paths (List[str]): The image_paths to set.

        Raises:
            AttributeError: The image_paths field should not be set again.
            TypeError: The image_paths are not a list.

        Returns:
            None
        """

        if not isinstance(image_paths, list):
            raise TypeError("The image_paths should be a List[str].")

        self._image_paths = image_paths

    @property
    def embeddings(self) -> Dict[str, Tuple[int, torch.Tensor]]:
        """
        Embeddings property.

        Args:
            None
        
        Returns:
            Dict[str, Tuple[int, torch.Tensor]]: The embeddings property.
        """

        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings: Dict[str, Tuple[int, torch.Tensor]]) -> None:
        """
        Set the embeddings field.

        Args:
            embeddings (Dict[str, Tuple[int, torch.Tensor]]): The split to set.

        Raises:
            AttributeError: If the embeddings field should not be set again.
            TypeError: If the embeddings are not a dict.

        Returns:
            None
        """

        if not isinstance(embeddings, dict) and embeddings is not None:
            raise TypeError("The embeddings should be a dict.")

        self._embeddings = embeddings
