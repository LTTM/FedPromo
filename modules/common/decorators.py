"""
Module with decorators.

Classes:
    None

Functions:
    register_method: Register a method to be implemented in a class.
    register_property: Register a property to be implemented in a class.
    register_new_class_elements: Register the new methods and properties in the model.
    add_dts: Add the dataset implemented by the class to the supported datasets.
    get_dts: Get the dataset implemented by the class.

Esceptions:
    None

Author: Matteo Caligiuri
        Francesco Barbato
"""

import sys
from types import MethodType
from typing import Optional, Tuple, Callable, Type
from collections import defaultdict
from inspect import signature

import torch
from torch.nn import Module


__all__ = [
    "register_method",
    "register_property",
    "register_new_class_elements",
    "add_dts",
    "add_dts_alt",
    "get_dts",
    "filter_kwargs",
]


# Auxiliarry variables used to store data
_implemented_methods = defaultdict(dict)
_implemented_properties = defaultdict(dict)

_supported_dts = defaultdict(dict)


def register_method(fn: MethodType) -> MethodType:
    """
    Register a method to be implemented in a class.

    Args:
        fn: The function to register.
    Returns:
        The registered function.
    """

    # Get the target class
    mod = sys.modules[fn.__module__]
    target_class = getattr(mod, "_target_class", None)
    assert target_class is not None, f'please specify "_target_class" in "{mod}"!'

    # Register the method
    target_method = fn.__name__
    _implemented_methods[target_class][target_method] = fn

    return fn


def register_property(fn: MethodType) -> MethodType:
    """
    Register a property to be implemented in a class.

    Args:
        fn: The function to register.
    Returns:
        The registered function.
    """

    # Get the target class
    mod = sys.modules[fn.__module__]
    target_class = getattr(mod, "_target_class", None)
    assert target_class is not None, f'please specify "_target_class" in "{mod}"!'

    # Register the property
    target_property = fn.__name__
    _implemented_properties[target_class][target_property] = fn

    return fn


def register_new_class_elements(model: Module) -> None:
    """
    Register the new methods and properties in the model.

    Args:
        model: The model to register the methods.

    Returns:
        None
    """

    # Get the method and property dictionaries filtering by the caller filename
    method_dict = _implemented_methods[model.__class__]
    property_dict = _implemented_properties[model.__class__]

    # If the method dict is not empty, register the methods
    if len(method_dict) > 0:
        for method in method_dict:
            fn = method_dict[method]
            setattr(model, method, MethodType(fn, model))

    # If the property dict is not empty, register the properties
    if len(property_dict) > 0:
        for prop in property_dict:
            value = property_dict[prop]()
            setattr(model, prop, value)


def getitem_embed(
    fun: Callable[[int], Tuple[torch.Tensor, torch.Tensor]]
) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Decorator to add the embeddings to the getitem method.

    Args:
        fun (Callable[[int], Tuple[torch.Tensor, torch.Tensor]]): The getitem method.

    Returns:
        Callable[[int], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
            The new getitem method.
    """

    def new_getitem(self, item):
        image, lb1 = fun(self, item)
        if self.embeddings is not None:
            impath = self.image_paths[item]
            if impath.startswith("/"):
                impath = impath[1:]
            lb2, embed = self.embeddings[impath]
            if lb1 != lb2:
                raise ValueError(
                    "Inconsistency bettwen cached and online dataloading Found! "
                    + "Computed and cached labels differ. You need to re-generate the cache."
                    + f"Affected dataset: {self.__class__.__name__.lower()}"
                )
        else:
            embed = None
        return image, lb1, embed

    return new_getitem


def add_dts(cls) -> None:
    """
    Add the dataset implemented by the class to the supported datasets.

    Args:
        cls: The class to add the dataset.

    Returns:
        None
    """

    # Get the number of classes from the dataset
    mod = sys.modules[cls.__module__]
    n_classes = getattr(mod, "NUM_CLASSES", None)

    # Add the dataset to the supported datasets list
    _supported_dts[cls.__name__] = {"dts": cls, "n_classes": n_classes}

    # Overwrite getitem so that it supports embedding cache
    cls.__getitem__ = getitem_embed(cls.__getitem__)

    return cls


def add_dts_alt(cls) -> None:
    """
    Add the dataset implemented by the class to the supported datasets.
    WRT the other add_dts, this function is used when the dataset use the NUM_CLASSES_ALT
    variable instead of the standard var.

    Args:
        cls: The class to add the dataset.

    Returns:
        None
    """

    # Get the number of classes from the dataset
    mod = sys.modules[cls.__module__]
    n_classes = getattr(mod, "NUM_CLASSES_ALT", None)

    # Add the dataset to the supported datasets list
    _supported_dts[cls.__name__] = {"dts": cls, "n_classes": n_classes}

    # Overwrite getitem so that it supports embedding cache
    cls.__getitem__ = getitem_embed(cls.__getitem__)

    return cls


def get_dts() -> Module:
    """
    Get the dataset implemented by the class.

    Args:
        None

    Returns:
        Module: The dataset implemented by the class.
    """

    # Get the dataset
    return _supported_dts


def filter_kwargs(cls: Type[object]) -> Callable:
    """
    Decorator to filter the keyword arguments of a class.

    Args:
        cls (Type[Object]): The class to filter the keyword arguments.

    Returns:
        Callable: The function to filter the keyword arguments.
    """

    def wrapper(*args, **kwargs) -> object:
        """
        Filter the keyword arguments of the class.

        Args:
            *args: The arguments of the class.
            **kwargs: The keyword arguments of the class.

        Returns:
            Object: The class with the filtered keyword arguments.
        """

        cls_params = signature(cls.__init__).parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in cls_params}
        return cls(*args, **valid_kwargs)

    return wrapper
