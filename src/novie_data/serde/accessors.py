"""Functions to access fields from HDF5 files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from h5py import Dataset as Hdf5Dataset
from h5py import File as Hdf5File

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

log: logging.Logger = logging.getLogger(__name__)


def get_and_read_dataset_from_hdf5(file: Hdf5File, name: str) -> tuple[NDArray[Any], Hdf5Dataset]:
    """Read a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    array : NDArray[Any]
        The dataset's array data.
    dataset : h5py.Dataset
        The dataset.

    """
    value = file[name]
    if not isinstance(value, Hdf5Dataset):
        msg = f"`{name}` is not a dataset of {file}!"
        raise TypeError(msg)
    array = np.zeros(value.shape, dtype=value.dtype)
    value.read_direct(array)
    return array, value


def read_dataset_from_hdf5(file: Hdf5File, name: str) -> NDArray[Any]:
    """Read a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    array : NDArray[Any]
        The dataset's array data.

    """
    array, _ = get_and_read_dataset_from_hdf5(file, name)
    return array


def get_array_from_hdf5_attrs(dataset: Hdf5Dataset, name: str) -> NDArray[Any]:
    """Return an NDArray which is an attribute of a dataset.

    Parameters
    ----------
    dataset : h5py.Dataset
        The dataset.
    name : str
        The name of the attribute.

    Returns
    -------
    array : NDArray[Any]
        The attribute array.

    """
    value = dataset.attrs[name]
    if not isinstance(value, np.ndarray):
        msg = f"The `{name}` attribute is not an array of {dataset}!"
        raise TypeError(msg)
    return value


def get_string_sequence_from_hdf5(file: Hdf5File, name: str) -> Sequence[str]:
    """Read a dataset from a HDF5 file and return it as a sequence of strings.

    This assumes that the dataset contains an array of strings.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    string_sequence : Sequence[str]
        The strings from the given dataset.

    """
    array, _ = get_and_read_dataset_from_hdf5(file, name)
    string_sequence: Sequence[str] = [value.decode("utf-8") for value in array]
    return string_sequence


def get_int_attr_from_hdf5(file: Hdf5File, name: str) -> int:
    """Get an integer attribute from a HDF5 file.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the attribute.

    Returns
    -------
    value : int
        The integer attribute queried.

    """
    value = file.attrs[name]
    return int(cast(int, value))


def get_float_attr_from_hdf5(file: Hdf5File, name: str) -> float:
    """Get a float attribute from a HDF5 file.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the attribute.

    Returns
    -------
    value : int
        The float attribute queried.

    """
    value = file.attrs[name]
    return float(cast(float, value))
