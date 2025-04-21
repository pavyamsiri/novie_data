"""Module of getter functions on a HDF5 file object."""

from __future__ import annotations

import logging
from typing import TypeAlias, TypeVar

import numpy as np
import optype as op
from h5py import Dataset as Hdf5Dataset
from h5py import File as Hdf5File
from packaging.version import Version

from ._array import GenericArray, require_dtype

__all__ = [
    "get_dataset_from_hdf5",
    "get_dataset_metadata",
    "get_file_version",
    "get_float_attr_from_hdf5",
    "get_int_attr_from_hdf5",
    "get_str_attr_from_hdf5",
    "get_string_sequence_from_hdf5",
    "read_dataset_from_hdf5_with_dtype",
]

_DType: TypeAlias = np.dtype[np.generic]
_SCT = TypeVar("_SCT", bound=np.generic)

log: logging.Logger = logging.getLogger(__name__)


def get_dataset_from_hdf5(file: Hdf5File, name: str) -> Hdf5Dataset:
    """Get a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    dataset : h5py.Dataset
        The dataset.

    """
    value = file[name]
    if not isinstance(value, Hdf5Dataset):
        msg = f"`{name}` is not a dataset of {file}!"
        raise TypeError(msg)
    return value


def get_dataset_metadata(file: Hdf5File, name: str) -> tuple[tuple[int, ...], _DType]:
    """Get the dataset's metadata.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    shape : tuple[int, ...]
        The dataset shape.
    dtype : DType
        The dataset data type.

    """
    value = file[name]
    if not isinstance(value, Hdf5Dataset):
        msg = f"`{name}` is not a dataset of {file}!"
        raise TypeError(msg)
    return (value.shape, value.dtype)


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
    if isinstance(value, op.CanInt):
        return int(value)
    msg = f"The attribute {name} is not an integer!"
    raise TypeError(msg)


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
    value : float
        The float attribute queried.

    """
    value = file.attrs[name]
    if isinstance(value, op.CanFloat):
        return float(value)
    msg = f"The attribute {name} is not a float!"
    raise TypeError(msg)


def get_str_attr_from_hdf5(file: Hdf5File, name: str) -> str:
    """Get a string attribute from a HDF5 file.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the attribute.

    Returns
    -------
    value : str
        The string attribute queried.

    """
    return str(file.attrs[name])


def get_file_version(file: Hdf5File) -> Version:
    """Return the file version.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.

    Returns
    -------
    version : Version
        The file version.

    """
    version_str: str = get_str_attr_from_hdf5(file, "version")
    return Version(version_str)


def read_dataset_from_hdf5_with_dtype(file: Hdf5File, name: str, *, dtype: type[_SCT]) -> GenericArray[_SCT]:
    """Read a dataset from a HDF5 file.

    Parameters
    ----------
    file : Hdf5File
        The HDF5 file to read from.
    name : str
        The name of the dataset to read from.
    dtype : dtype[T]
        The data type to convert to.

    Returns
    -------
    array : NDArray[T]
        The dataset's array data.

    """
    value = file[name]
    if not isinstance(value, Hdf5Dataset):
        msg = f"`{name}` is not a dataset of {file}!"
        raise TypeError(msg)
    array = np.zeros(value.shape, dtype=value.dtype)

    if array.dtype != dtype:
        log.warning("`%s` has dtype %s but %s was expected.", name, array.dtype, dtype)

    value.read_direct(array)
    return require_dtype(array, dtype)


def get_string_sequence_from_hdf5(file: Hdf5File, name: str) -> tuple[str, ...]:
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
    string_sequence : tuple[str, ...]
        The strings from the given dataset.

    """
    dset = get_dataset_from_hdf5(file, name)
    array = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(array)
    return tuple(value.decode("utf-8") for value in array)
