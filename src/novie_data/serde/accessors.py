"""Functions to access fields from HDF5 files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, TypeVar, cast

import numpy as np
from h5py import Dataset as Hdf5Dataset
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import AnyArray, Array, require_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence


_ST = TypeVar("_ST", bound=np.generic)
type _DType = np.dtype[np.generic]


log: logging.Logger = logging.getLogger(__name__)


def get_dataset_metadata(file: Hdf5File, name: str) -> tuple[tuple[int, ...], _DType]:
    """Get the dataset's metadata.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
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


def get_and_read_dataset_from_hdf5(file: Hdf5File, name: str) -> tuple[AnyArray, Hdf5Dataset]:
    """Read a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    array : NDArray
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


def read_dataset_from_hdf5_with_dtype(file: Hdf5File, name: str, *, dtype: type[_ST]) -> Array[_ST]:
    """Read a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.
    dtype : dtype[T]
        The data type to convert to.

    Returns
    -------
    array : NDArray[T]
        The dataset's array data.

    """
    array, _ = get_and_read_dataset_from_hdf5(file, name)
    return require_dtype(array, dtype)


def read_dataset_from_hdf5(file: Hdf5File, name: str) -> AnyArray:
    """Read a dataset from a hdf5 file.

    Parameters
    ----------
    file : h5py.File
        The hdf5 file to read from.
    name : str
        The name of the dataset to read from.

    Returns
    -------
    array : NDArray
        The dataset's array data.

    """
    array, _ = get_and_read_dataset_from_hdf5(file, name)
    return array


def get_array_from_hdf5_attrs(dataset: Hdf5Dataset, name: str) -> AnyArray:
    """Return an NDArray which is an attribute of a dataset.

    Parameters
    ----------
    dataset : h5py.Dataset
        The dataset.
    name : str
        The name of the attribute.

    Returns
    -------
    array : NDArray
        The attribute array.

    """
    value = dataset.attrs[name]
    msg = f"The `{name}` attribute is not an array of {dataset}!"
    return _verify_ndarray(value, msg)


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
    if isinstance(value, SupportsInt):
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
    if isinstance(value, SupportsFloat):
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


def _verify_ndarray(arr: object, msg: str) -> AnyArray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(msg)
    return cast(AnyArray, arr)
