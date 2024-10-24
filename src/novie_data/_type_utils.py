"""Useful types."""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np

_SCT = TypeVar("_SCT", bound=np.generic)

AnyArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
Array = np.ndarray[tuple[int, ...], np.dtype[_SCT]]
Array1D = np.ndarray[tuple[int], np.dtype[_SCT]]
Array2D = np.ndarray[tuple[int, int], np.dtype[_SCT]]
Array3D = np.ndarray[tuple[int, int, int], np.dtype[_SCT]]
Array4D = np.ndarray[tuple[int, int, int, int], np.dtype[_SCT]]


def require_dtype(arr: AnyArray, dtype: type[_SCT]) -> Array[_SCT]:
    """Convert the given array to the given dtype.

    This is mostly used to help with static type checking.

    Parameters
    ----------
    arr : AnyArray
        The array to convert.
    dtype : type[_SCT]
        The data type to convert to.

    Returns
    -------
    converted_arr : Array[_SCT]
        The converted array.

    """
    return arr.astype(dtype)


def verify_array_is_1d(arr: Array[_SCT]) -> Array1D[_SCT]:
    """Verify that the given array is 1D.

    This is mostly used to cast the given array to a 1D array.

    Parameters
    ----------
    arr : Array[_SCT]
        The array to verify.

    Returns
    -------
    verified_arr : Array1D[_SCT]
        The verified 1D array.

    """
    target_num_dim: int = 1
    if len(arr.shape) != target_num_dim:
        msg = f"The given array is not {target_num_dim}D! Got {len(arr.shape)}D instead..."
        raise TypeError(msg)
    return cast(Array1D[_SCT], arr)


def verify_array_is_2d(arr: Array[_SCT]) -> Array2D[_SCT]:
    """Verify that the given array is 2D.

    This is mostly used to cast the given array to a 2D array.

    Parameters
    ----------
    arr : Array[_SCT]
        The array to verify.

    Returns
    -------
    verified_arr : Array2D[_SCT]
        The verified 2D array.

    """
    target_num_dim: int = 2
    if len(arr.shape) != target_num_dim:
        msg = f"The given array is not {target_num_dim}D! Got {len(arr.shape)}D instead..."
        raise TypeError(msg)
    return cast(Array2D[_SCT], arr)


def verify_array_is_3d(arr: Array[_SCT]) -> Array3D[_SCT]:
    """Verify that the given array is 3D.

    This is mostly used to cast the given array to a 3D array.

    Parameters
    ----------
    arr : Array[_SCT]
        The array to verify.

    Returns
    -------
    verified_arr : Array3D[_SCT]
        The verified 3D array.

    """
    target_num_dim: int = 3
    if len(arr.shape) != target_num_dim:
        msg = f"The given array is not {target_num_dim}D! Got {len(arr.shape)}D instead..."
        raise TypeError(msg)
    return cast(Array3D[_SCT], arr)


def verify_array_is_4d(arr: Array[_SCT]) -> Array4D[_SCT]:
    """Verify that the given array is 4D.

    This is mostly used to cast the given array to a 4D array.

    Parameters
    ----------
    arr : Array[_SCT]
        The array to verify.

    Returns
    -------
    verified_arr : Array1D[_SCT]
        The verified 1D array.

    """
    target_num_dim: int = 4
    if len(arr.shape) != target_num_dim:
        msg = f"The given array is not {target_num_dim}D! Got {len(arr.shape)}D instead..."
        raise TypeError(msg)
    return cast(Array4D[_SCT], arr)
