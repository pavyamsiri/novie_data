"""Useful types."""

from typing import TypeVar

import numpy as np

_ST = TypeVar("_ST", bound=np.generic)

AnyArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
AnyArrayWithDType = np.ndarray[tuple[int, ...], np.dtype[_ST]]


def require_dtype(arr: AnyArray, dtype: type[_ST]) -> AnyArrayWithDType[_ST]:
    """Convert the given array to the given dtype.

    This is mostly used to help with static type checking.

    Parameters
    ----------
    arr : NDArray
        The array to convert.
    dtype : T
        The data type to convert to.

    Returns
    -------
    converted_arr : NDArray[dtype[T]]
        The converted array.

    """
    return arr.astype(dtype)
