"""Module containing utilties relating to slices."""

from __future__ import annotations

import math

type Index1D = int | slice[int, int, int | None]


def index_length_1d(index: Index1D) -> int:
    """Calculate the length of a 1D index expression.

    Parameters
    ----------
    index : Index1D
        The 1D index expression.

    Returns
    -------
    slice_length : int
        The length of the index expression.

    """
    if isinstance(index, int):
        return 1

    start: int = index.start
    stop: int = index.stop
    step: int
    # Step is given
    if index.step is not None:
        step = index.step
    # Increasing
    elif stop >= start:
        step = 1
    # Decreasing
    else:
        step = -1
    return max(0, int(math.ceil((stop - start) / step)))
