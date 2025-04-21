"""A collection of verification functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "InconsistentArrayLengthError",
    "UnexpectedArrayLengthError",
    "check_axis_length",
]


class InconsistentArrayLengthError(ValueError):
    """Validation error for when the given arrays differ in length on specific axes."""

    def __init__(self, common: int) -> None:
        msg = f"The arrays should have length {common} but some do not!"
        super().__init__(msg)


class UnexpectedArrayLengthError(ValueError):
    """Validation error for when the given arrays differ to the expected length."""

    def __init__(self, *, actual: int, expected: int) -> None:
        msg = f"The arrays should have length {expected} but some are instead {actual}"
        super().__init__(msg)


def check_axis_length(arrays: Sequence[tuple[int, tuple[int, ...]]], expected: int | None = None) -> int:
    """Check that all arrays have the same length along the specified axis.

    Parameters
    ----------
    arrays : Sequence[tuple[int, tuple[int, ...]]]
        A list of array shapes and a corresponding axis to check lengths of.
    expected : int | None
        The expected length to check against if given.

    Returns
    -------
    common_length : int
        The common length along the specified axis.

    Raises
    ------
    ValueError
        If the arrays have different lengths along the specified axis.

    Notes
    -----
    This function assumes that the given sequence of arrays and axes are not empty.

    """
    assert len(arrays) != 0
    lengths = [shape[axis] for axis, shape in arrays]
    if lengths.count(lengths[0]) != len(lengths):
        raise InconsistentArrayLengthError(lengths[0])

    if expected is not None and lengths[0] != expected:
        raise UnexpectedArrayLengthError(actual=lengths[0], expected=expected)

    return lengths[0]
