"""Various novie data exceptions."""

from collections.abc import Sequence

import optype as op

from novie_data._type_utils import AnyArray


class NegativeValueError(ValueError):
    """Validation error for when the given value is negative."""


class NonPositiveValueError(ValueError):
    """Validation error for when the given value is non-positive."""


class InconsistentArrayLengthError(ValueError):
    """Validation error for when the given arrays differ in length on specific axes."""


class WrongArrayLengthError(ValueError):
    """Validation error for when the given arrays have the wrong length on specific axes."""


class InconsistentArrayShapeError(InconsistentArrayLengthError):
    """Validation error for when the given arrays differ in shape."""


def verify_value_is_nonnegative(value: op.CanLt[int, bool], *, msg: str) -> None:
    """Verify that the given value is non-negative.

    Parameters
    ----------
    value : number
        The value to check.
    msg : str
        The message to display on failure.

    """
    if value < 0:
        raise NegativeValueError(msg)


def verify_value_is_positive(value: op.CanLe[int, bool], *, msg: str) -> None:
    """Verify that the given value is positive.

    Parameters
    ----------
    value : number
        The value to check.
    msg : str
        The message to display on failure.

    """
    if value <= 0:
        raise NonPositiveValueError(msg)


def verify_arrays_have_same_shape(arrays: Sequence[AnyArray], *, msg: str) -> None:
    """Verify that the given value is non-negative.

    Parameters
    ----------
    arrays : Sequence[AnyArray]
        A sequence of arrays to check that they have the same shape.
    msg : str
        The message to display on failure.

    """
    shapes = [array.shape for array in arrays]
    if len(shapes) == 0:
        return
    if shapes.count(shapes[0]) != len(shapes):
        raise InconsistentArrayShapeError(msg)


def verify_arrays_are_consistent(array_axes: Sequence[tuple[AnyArray, int]], *, msg: str) -> None:
    """Verify that the given value is non-negative.

    Parameters
    ----------
    array_axes : Sequence[tuple[AnyArray, int]]
        A sequence of tuples containing the arrays to check for consistency and the axis in which to check.
    msg : str
        The message to display on failure.

    """
    lengths = [array.shape[axis] for array, axis in array_axes]
    if len(lengths) == 0:
        return
    if lengths.count(lengths[0]) != len(lengths):
        raise InconsistentArrayLengthError(msg)


def verify_arrays_have_correct_length(array_axes: Sequence[tuple[AnyArray, int]], expected_length: int, *, msg: str) -> None:
    """Verify that the given value is non-negative.

    Parameters
    ----------
    array_axes : Sequence[tuple[AnyArray, int]]
        A sequence of tuples containing the arrays to check for consistency and the axis in which to check.
    expected_length : int
        The expected length.
    msg : str
        The message to display on failure.

    """
    if not all(array.shape[axis] == expected_length for array, axis in array_axes):
        raise WrongArrayLengthError(msg)
