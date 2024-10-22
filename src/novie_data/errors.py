"""Various novie data exceptions."""

import optype as op


class NegativeValueError(ValueError):
    """Validation error for when the given value is negative."""


class InconsistentArrayLengthError(ValueError):
    """Validation error for when the given arrays differ in length."""


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
