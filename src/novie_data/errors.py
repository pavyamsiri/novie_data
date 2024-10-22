"""Various novie data exceptions."""


class NegativeValueError(ValueError):
    """Validation error for when the given value is negative."""


class InconsistentArrayLengthError(ValueError):
    """Validation error for when the given arrays differ in length."""
