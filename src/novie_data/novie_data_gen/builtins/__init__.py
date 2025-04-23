"""Module of builtin functions needed by generated data classes."""

from ._array import (
    AnyArray,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    GenericArray,
    require_dtype,
    verify_array_is_1d,
    verify_array_is_2d,
    verify_array_is_3d,
    verify_array_is_4d,
)
from ._getter import (
    get_dataset_from_hdf5,
    get_dataset_metadata,
    get_file_version,
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    get_string_sequence_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from ._verifier import InconsistentArrayLengthError, UnexpectedArrayLengthError, check_axis_length

__all__ = [
    "AnyArray",
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "GenericArray",
    "InconsistentArrayLengthError",
    "UnexpectedArrayLengthError",
    "check_axis_length",
    "get_dataset_from_hdf5",
    "get_dataset_metadata",
    "get_file_version",
    "get_float_attr_from_hdf5",
    "get_int_attr_from_hdf5",
    "get_str_attr_from_hdf5",
    "get_string_sequence_from_hdf5",
    "read_dataset_from_hdf5_with_dtype",
    "require_dtype",
    "verify_array_is_1d",
    "verify_array_is_2d",
    "verify_array_is_3d",
    "verify_array_is_4d",
]
