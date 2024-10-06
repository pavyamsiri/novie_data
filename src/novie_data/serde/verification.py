"""Deserialization verification functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from packaging.version import Version

if TYPE_CHECKING:
    from h5py import File as Hdf5File


log: logging.Logger = logging.getLogger(__name__)


def verify_file_type(test_type: str, expected_type: str) -> None:
    """Verify that the given file type is the same as the expected.

    Parameters
    ----------
    test_type : str
        The type to test.
    expected_type : str
        The expected type.

    """
    if test_type == expected_type:
        return
    msg = f"Expected hdf5 file of type {expected_type} but got {test_type}"
    raise ValueError(msg)


def verify_version(test_version: Version, host_version: Version) -> None:
    """Verify that the given version is compatible with the host version.

    Parameters
    ----------
    test_version : Version
        The version to test.
    host_version : Version
        The version of the deserializer.

    """
    if test_version.major != host_version.major:
        msg = "Expected the file to have the same major version!"
        raise ValueError(msg)
    if test_version.minor > host_version.minor:
        log.warning("The file's minor version is higher than the deserializer. Might be missing some features.")


def verify_file_type_from_hdf5(in_file: Hdf5File, expected_type: str) -> None:
    """Verify that the file type of the HDF5 file is the same as the expected.

    Parameters
    ----------
    in_file : Hdf5File
        The file to read from.
    expected_type : str
        The expected type.

    """
    input_type: str = cast(str, in_file.attrs["type"])
    verify_file_type(input_type, expected_type)


def verify_file_version_from_hdf5(in_file: Hdf5File, expected_version: Version) -> None:
    """Verify that the file version of the HDF5 file is the compatible with the expected.

    Parameters
    ----------
    in_file : Hdf5File
        The file to read from.
    expected_version : Version
        The expected version.

    """
    input_version: Version = Version(cast(str, in_file.attrs["version"]))
    verify_version(input_version, expected_version)
