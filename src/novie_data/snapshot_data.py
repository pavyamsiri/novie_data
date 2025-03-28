"""Data general to all snapshot-based data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._slice_utils import Index1D, index_length_1d
from novie_data._type_utils import Array1D, verify_array_is_1d
from novie_data.errors import verify_arrays_have_same_shape

from .serde.accessors import (
    get_dataset_from_hdf5,
    get_dataset_metadata,
    get_file_version,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
)
from .serde.verification import verify_file_type_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path


_Array1D_f64: TypeAlias = Array1D[np.float64]
_Array1D_u16: TypeAlias = Array1D[np.uint16]
_Array1D_b8: TypeAlias = Array1D[np.bool_]


log: logging.Logger = logging.getLogger(__name__)


class _SnapshotDataConverter(Protocol):
    """Description of a `SnapshotData` file converter function."""

    def __call__(self, file: Hdf5File, *, is_complete: bool) -> None: ...


class _SnapshotDataVerifier(Protocol):
    """Description of a `SnapshotData` file verifier function."""

    def __call__(self, file: Hdf5File) -> int: ...


class _SnapshotDataLoader(Protocol):
    """Description of a `SnapshotData` file loader function."""

    def __call__(self, file: Hdf5File) -> SnapshotData: ...


LATEST_VERSION_V1: Version = Version("1.0.0")


class SnapshotData:
    """The data that is general to a collection of snapshots.

    Attributes
    ----------
    name : str
        The name of the dataset.
    codes : Array1D[u16]
        The snapshot codes.
    times : Array1D[f64]
        The time associated with each snapshot in Myr.
    completeness : Array1D[b8]
        A flag associated with each snapshot that shows whether data has been written to it yet.

    Notes
    -----
    v0 -> v1:
        - Added completeness {Array1D[b8]} field that tracks if a frame has been written to yet.
        - Change times dtype from f32 -> f64; better precision
        - Change codes dtype from u32 -> u16; unlikely that 2^16 - 1 limit will be reached.

    """

    DATA_FILE_TYPE: ClassVar[str] = "Snapshot"
    VERSION: ClassVar[Version] = LATEST_VERSION_V1

    def __init__(self, name: str, codes: _Array1D_u16, times: _Array1D_f64, *, complete: _Array1D_b8 | bool = False) -> None:
        """Initialize the snapshot data.

        Parameters
        ----------
        name : str
            The name of the dataset.
        codes : Array1D[u16]
            The snapshot codes.
        times : Array1D[f32]
            The time associated with each snapshot in Myr.
        complete : Array1D[b8] | bool
            Set this flag to signify that the given data is complete or give an array of boolean values.

        """
        num_times: int = len(times)
        completeness: _Array1D_b8
        match complete:
            case True:
                completeness = np.ones(num_times, dtype=np.bool_)
            case False:
                completeness = np.zeros(num_times, dtype=np.bool_)
            case _:
                completeness = complete

        verify_arrays_have_same_shape(
            [codes, times, completeness],
            msg="Expected `codes`, `times` and `completeness` to have the same length.",
        )

        self.name: str = name
        self.codes: _Array1D_u16 = codes
        self.times: _Array1D_f64 = times
        self.num_frames: int = num_times
        self.completeness: _Array1D_b8 = completeness

    @classmethod
    def empty(cls, num_frames: int) -> Self:
        """Create an empty dataset given the number of expected frames.

        Parameters
        ----------
        num_frames : int
            The number of frames.

        """
        codes: _Array1D_u16 = np.zeros(num_frames, dtype=np.uint16)
        times: _Array1D_f64 = np.zeros(num_frames, dtype=np.float64)
        return cls("UNKNOWN", codes, times, complete=False)

    def __eq__(self, other: object) -> bool:
        """Compare for equality.

        Parameters
        ----------
        other : object
            The object to compare to.

        Returns
        -------
        bool
            `True` if the other object is equal to this object, `False` otherwise.

        Notes
        -----
        Equality means all fields are equal.

        """
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.name == other.name
        equality &= np.array_equal(self.codes, other.codes)
        equality &= np.array_equal(self.times, other.times)
        equality &= np.array_equal(self.completeness, other.completeness)
        return bool(equality)

    def is_complete(self) -> bool:
        """Return if the dataset is complete.

        Returns
        -------
        is_complete : bool
            Whether the dataset is complete.

        """
        return bool(np.all(self.completeness))

    def snapshot_names(self) -> Iterator[str]:
        """Yield the snapshot names.

        Yields
        ------
        str
            The name of each snapshot.

        """
        for code in self.codes:
            yield f"{self.name}: {code}"

    @classmethod
    def save_init(cls, name: str, path: Path) -> None:
        """Serialize initial data to disk.

        Initial more so meaning data that is global to all frames.

        Parameters
        ----------
        name : str
            The name of the dataset.
        path : Path
            The path to the data.

        """
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)
        with Hdf5File(path, "a") as file:
            _ = cls.verify_data(file)

            cls.migrate_version(file, cls.VERSION, is_complete=False)
            # Set the global attr name
            file.attrs["name"] = name

        log.info("Successfully saved [cyan]%s[/cyan] initial to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_frame(cls, frame: int, code: int, time: float, path: Path) -> None:
        """Serialize frame data to disk.

        Parameters
        ----------
        frame : int
            The frame index.
        code : int
            The snapshot code.
        time : float
            The simulation time in Myr.
        path : Path
            The path to the data.


        """
        with Hdf5File(path, "a") as file:
            _ = cls.verify_data(file)
            cls.migrate_version(file, cls.VERSION, is_complete=False)

            get_dataset_from_hdf5(file, "codes").write_direct(
                np.asarray(code, dtype=np.uint16).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "times").write_direct(
                np.asarray(time, dtype=np.float64).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(a=True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
        log.info("Successfully saved [cyan]%s[/cyan] frame to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_chunk(cls, chunk_slice: Index1D, codes: _Array1D_u16, times: _Array1D_f64, path: Path) -> None:
        """Serialize frame data to disk.

        Parameters
        ----------
        chunk_slice : Index1D
            The frames to write to.
        codes : Array1D[u16]
            The snapshot codes.
        times : Array1D[f64]
            The simulation times in Myr.
        path : Path
            The path to the data.

        """
        # Assert that len(codes) == len(times)
        if len(codes) != len(times):
            msg = "`codes` and `times` are not the same length!"
            raise ValueError(msg)

        # Assert that chunk slice represents a chunk of length equal to len(codes) or len(times)
        num_frames: int = len(codes)
        num_slice_frames: int = index_length_1d(chunk_slice)
        if num_frames != num_slice_frames:
            msg = f"The given number of frames by the arrays ({num_frames}) differs from the slice length ({num_slice_frames})."
            raise ValueError(msg)

        completeness = np.ones_like(codes, dtype=np.bool_)
        with Hdf5File(path, "a") as file:
            _ = cls.verify_data(file)
            cls.migrate_version(file, cls.VERSION, is_complete=False)

            get_dataset_from_hdf5(file, "codes").write_direct(codes, np.s_[:], chunk_slice)
            get_dataset_from_hdf5(file, "times").write_direct(times, np.s_[:], chunk_slice)
            get_dataset_from_hdf5(file, "completeness").write_direct(completeness, np.s_[:], chunk_slice)
        log.info("Successfully saved [cyan]%s[/cyan] frame to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    def dump(self, path: Path) -> None:
        """Serialize data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        cls = type(self)
        with Hdf5File(path, "a") as file:
            # General
            file.attrs["type"] = cls.DATA_FILE_TYPE
            file.attrs["version"] = str(cls.VERSION)
            file.attrs["name"] = self.name

            file.create_dataset("codes", data=self.codes)
            file.create_dataset("times", data=self.times)
            file.create_dataset("completeness", data=self.completeness)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @staticmethod
    def load(path: Path) -> SnapshotData:
        """Deserialize data from disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        cls = SnapshotData
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)

            file_version = get_file_version(file)
            assert file_version.major in _LOADERS
            data = _LOADERS[file_version.major](file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return data

    @classmethod
    def verify_data(cls, file: Hdf5File) -> tuple[Version, int]:
        """Verify the data is valid.

        Parameters
        ----------
        file : Hdf5File
            The open HDF5 file.

        Returns
        -------
        version : Version
            The file version.
        num_frames : int
            The total number of frames.

        """
        file_version = get_file_version(file)
        assert file_version.major in _VERIFIERS
        num_frames: int = _VERIFIERS[file_version.major](file)
        return (file_version, num_frames)

    @classmethod
    def migrate_version(cls, file: Hdf5File, target_version: Version, *, is_complete: bool) -> None:
        """Upgrade the HDF5 file to the target version incrementally.

        Parameters
        ----------
        file : Hdf5File
            The open file.
        target_version : Version
            The target version.
        is_complete : bool
            Set this flag to make the completeness array all set to true.

        """
        current_version: Version = get_file_version(file)

        assert current_version.major <= target_version.major, "The target version is lower than the file's version!"

        while current_version.major < target_version.major:
            next_version = current_version.major + 1
            converter_key = (next_version, current_version.major)
            if converter_key not in _CONVERTERS:
                msg = f"No converter found for v{current_version.major} -> v{next_version}"
                raise ValueError(msg)

            _CONVERTERS[converter_key](file, is_complete=is_complete)
            current_version = Version(str(file.attrs["version"]))


def convert_v0_to_v1(file: Hdf5File, *, is_complete: bool) -> None:
    """Convert a v0 file to a v1 file.

    Parameters
    ----------
    file : Hdf5File
        The v0 file.
    is_complete : bool
        Set this flag to make the completeness array all set to true.

    """
    log.debug("Converting %s v0 to v1...", SnapshotData.__name__)
    version, num_frames = SnapshotData.verify_data(file)
    assert version.major == 0

    # Check that we are missing the completeness field
    assert "completeness" not in file

    # Add completeness field
    if is_complete:
        file.create_dataset("completeness", data=np.ones(num_frames, dtype=np.bool_))
    else:
        file.create_dataset("completeness", data=np.zeros(num_frames, dtype=np.bool_))

    # Update version
    file.attrs["version"] = str(LATEST_VERSION_V1)


_CONVERTERS: Mapping[tuple[int, int], _SnapshotDataConverter] = {(1, 0): convert_v0_to_v1}


def verify_v0(file: Hdf5File) -> int:
    """Verify v0 data.

    Parameters
    ----------
    file : Hdf5File
        The open file.

    Returns
    -------
    num_frames : int
        The number of frames.

    """
    file_version: Version = get_file_version(file)
    assert file_version.major == 0

    codes_shape, codes_dtype = get_dataset_metadata(file, "codes")
    times_shape, times_dtype = get_dataset_metadata(file, "times")
    assert codes_dtype == np.uint32
    assert times_dtype == np.float32
    shapes: list[tuple[int, ...]] = [codes_shape, times_shape]

    # Assert 1D
    dimensions = [len(shape) for shape in shapes]
    assert dimensions.count(1) == len(dimensions)

    # Assert consistency
    assert dimensions.count(dimensions[0]) == len(dimensions)
    return dimensions[0]


def verify_v1(file: Hdf5File) -> int:
    """Verify v1 data.

    Parameters
    ----------
    file : Hdf5File
        The open file.

    Returns
    -------
    num_frames : int
        The number of frames.

    """
    file_version: Version = get_file_version(file)
    assert file_version.major == 1

    codes_shape, codes_dtype = get_dataset_metadata(file, "codes")
    times_shape, times_dtype = get_dataset_metadata(file, "times")
    completeness_shape, completeness_dtype = get_dataset_metadata(file, "completeness")
    assert codes_dtype == np.uint16
    assert times_dtype == np.float64
    assert completeness_dtype == np.bool_
    shapes: list[tuple[int, ...]] = [codes_shape, times_shape, completeness_shape]

    # Assert 1D
    dimensions = [len(shape) for shape in shapes]
    assert dimensions.count(1) == len(dimensions)

    # Assert consistency
    assert dimensions.count(dimensions[0]) == len(dimensions)

    return dimensions[0]


_VERIFIERS: Mapping[int, _SnapshotDataVerifier] = {0: verify_v0, 1: verify_v1}


def load_v0(file: Hdf5File) -> SnapshotData:
    """Deserialize v0 data from disk.

    Parameters
    ----------
    file : Hdf5File
        The open file.

    """
    cls = SnapshotData

    file_version: Version = get_file_version(file)
    assert file_version.major == 0

    name: str = get_str_attr_from_hdf5(file, "name")
    codes = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "codes", dtype=np.uint16))
    times = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "times", dtype=np.float64))

    return cls(
        name=name,
        codes=codes,
        times=times,
        complete=True,
    )


def load_v1(file: Hdf5File) -> SnapshotData:
    """Deserialize v1 data from disk.

    Parameters
    ----------
    file : Hdf5File
        The open file.

    """
    cls = SnapshotData
    verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)

    # v0
    file_version: Version = get_file_version(file)
    assert file_version.major == 1

    name: str = get_str_attr_from_hdf5(file, "name")
    codes = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "codes", dtype=np.uint16))
    times = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "times", dtype=np.float64))
    complete = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))

    return cls(
        name=name,
        codes=codes,
        times=times,
        complete=complete,
    )


_LOADERS: Mapping[int, _SnapshotDataLoader] = {0: load_v0, 1: load_v1}
