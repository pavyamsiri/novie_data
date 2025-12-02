from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, override

import numpy as np
from h5py import File as Hdf5File
from novie_helpers import (
    check_axis_length,
    get_dataset_from_hdf5,
    get_file_version,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
)
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_helpers import Array1D


LATEST_VERSION_V0 = Version("0.0.0")
LATEST_VERSION_V1 = Version("1.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _SnapshotDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> SnapshotData: ...


class SnapshotData:
    DATA_FILE_TYPE: ClassVar[str] = "Snapshot"
    VERSION: ClassVar[Version] = LATEST_VERSION_V1


    def __init__(
        self,
        *,
        codes: Array1D[np.uint16],
        times: Array1D[np.float64],
        completeness: Array1D[np.bool_] | bool = True,
        name: str = "UNKNOWN",
    ) -> None:
        n = check_axis_length(
            ((0, codes.shape), (0, times.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((n,), dtype=np.bool_)
            case False:
                completeness = np.zeros((n,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
        self.num_frames: int = n
        self.name: str = name
        self.codes: Array1D[np.uint16] = codes
        self.completeness: Array1D[np.bool_] = completeness
        self.times: Array1D[np.float64] = times

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.name == other.name
        equality &= np.array_equal(self.codes, other.codes, equal_nan=True)
        equality &= np.array_equal(self.completeness, other.completeness, equal_nan=True)
        equality &= np.array_equal(self.times, other.times, equal_nan=True)
        return bool(equality)

    @classmethod
    def empty(cls, *, n: int) -> Self:
        codes: Array1D[np.uint16] = np.zeros((n,), dtype=np.uint16)
        completeness: Array1D[np.bool_] = np.zeros((n,), dtype=np.bool_)
        times: Array1D[np.float64] = np.zeros((n,), dtype=np.float64)
        return cls(
            codes=codes,
            completeness=completeness,
            times=times,
        )

    @staticmethod
    def load(path: Path) -> SnapshotData:
        path = path.expanduser()
        cls = SnapshotData
        with Hdf5File(path, "r") as file:
            assert str(file.attrs["type"]) == cls.DATA_FILE_TYPE

            file_version = get_file_version(file)
            assert file_version.major in _LOADERS
            data = _LOADERS[file_version.major](file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path)
        return data

    @classmethod
    def migrate(cls, path: Path) -> None:
        path = path.expanduser()
        with Hdf5File(path, "r") as file:
            assert str(file.attrs["type"]) == cls.DATA_FILE_TYPE

            file_version = get_file_version(file)
            if file_version == LATEST_VERSION_V1:
                return
            file_version = get_file_version(file)
            assert file_version.major in _LOADERS
            data = _LOADERS[file_version.major](file)

        data.dump(path)

    def dump(self, path: Path) -> None:
        path = path.expanduser()
        cls = type(self)
        with Hdf5File(path, "w") as file:
            file.attrs.create("type", str(cls.DATA_FILE_TYPE))
            file.attrs.create("version", str(cls.VERSION))
            file.attrs.create("name", str(self.name))
            _ = file.create_dataset("codes", data=self.codes)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("times", data=self.times)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def is_compatible(cls, path: Path, *, name: str) -> bool:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't check compatibility of {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        is_compatible: bool = True
        with Hdf5File(path, "r") as file:
            is_compatible &= name == get_str_attr_from_hdf5(file, "name")
        return is_compatible

    @classmethod
    def save_init(cls, path: Path, *, name: str) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("name", name)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        codes: int,
        times: float,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "codes").write_direct(
                np.asarray(codes, dtype=np.uint16).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "times").write_direct(
                np.asarray(times, dtype=np.float64).reshape(1), np.s_[0], np.s_[frame]
            )
        log.info("Successfully saved frame %d of [cyan]%s[/cyan] to [magenta]%s[/magenta]", frame, cls.__name__, path)


def load_v0(file: Hdf5File) -> SnapshotData:
    cls = SnapshotData
    name = get_str_attr_from_hdf5(file, "name")
    codes = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "codes", dtype=np.uint32))
    times = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "times", dtype=np.float32))

    return cls(
        name=name,
        codes=codes.astype(np.uint16),
        times=times.astype(np.float64),
    )


def load_v1(file: Hdf5File) -> SnapshotData:
    cls = SnapshotData
    name = get_str_attr_from_hdf5(file, "name")
    codes = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "codes", dtype=np.uint16))
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    times = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "times", dtype=np.float64))

    return cls(
        name=name,
        codes=codes,
        completeness=completeness,
        times=times,
    )


_LOADERS: Mapping[int, _SnapshotDataLoader] = {
    0: load_v0,
    1: load_v1,
}


