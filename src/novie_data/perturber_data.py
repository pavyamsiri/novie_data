from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version
from typing_extensions import override

from novie_data_gen import (
    check_axis_length,
    get_dataset_from_hdf5,
    get_file_version,
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_2d,
    verify_array_is_3d,
    verify_array_is_4d,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_data_gen import Array1D, Array2D, Array3D, Array4D

    _Array1D_f32: TypeAlias = Array1D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array1D_f64: TypeAlias = Array1D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V2 = Version("2.0.0")
LATEST_VERSION_V3 = Version("3.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _PerturberDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> PerturberData: ...


class PerturberData:
    DATA_FILE_TYPE: ClassVar[str] = "Perturber"
    VERSION: ClassVar[Version] = LATEST_VERSION_V3


    def __init__(
        self,
        *,
        mass: _Array1D_f64,
        position: _Array2D_f64,
        velocity: _Array2D_f64,
        completeness: _Array1D_b8 | bool = True,
        name: str = "UNKNOWN",
    ) -> None:
        _ = check_axis_length(
            ((0, position.shape), (0, velocity.shape)), expected=3
        )
        n = check_axis_length(
            ((0, mass.shape), (1, position.shape), (1, velocity.shape))
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
        self.completeness: _Array1D_b8 = completeness
        self.mass: _Array1D_f64 = mass
        self.position: _Array2D_f64 = position
        self.velocity: _Array2D_f64 = velocity

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.name == other.name
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.mass, other.mass)
        equality &= np.array_equal(self.position, other.position)
        equality &= np.array_equal(self.velocity, other.velocity)
        return bool(equality)

    @classmethod
    def empty(cls, *, n: int) -> Self:
        completeness: _Array1D_b8 = np.zeros((n,), dtype=np.bool_)
        mass: _Array1D_f64 = np.zeros((n,), dtype=np.float64)
        position: _Array2D_f64 = np.zeros((3, n), dtype=np.float64)
        velocity: _Array2D_f64 = np.zeros((3, n), dtype=np.float64)
        return cls(
            completeness=completeness,
            mass=mass,
            position=position,
            velocity=velocity,
        )

    @staticmethod
    def load(path: Path) -> PerturberData:
        path = path.expanduser()
        cls = PerturberData
        with Hdf5File(path, "r") as file:
            assert str(file.attrs["type"]) == cls.DATA_FILE_TYPE

            file_version = get_file_version(file)
            assert file_version.major in _LOADERS
            data = _LOADERS[file_version.major](file)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)
        return data

    @classmethod
    def migrate(cls, path: Path) -> None:
        path = path.expanduser()
        with Hdf5File(path, "r") as file:
            assert str(file.attrs["type"]) == cls.DATA_FILE_TYPE

            file_version = get_file_version(file)
            if file_version == LATEST_VERSION_V3:
                return
            file_version = get_file_version(file)
            assert file_version.major in _LOADERS
            data = _LOADERS[file_version.major](file)

        data.dump(path)

    def dump(self, path: Path) -> None:
        path = path.expanduser()
        cls = type(self)
        with Hdf5File(path, "w") as file:
            file.attrs["type"] = cls.DATA_FILE_TYPE
            file.attrs["version"] = str(cls.VERSION)
            file.attrs["name"] = self.name
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("mass", data=self.mass)
            _ = file.create_dataset("position", data=self.position)
            _ = file.create_dataset("velocity", data=self.velocity)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(cls, path: Path, *, name: str) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs["name"] = name
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        mass: float,
        position: _Array1D_f64,
        velocity: _Array1D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        _ = check_axis_length(
            ((0, position.shape), (0, velocity.shape)), expected=3
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "mass").write_direct(
                np.asarray(mass, dtype=np.float64).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "position").write_direct(
                np.asarray(position, dtype=np.float64).reshape((3,)), np.s_[:], np.s_[:, frame]
            )
            get_dataset_from_hdf5(file, "velocity").write_direct(
                np.asarray(velocity, dtype=np.float64).reshape((3,)), np.s_[:], np.s_[:, frame]
            )
        log.info("Successfully saved frame {frame} of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)


def load_v2(file: Hdf5File) -> PerturberData:
    cls = PerturberData
    name = get_str_attr_from_hdf5(file, "name")
    mass = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "mass", dtype=np.float32))
    position = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "position", dtype=np.float32))
    velocity = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "velocity", dtype=np.float32))
    n = check_axis_length(
        ((0, mass.shape), (1, position.shape), (1, velocity.shape),)
    )

    return cls(
        name=name,
        mass=mass.astype(np.float64),
        position=position.astype(np.float64),
        velocity=velocity.astype(np.float64),
    )


def load_v3(file: Hdf5File) -> PerturberData:
    cls = PerturberData
    name = get_str_attr_from_hdf5(file, "name")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    mass = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "mass", dtype=np.float64))
    position = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "position", dtype=np.float64))
    velocity = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "velocity", dtype=np.float64))
    n = check_axis_length(
        ((0, mass.shape), (1, position.shape), (1, velocity.shape),)
    )

    return cls(
        name=name,
        completeness=completeness,
        mass=mass,
        position=position,
        velocity=velocity,
    )


_LOADERS: Mapping[int, _PerturberDataLoader] = {
    2: load_v2,
    3: load_v3,
}


