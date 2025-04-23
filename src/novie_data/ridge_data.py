from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version
from typing_extensions import override

from novie_data.novie_data_gen import (
    check_axis_length,
    get_dataset_from_hdf5,
    get_file_version,
    get_float_attr_from_hdf5,
    get_int_attr_from_hdf5,
    get_str_attr_from_hdf5,
    get_string_sequence_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_2d,
    verify_array_is_3d,
    verify_array_is_4d,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from novie_data.novie_data_gen import Array1D, Array2D, Array3D, Array4D

    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V2 = Version("2.0.0")
LATEST_VERSION_V3 = Version("3.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _RidgeDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> RidgeData: ...


class RidgeData:
    DATA_FILE_TYPE: ClassVar[str] = "Ridge"
    VERSION: ClassVar[Version] = LATEST_VERSION_V3


    def __init__(
        self,
        *,
        mass_density: _Array3D_f64,
        number_density: _Array3D_f64,
        completeness: _Array1D_b8 | bool = True,
        max_radius: float = 12,
        max_velocity: float = 255,
        min_radius: float = 0,
        min_velocity: float = -255,
        name: str = "UNKNOWN",
    ) -> None:
        num_frames = check_axis_length(
            ((2, mass_density.shape), (2, number_density.shape))
        )
        num_radial_bins = check_axis_length(
            ((1, mass_density.shape), (1, number_density.shape))
        )
        num_velocity_bins = check_axis_length(
            ((0, mass_density.shape), (0, number_density.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((num_frames,), dtype=np.bool_)
            case False:
                completeness = np.zeros((num_frames,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
        self.num_frames: int = num_frames
        self.num_radial_bins: int = num_radial_bins
        self.num_velocity_bins: int = num_velocity_bins
        self.max_radius: float = max_radius
        self.max_velocity: float = max_velocity
        self.min_radius: float = min_radius
        self.min_velocity: float = min_velocity
        self.name: str = name
        self.completeness: _Array1D_b8 = completeness
        self.mass_density: _Array3D_f64 = mass_density
        self.number_density: _Array3D_f64 = number_density

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.num_radial_bins == other.num_radial_bins
        equality &= self.num_velocity_bins == other.num_velocity_bins
        equality &= self.max_radius == other.max_radius
        equality &= self.max_velocity == other.max_velocity
        equality &= self.min_radius == other.min_radius
        equality &= self.min_velocity == other.min_velocity
        equality &= self.name == other.name
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.mass_density, other.mass_density)
        equality &= np.array_equal(self.number_density, other.number_density)
        return bool(equality)

    @classmethod
    def empty(
        cls,
        *,
        num_frames: int,
        num_radial_bins: int,
        num_velocity_bins: int,
    ) -> Self:
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        mass_density: _Array3D_f64 = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float64)
        number_density: _Array3D_f64 = np.zeros((num_velocity_bins, num_radial_bins, num_frames), dtype=np.float64)
        return cls(
            completeness=completeness,
            mass_density=mass_density,
            number_density=number_density,
        )

    @staticmethod
    def load(path: Path) -> RidgeData:
        path = path.expanduser()
        cls = RidgeData
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
            file.attrs.create("type", str(cls.DATA_FILE_TYPE))
            file.attrs.create("version", str(cls.VERSION))
            file.attrs.create("max_radius", self.max_radius, dtype=np.float64)
            file.attrs.create("max_velocity", self.max_velocity, dtype=np.float64)
            file.attrs.create("min_radius", self.min_radius, dtype=np.float64)
            file.attrs.create("min_velocity", self.min_velocity, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("mass_density", data=self.mass_density)
            _ = file.create_dataset("number_density", data=self.number_density)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        max_radius: float,
        max_velocity: float,
        min_radius: float,
        min_velocity: float,
        name: str,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("max_radius", max_radius)
            file.attrs.modify("max_velocity", max_velocity)
            file.attrs.modify("min_radius", min_radius)
            file.attrs.modify("min_velocity", min_velocity)
            file.attrs.modify("name", name)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        mass_density: _Array2D_f64,
        number_density: _Array2D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_radial_bins = check_axis_length(
            ((1, mass_density.shape), (1, number_density.shape))
        )
        num_velocity_bins = check_axis_length(
            ((0, mass_density.shape), (0, number_density.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "mass_density").write_direct(
                np.asarray(mass_density, dtype=np.float64).reshape((num_velocity_bins, num_radial_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "number_density").write_direct(
                np.asarray(number_density, dtype=np.float64).reshape((num_velocity_bins, num_radial_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
        log.info("Successfully saved frame {frame} of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)


def load_v2(file: Hdf5File) -> RidgeData:
    cls = RidgeData
    max_radius = get_float_attr_from_hdf5(file, "max_radius")
    max_velocity = get_float_attr_from_hdf5(file, "max_velocity")
    min_radius = get_float_attr_from_hdf5(file, "min_radius")
    min_velocity = get_float_attr_from_hdf5(file, "min_velocity")
    name = get_str_attr_from_hdf5(file, "name")
    mass_density = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mass_density", dtype=np.float32))
    number_density = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "number_density", dtype=np.float32))

    return cls(
        max_radius=max_radius,
        max_velocity=max_velocity,
        min_radius=min_radius,
        min_velocity=min_velocity,
        name=name,
        mass_density=mass_density.astype(np.float64),
        number_density=number_density.astype(np.float64),
    )


def load_v3(file: Hdf5File) -> RidgeData:
    cls = RidgeData
    max_radius = get_float_attr_from_hdf5(file, "max_radius")
    max_velocity = get_float_attr_from_hdf5(file, "max_velocity")
    min_radius = get_float_attr_from_hdf5(file, "min_radius")
    min_velocity = get_float_attr_from_hdf5(file, "min_velocity")
    name = get_str_attr_from_hdf5(file, "name")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    mass_density = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mass_density", dtype=np.float64))
    number_density = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "number_density", dtype=np.float64))

    return cls(
        max_radius=max_radius,
        max_velocity=max_velocity,
        min_radius=min_radius,
        min_velocity=min_velocity,
        name=name,
        completeness=completeness,
        mass_density=mass_density,
        number_density=number_density,
    )


_LOADERS: Mapping[int, _RidgeDataLoader] = {
    2: load_v2,
    3: load_v3,
}


