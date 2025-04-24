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
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_2d,
    verify_array_is_3d,
    verify_array_is_4d,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_data.novie_data_gen import Array1D, Array2D, Array3D, Array4D

    _Array1D_f32: TypeAlias = Array1D[np.float32]
    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array1D_f64: TypeAlias = Array1D[np.float64]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V3 = Version("3.0.0")
LATEST_VERSION_V4 = Version("4.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _WrinkleDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> WrinkleData: ...


class WrinkleData:
    DATA_FILE_TYPE: ClassVar[str] = "Wrinkle"
    VERSION: ClassVar[Version] = LATEST_VERSION_V4

    def __init__(
        self,
        *,
        angular_momentum: _Array1D_f64,
        mean_radial_velocity: _Array3D_f64,
        mean_radial_velocity_error: _Array3D_f64,
        completeness: _Array1D_b8 | bool = True,
        distance_error: float = 0,
        max_lz: float = 255,
        min_lz: float = 0,
        name: str = "UNKNOWN",
        sphere_radius: float = 2,
    ) -> None:
        num_frames = check_axis_length(((1, mean_radial_velocity.shape), (1, mean_radial_velocity_error.shape)))
        num_locations = check_axis_length(((2, mean_radial_velocity.shape), (2, mean_radial_velocity_error.shape)))
        num_momentum_bins = check_axis_length(
            ((0, angular_momentum.shape), (0, mean_radial_velocity.shape), (0, mean_radial_velocity_error.shape))
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
        self.num_locations: int = num_locations
        self.num_momentum_bins: int = num_momentum_bins
        self.distance_error: float = distance_error
        self.max_lz: float = max_lz
        self.min_lz: float = min_lz
        self.name: str = name
        self.sphere_radius: float = sphere_radius
        self.angular_momentum: _Array1D_f64 = angular_momentum
        self.completeness: _Array1D_b8 = completeness
        self.mean_radial_velocity: _Array3D_f64 = mean_radial_velocity
        self.mean_radial_velocity_error: _Array3D_f64 = mean_radial_velocity_error

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.num_locations == other.num_locations
        equality &= self.num_momentum_bins == other.num_momentum_bins
        equality &= self.distance_error == other.distance_error
        equality &= self.max_lz == other.max_lz
        equality &= self.min_lz == other.min_lz
        equality &= self.name == other.name
        equality &= self.sphere_radius == other.sphere_radius
        equality &= np.array_equal(self.angular_momentum, other.angular_momentum)
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.mean_radial_velocity, other.mean_radial_velocity)
        equality &= np.array_equal(self.mean_radial_velocity_error, other.mean_radial_velocity_error)
        return bool(equality)

    @classmethod
    def empty(
        cls,
        *,
        num_frames: int,
        num_locations: int,
        num_momentum_bins: int,
    ) -> Self:
        angular_momentum: _Array1D_f64 = np.zeros((num_momentum_bins,), dtype=np.float64)
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        mean_radial_velocity: _Array3D_f64 = np.zeros((num_momentum_bins, num_frames, num_locations), dtype=np.float64)
        mean_radial_velocity_error: _Array3D_f64 = np.zeros((num_momentum_bins, num_frames, num_locations), dtype=np.float64)
        return cls(
            angular_momentum=angular_momentum,
            completeness=completeness,
            mean_radial_velocity=mean_radial_velocity,
            mean_radial_velocity_error=mean_radial_velocity_error,
        )

    @staticmethod
    def load(path: Path) -> WrinkleData:
        path = path.expanduser()
        cls = WrinkleData
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
            if file_version == LATEST_VERSION_V4:
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
            file.attrs.create("distance_error", self.distance_error, dtype=np.float64)
            file.attrs.create("max_lz", self.max_lz, dtype=np.float64)
            file.attrs.create("min_lz", self.min_lz, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            file.attrs.create("sphere_radius", self.sphere_radius, dtype=np.float64)
            _ = file.create_dataset("angular_momentum", data=self.angular_momentum)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("mean_radial_velocity", data=self.mean_radial_velocity)
            _ = file.create_dataset("mean_radial_velocity_error", data=self.mean_radial_velocity_error)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        angular_momentum: _Array1D_f64,
        distance_error: float,
        max_lz: float,
        min_lz: float,
        name: str,
        sphere_radius: float,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("distance_error", distance_error)
            file.attrs.modify("max_lz", max_lz)
            file.attrs.modify("min_lz", min_lz)
            file.attrs.modify("name", name)
            file.attrs.modify("sphere_radius", sphere_radius)
            get_dataset_from_hdf5(file, "angular_momentum").write_direct(
                np.asarray(angular_momentum, dtype=np.float64).reshape(angular_momentum.shape), np.s_[:], np.s_[:]
            )
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        mean_radial_velocity: _Array2D_f64,
        mean_radial_velocity_error: _Array2D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_locations = check_axis_length(((1, mean_radial_velocity.shape), (1, mean_radial_velocity_error.shape)))
        num_momentum_bins = check_axis_length(((0, mean_radial_velocity.shape), (0, mean_radial_velocity_error.shape)))
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "mean_radial_velocity").write_direct(
                np.asarray(mean_radial_velocity, dtype=np.float64).reshape((num_momentum_bins, num_locations)),
                np.s_[:, :],
                np.s_[:, frame, :],
            )
            get_dataset_from_hdf5(file, "mean_radial_velocity_error").write_direct(
                np.asarray(mean_radial_velocity_error, dtype=np.float64).reshape((num_momentum_bins, num_locations)),
                np.s_[:, :],
                np.s_[:, frame, :],
            )
        log.info("Successfully saved frame %d of [cyan]%s[/cyan] to [magenta]%s[/magenta]", frame, cls.__name__, path)


def load_v3(file: Hdf5File) -> WrinkleData:
    cls = WrinkleData
    distance_error = get_float_attr_from_hdf5(file, "distance_error")
    max_lz = get_float_attr_from_hdf5(file, "max_lz")
    min_lz = get_float_attr_from_hdf5(file, "min_lz")
    name = get_str_attr_from_hdf5(file, "name")
    sphere_radius = get_float_attr_from_hdf5(file, "sphere_radius")
    angular_momentum = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "angular_momentum", dtype=np.float32))
    mean_radial_velocity = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity", dtype=np.float32))
    mean_radial_velocity_error = verify_array_is_3d(
        read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity_error", dtype=np.float32)
    )
    num_frames = check_axis_length(
        (
            (1, mean_radial_velocity.shape),
            (1, mean_radial_velocity_error.shape),
        )
    )
    num_locations = check_axis_length(
        (
            (2, mean_radial_velocity.shape),
            (2, mean_radial_velocity_error.shape),
        )
    )
    num_momentum_bins = check_axis_length(
        (
            (0, angular_momentum.shape),
            (0, mean_radial_velocity.shape),
            (0, mean_radial_velocity_error.shape),
        )
    )

    return cls(
        distance_error=distance_error,
        max_lz=max_lz,
        min_lz=min_lz,
        name=name,
        sphere_radius=sphere_radius,
        angular_momentum=angular_momentum.astype(np.float64),
        mean_radial_velocity=mean_radial_velocity.astype(np.float64),
        mean_radial_velocity_error=mean_radial_velocity_error.astype(np.float64),
    )


def load_v4(file: Hdf5File) -> WrinkleData:
    cls = WrinkleData
    distance_error = get_float_attr_from_hdf5(file, "distance_error")
    max_lz = get_float_attr_from_hdf5(file, "max_lz")
    min_lz = get_float_attr_from_hdf5(file, "min_lz")
    name = get_str_attr_from_hdf5(file, "name")
    sphere_radius = get_float_attr_from_hdf5(file, "sphere_radius")
    angular_momentum = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "angular_momentum", dtype=np.float64))
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    mean_radial_velocity = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity", dtype=np.float64))
    mean_radial_velocity_error = verify_array_is_3d(
        read_dataset_from_hdf5_with_dtype(file, "mean_radial_velocity_error", dtype=np.float64)
    )
    num_frames = check_axis_length(
        (
            (1, mean_radial_velocity.shape),
            (1, mean_radial_velocity_error.shape),
        )
    )
    num_locations = check_axis_length(
        (
            (2, mean_radial_velocity.shape),
            (2, mean_radial_velocity_error.shape),
        )
    )
    num_momentum_bins = check_axis_length(
        (
            (0, angular_momentum.shape),
            (0, mean_radial_velocity.shape),
            (0, mean_radial_velocity_error.shape),
        )
    )

    return cls(
        distance_error=distance_error,
        max_lz=max_lz,
        min_lz=min_lz,
        name=name,
        sphere_radius=sphere_radius,
        angular_momentum=angular_momentum,
        completeness=completeness,
        mean_radial_velocity=mean_radial_velocity,
        mean_radial_velocity_error=mean_radial_velocity_error,
    )


_LOADERS: Mapping[int, _WrinkleDataLoader] = {
    3: load_v3,
    4: load_v4,
}
