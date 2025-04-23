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

    _Array4D_f32: TypeAlias = Array4D[np.float32]
    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array4D_f64: TypeAlias = Array4D[np.float64]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]

LATEST_VERSION_V4 = Version("4.0.0")
LATEST_VERSION_V5 = Version("5.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _SnailDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> SnailData: ...


class SnailData:
    DATA_FILE_TYPE: ClassVar[str] = "Snail"
    VERSION: ClassVar[Version] = LATEST_VERSION_V5

    def __init__(
        self,
        *,
        azimuthal_velocity: _Array4D_f64,
        radial_velocity: _Array4D_f64,
        surface_density: _Array4D_f64,
        completeness: _Array1D_b8 | bool = True,
        max_height: float = 1,
        max_velocity: float = 60,
        name: str = "UNKNOWN",
        sphere_radius: float = 2,
    ) -> None:
        num_frames = check_axis_length(((2, azimuthal_velocity.shape), (2, radial_velocity.shape), (2, surface_density.shape)))
        num_height_bins = check_axis_length(
            ((1, azimuthal_velocity.shape), (1, radial_velocity.shape), (1, surface_density.shape))
        )
        num_locations = check_axis_length(((3, azimuthal_velocity.shape), (3, radial_velocity.shape), (3, surface_density.shape)))
        num_velocity_bins = check_axis_length(
            ((0, azimuthal_velocity.shape), (0, radial_velocity.shape), (0, surface_density.shape))
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
        self.num_height_bins: int = num_height_bins
        self.num_locations: int = num_locations
        self.num_velocity_bins: int = num_velocity_bins
        self.max_height: float = max_height
        self.max_velocity: float = max_velocity
        self.name: str = name
        self.sphere_radius: float = sphere_radius
        self.azimuthal_velocity: _Array4D_f64 = azimuthal_velocity
        self.completeness: _Array1D_b8 = completeness
        self.radial_velocity: _Array4D_f64 = radial_velocity
        self.surface_density: _Array4D_f64 = surface_density

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.num_height_bins == other.num_height_bins
        equality &= self.num_locations == other.num_locations
        equality &= self.num_velocity_bins == other.num_velocity_bins
        equality &= self.max_height == other.max_height
        equality &= self.max_velocity == other.max_velocity
        equality &= self.name == other.name
        equality &= self.sphere_radius == other.sphere_radius
        equality &= np.array_equal(self.azimuthal_velocity, other.azimuthal_velocity)
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.radial_velocity, other.radial_velocity)
        equality &= np.array_equal(self.surface_density, other.surface_density)
        return bool(equality)

    @classmethod
    def empty(
        cls,
        *,
        num_frames: int,
        num_height_bins: int,
        num_locations: int,
        num_velocity_bins: int,
    ) -> Self:
        azimuthal_velocity: _Array4D_f64 = np.zeros(
            (num_velocity_bins, num_height_bins, num_frames, num_locations), dtype=np.float64
        )
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        radial_velocity: _Array4D_f64 = np.zeros(
            (num_velocity_bins, num_height_bins, num_frames, num_locations), dtype=np.float64
        )
        surface_density: _Array4D_f64 = np.zeros(
            (num_velocity_bins, num_height_bins, num_frames, num_locations), dtype=np.float64
        )
        return cls(
            azimuthal_velocity=azimuthal_velocity,
            completeness=completeness,
            radial_velocity=radial_velocity,
            surface_density=surface_density,
        )

    @staticmethod
    def load(path: Path) -> SnailData:
        path = path.expanduser()
        cls = SnailData
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
            if file_version == LATEST_VERSION_V5:
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
            file.attrs.create("max_height", self.max_height, dtype=np.float64)
            file.attrs.create("max_velocity", self.max_velocity, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            file.attrs.create("sphere_radius", self.sphere_radius, dtype=np.float64)
            _ = file.create_dataset("azimuthal_velocity", data=self.azimuthal_velocity)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("radial_velocity", data=self.radial_velocity)
            _ = file.create_dataset("surface_density", data=self.surface_density)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        max_height: float,
        max_velocity: float,
        name: str,
        sphere_radius: float,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("max_height", max_height)
            file.attrs.modify("max_velocity", max_velocity)
            file.attrs.modify("name", name)
            file.attrs.modify("sphere_radius", sphere_radius)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        azimuthal_velocity: _Array3D_f64,
        radial_velocity: _Array3D_f64,
        surface_density: _Array3D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_height_bins = check_axis_length(
            ((1, azimuthal_velocity.shape), (1, radial_velocity.shape), (1, surface_density.shape))
        )
        num_locations = check_axis_length(((2, azimuthal_velocity.shape), (2, radial_velocity.shape), (2, surface_density.shape)))
        num_velocity_bins = check_axis_length(
            ((0, azimuthal_velocity.shape), (0, radial_velocity.shape), (0, surface_density.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "azimuthal_velocity").write_direct(
                np.asarray(azimuthal_velocity, dtype=np.float64).reshape((num_velocity_bins, num_height_bins, num_locations)),
                np.s_[:, :, :],
                np.s_[:, :, frame, :],
            )
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "radial_velocity").write_direct(
                np.asarray(radial_velocity, dtype=np.float64).reshape((num_velocity_bins, num_height_bins, num_locations)),
                np.s_[:, :, :],
                np.s_[:, :, frame, :],
            )
            get_dataset_from_hdf5(file, "surface_density").write_direct(
                np.asarray(surface_density, dtype=np.float64).reshape((num_velocity_bins, num_height_bins, num_locations)),
                np.s_[:, :, :],
                np.s_[:, :, frame, :],
            )
        log.info("Successfully saved frame {frame} of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)


def load_v4(file: Hdf5File) -> SnailData:
    cls = SnailData
    max_height = get_float_attr_from_hdf5(file, "max_height")
    max_velocity = get_float_attr_from_hdf5(file, "max_velocity")
    name = get_str_attr_from_hdf5(file, "name")
    sphere_radius = get_float_attr_from_hdf5(file, "sphere_radius")
    azimuthal_velocity = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "azimuthal_velocity", dtype=np.float32))
    radial_velocity = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "radial_velocity", dtype=np.float32))
    surface_density = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "surface_density", dtype=np.float32))
    num_frames = check_axis_length(
        (
            (2, azimuthal_velocity.shape),
            (2, radial_velocity.shape),
            (2, surface_density.shape),
        )
    )
    num_height_bins = check_axis_length(
        (
            (1, azimuthal_velocity.shape),
            (1, radial_velocity.shape),
            (1, surface_density.shape),
        )
    )
    num_locations = check_axis_length(
        (
            (3, azimuthal_velocity.shape),
            (3, radial_velocity.shape),
            (3, surface_density.shape),
        )
    )
    num_velocity_bins = check_axis_length(
        (
            (0, azimuthal_velocity.shape),
            (0, radial_velocity.shape),
            (0, surface_density.shape),
        )
    )

    return cls(
        max_height=max_height,
        max_velocity=max_velocity,
        name=name,
        sphere_radius=sphere_radius,
        azimuthal_velocity=azimuthal_velocity.astype(np.float64),
        radial_velocity=radial_velocity.astype(np.float64),
        surface_density=surface_density.astype(np.float64),
    )


def load_v5(file: Hdf5File) -> SnailData:
    cls = SnailData
    max_height = get_float_attr_from_hdf5(file, "max_height")
    max_velocity = get_float_attr_from_hdf5(file, "max_velocity")
    name = get_str_attr_from_hdf5(file, "name")
    sphere_radius = get_float_attr_from_hdf5(file, "sphere_radius")
    azimuthal_velocity = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "azimuthal_velocity", dtype=np.float64))
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    radial_velocity = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "radial_velocity", dtype=np.float64))
    surface_density = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "surface_density", dtype=np.float64))
    num_frames = check_axis_length(
        (
            (2, azimuthal_velocity.shape),
            (2, radial_velocity.shape),
            (2, surface_density.shape),
        )
    )
    num_height_bins = check_axis_length(
        (
            (1, azimuthal_velocity.shape),
            (1, radial_velocity.shape),
            (1, surface_density.shape),
        )
    )
    num_locations = check_axis_length(
        (
            (3, azimuthal_velocity.shape),
            (3, radial_velocity.shape),
            (3, surface_density.shape),
        )
    )
    num_velocity_bins = check_axis_length(
        (
            (0, azimuthal_velocity.shape),
            (0, radial_velocity.shape),
            (0, surface_density.shape),
        )
    )

    return cls(
        max_height=max_height,
        max_velocity=max_velocity,
        name=name,
        sphere_radius=sphere_radius,
        azimuthal_velocity=azimuthal_velocity,
        completeness=completeness,
        radial_velocity=radial_velocity,
        surface_density=surface_density,
    )


_LOADERS: Mapping[int, _SnailDataLoader] = {
    4: load_v4,
    5: load_v5,
}
