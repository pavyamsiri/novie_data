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

    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V3 = Version("3.0.0")
LATEST_VERSION_V4 = Version("4.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _GridDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> GridData: ...


class GridData:
    DATA_FILE_TYPE: ClassVar[str] = "Grid"
    VERSION: ClassVar[Version] = LATEST_VERSION_V4


    def __init__(
        self,
        *,
        flat_projection_xy: _Array3D_f64,
        projection_xy: _Array3D_f64,
        projection_xz: _Array3D_f64,
        projection_yz: _Array3D_f64,
        completeness: _Array1D_b8 | bool = True,
        disc_scale_length: float = 1,
        disc_scale_mass: float = 1,
        extent: float = 1,
        name: str = "UNKNOWN",
    ) -> None:
        num_bins = check_axis_length(
            ((0, flat_projection_xy.shape), (1, flat_projection_xy.shape), (0, projection_xy.shape), (1, projection_xy.shape), (0, projection_xz.shape), (1, projection_xz.shape), (0, projection_yz.shape), (1, projection_yz.shape))
        )
        num_frames = check_axis_length(
            ((2, flat_projection_xy.shape), (2, projection_xy.shape), (2, projection_xz.shape), (2, projection_yz.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((num_frames,), dtype=np.bool_)
            case False:
                completeness = np.zeros((num_frames,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
        self.num_bins: int = num_bins
        self.num_frames: int = num_frames
        self.disc_scale_length: float = disc_scale_length
        self.disc_scale_mass: float = disc_scale_mass
        self.extent: float = extent
        self.name: str = name
        self.completeness: _Array1D_b8 = completeness
        self.flat_projection_xy: _Array3D_f64 = flat_projection_xy
        self.projection_xy: _Array3D_f64 = projection_xy
        self.projection_xz: _Array3D_f64 = projection_xz
        self.projection_yz: _Array3D_f64 = projection_yz

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_bins == other.num_bins
        equality &= self.num_frames == other.num_frames
        equality &= self.disc_scale_length == other.disc_scale_length
        equality &= self.disc_scale_mass == other.disc_scale_mass
        equality &= self.extent == other.extent
        equality &= self.name == other.name
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.flat_projection_xy, other.flat_projection_xy)
        equality &= np.array_equal(self.projection_xy, other.projection_xy)
        equality &= np.array_equal(self.projection_xz, other.projection_xz)
        equality &= np.array_equal(self.projection_yz, other.projection_yz)
        return bool(equality)

    @classmethod
    def empty(cls, *, num_bins: int, num_frames: int) -> Self:
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        flat_projection_xy: _Array3D_f64 = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        projection_xy: _Array3D_f64 = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        projection_xz: _Array3D_f64 = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        projection_yz: _Array3D_f64 = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        return cls(
            completeness=completeness,
            flat_projection_xy=flat_projection_xy,
            projection_xy=projection_xy,
            projection_xz=projection_xz,
            projection_yz=projection_yz,
        )

    @staticmethod
    def load(path: Path) -> GridData:
        path = path.expanduser()
        cls = GridData
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
            file.attrs.create("disc_scale_length", self.disc_scale_length, dtype=np.float64)
            file.attrs.create("disc_scale_mass", self.disc_scale_mass, dtype=np.float64)
            file.attrs.create("extent", self.extent, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("flat_projection_xy", data=self.flat_projection_xy)
            _ = file.create_dataset("projection_xy", data=self.projection_xy)
            _ = file.create_dataset("projection_xz", data=self.projection_xz)
            _ = file.create_dataset("projection_yz", data=self.projection_yz)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        disc_scale_length: float,
        disc_scale_mass: float,
        extent: float,
        name: str,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("disc_scale_length", disc_scale_length)
            file.attrs.modify("disc_scale_mass", disc_scale_mass)
            file.attrs.modify("extent", extent)
            file.attrs.modify("name", name)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        flat_projection_xy: _Array2D_f64,
        projection_xy: _Array2D_f64,
        projection_xz: _Array2D_f64,
        projection_yz: _Array2D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_bins = check_axis_length(
            ((0, flat_projection_xy.shape), (1, flat_projection_xy.shape), (0, projection_xy.shape), (1, projection_xy.shape), (0, projection_xz.shape), (1, projection_xz.shape), (0, projection_yz.shape), (1, projection_yz.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "flat_projection_xy").write_direct(
                np.asarray(flat_projection_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "projection_xy").write_direct(
                np.asarray(projection_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "projection_xz").write_direct(
                np.asarray(projection_xz, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "projection_yz").write_direct(
                np.asarray(projection_yz, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
        log.info("Successfully saved frame {frame} of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)


def load_v3(file: Hdf5File) -> GridData:
    cls = GridData
    disc_scale_length = get_float_attr_from_hdf5(file, "disc_scale_length")
    disc_scale_mass = get_float_attr_from_hdf5(file, "disc_scale_mass")
    extent = get_float_attr_from_hdf5(file, "extent")
    name = get_str_attr_from_hdf5(file, "name")
    flat_projection_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "flat_projection_xy", dtype=np.float32))
    projection_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xy", dtype=np.float32))
    projection_xz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xz", dtype=np.float32))
    projection_yz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_yz", dtype=np.float32))
    num_bins = check_axis_length(
        ((0, flat_projection_xy.shape), (1, flat_projection_xy.shape), (0, projection_xy.shape), (1, projection_xy.shape), (0, projection_xz.shape), (1, projection_xz.shape), (0, projection_yz.shape), (1, projection_yz.shape),)
    )
    num_frames = check_axis_length(
        ((2, flat_projection_xy.shape), (2, projection_xy.shape), (2, projection_xz.shape), (2, projection_yz.shape),)
    )

    return cls(
        disc_scale_length=disc_scale_length,
        disc_scale_mass=disc_scale_mass,
        extent=extent,
        name=name,
        flat_projection_xy=flat_projection_xy.astype(np.float64),
        projection_xy=projection_xy.astype(np.float64),
        projection_xz=projection_xz.astype(np.float64),
        projection_yz=projection_yz.astype(np.float64),
    )


def load_v4(file: Hdf5File) -> GridData:
    cls = GridData
    disc_scale_length = get_float_attr_from_hdf5(file, "disc_scale_length")
    disc_scale_mass = get_float_attr_from_hdf5(file, "disc_scale_mass")
    extent = get_float_attr_from_hdf5(file, "extent")
    name = get_str_attr_from_hdf5(file, "name")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    flat_projection_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "flat_projection_xy", dtype=np.float64))
    projection_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xy", dtype=np.float64))
    projection_xz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_xz", dtype=np.float64))
    projection_yz = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "projection_yz", dtype=np.float64))
    num_bins = check_axis_length(
        ((0, flat_projection_xy.shape), (1, flat_projection_xy.shape), (0, projection_xy.shape), (1, projection_xy.shape), (0, projection_xz.shape), (1, projection_xz.shape), (0, projection_yz.shape), (1, projection_yz.shape),)
    )
    num_frames = check_axis_length(
        ((2, flat_projection_xy.shape), (2, projection_xy.shape), (2, projection_xz.shape), (2, projection_yz.shape),)
    )

    return cls(
        disc_scale_length=disc_scale_length,
        disc_scale_mass=disc_scale_mass,
        extent=extent,
        name=name,
        completeness=completeness,
        flat_projection_xy=flat_projection_xy,
        projection_xy=projection_xy,
        projection_xz=projection_xz,
        projection_yz=projection_yz,
    )


_LOADERS: Mapping[int, _GridDataLoader] = {
    3: load_v3,
    4: load_v4,
}


