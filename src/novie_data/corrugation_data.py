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

    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array4D_f32: TypeAlias = Array4D[np.float32]
    _Array1D_f32: TypeAlias = Array1D[np.float32]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]
    _Array4D_f64: TypeAlias = Array4D[np.float64]
    _Array1D_f64: TypeAlias = Array1D[np.float64]

LATEST_VERSION_V3 = Version("3.0.0")
LATEST_VERSION_V4 = Version("4.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _CorrugationDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> CorrugationData: ...


class CorrugationData:
    DATA_FILE_TYPE: ClassVar[str] = "Corrugation"
    VERSION: ClassVar[Version] = LATEST_VERSION_V4


    def __init__(
        self,
        *,
        mean_height: _Array3D_f64,
        mean_height_error: _Array3D_f64,
        projection_rz: _Array4D_f64,
        radii: _Array1D_f64,
        completeness: _Array1D_b8 | bool = True,
        cutoff_frequency: float = 0,
        distance_error: float = 0,
        inner_radius: float = 0,
        max_height: float = 2,
        max_longitude_deg: float = 245,
        max_radius: float = 12,
        min_longitude_deg: float = 225,
        min_radius: float = 0,
        name: str = "UNKNOWN",
        outer_radius: float = 7,
    ) -> None:
        num_frames = check_axis_length(
            ((1, mean_height.shape), (1, mean_height_error.shape), (2, projection_rz.shape))
        )
        num_height_bins = check_axis_length(
            ((0, projection_rz.shape),)
        )
        num_locations = check_axis_length(
            ((2, mean_height.shape), (2, mean_height_error.shape), (3, projection_rz.shape))
        )
        num_radial_bins = check_axis_length(
            ((0, mean_height.shape), (0, mean_height_error.shape), (1, projection_rz.shape), (0, radii.shape))
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
        self.num_radial_bins: int = num_radial_bins
        self.cutoff_frequency: float = cutoff_frequency
        self.distance_error: float = distance_error
        self.inner_radius: float = inner_radius
        self.max_height: float = max_height
        self.max_longitude_deg: float = max_longitude_deg
        self.max_radius: float = max_radius
        self.min_longitude_deg: float = min_longitude_deg
        self.min_radius: float = min_radius
        self.name: str = name
        self.outer_radius: float = outer_radius
        self.completeness: _Array1D_b8 = completeness
        self.mean_height: _Array3D_f64 = mean_height
        self.mean_height_error: _Array3D_f64 = mean_height_error
        self.projection_rz: _Array4D_f64 = projection_rz
        self.radii: _Array1D_f64 = radii

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.num_height_bins == other.num_height_bins
        equality &= self.num_locations == other.num_locations
        equality &= self.num_radial_bins == other.num_radial_bins
        equality &= self.cutoff_frequency == other.cutoff_frequency
        equality &= self.distance_error == other.distance_error
        equality &= self.inner_radius == other.inner_radius
        equality &= self.max_height == other.max_height
        equality &= self.max_longitude_deg == other.max_longitude_deg
        equality &= self.max_radius == other.max_radius
        equality &= self.min_longitude_deg == other.min_longitude_deg
        equality &= self.min_radius == other.min_radius
        equality &= self.name == other.name
        equality &= self.outer_radius == other.outer_radius
        equality &= np.array_equal(self.completeness, other.completeness)
        equality &= np.array_equal(self.mean_height, other.mean_height)
        equality &= np.array_equal(self.mean_height_error, other.mean_height_error)
        equality &= np.array_equal(self.projection_rz, other.projection_rz)
        equality &= np.array_equal(self.radii, other.radii)
        return bool(equality)

    @classmethod
    def empty(
        cls,
        *,
        num_frames: int,
        num_height_bins: int,
        num_locations: int,
        num_radial_bins: int,
    ) -> Self:
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        mean_height: _Array3D_f64 = np.zeros((num_radial_bins, num_frames, num_locations), dtype=np.float64)
        mean_height_error: _Array3D_f64 = np.zeros((num_radial_bins, num_frames, num_locations), dtype=np.float64)
        projection_rz: _Array4D_f64 = np.zeros((num_height_bins, num_radial_bins, num_frames, num_locations), dtype=np.float64)
        radii: _Array1D_f64 = np.zeros((num_radial_bins,), dtype=np.float64)
        return cls(
            completeness=completeness,
            mean_height=mean_height,
            mean_height_error=mean_height_error,
            projection_rz=projection_rz,
            radii=radii,
        )

    @staticmethod
    def load(path: Path) -> CorrugationData:
        path = path.expanduser()
        cls = CorrugationData
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
            file.attrs.create("cutoff_frequency", self.cutoff_frequency, dtype=np.float64)
            file.attrs.create("distance_error", self.distance_error, dtype=np.float64)
            file.attrs.create("inner_radius", self.inner_radius, dtype=np.float64)
            file.attrs.create("max_height", self.max_height, dtype=np.float64)
            file.attrs.create("max_longitude_deg", self.max_longitude_deg, dtype=np.float64)
            file.attrs.create("max_radius", self.max_radius, dtype=np.float64)
            file.attrs.create("min_longitude_deg", self.min_longitude_deg, dtype=np.float64)
            file.attrs.create("min_radius", self.min_radius, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            file.attrs.create("outer_radius", self.outer_radius, dtype=np.float64)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("mean_height", data=self.mean_height)
            _ = file.create_dataset("mean_height_error", data=self.mean_height_error)
            _ = file.create_dataset("projection_rz", data=self.projection_rz)
            _ = file.create_dataset("radii", data=self.radii)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        cutoff_frequency: float,
        distance_error: float,
        inner_radius: float,
        max_height: float,
        max_longitude_deg: float,
        max_radius: float,
        min_longitude_deg: float,
        min_radius: float,
        name: str,
        outer_radius: float,
        radii: _Array1D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("cutoff_frequency", cutoff_frequency)
            file.attrs.modify("distance_error", distance_error)
            file.attrs.modify("inner_radius", inner_radius)
            file.attrs.modify("max_height", max_height)
            file.attrs.modify("max_longitude_deg", max_longitude_deg)
            file.attrs.modify("max_radius", max_radius)
            file.attrs.modify("min_longitude_deg", min_longitude_deg)
            file.attrs.modify("min_radius", min_radius)
            file.attrs.modify("name", name)
            file.attrs.modify("outer_radius", outer_radius)
            get_dataset_from_hdf5(file, "radii").write_direct(
                np.asarray(radii, dtype=np.float64).reshape(radii.shape), np.s_[:], np.s_[:]
            )
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        mean_height: _Array2D_f64,
        mean_height_error: _Array2D_f64,
        projection_rz: _Array3D_f64,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_height_bins = check_axis_length(
            ((0, projection_rz.shape),)
        )
        num_locations = check_axis_length(
            ((1, mean_height.shape), (1, mean_height_error.shape), (2, projection_rz.shape))
        )
        num_radial_bins = check_axis_length(
            ((0, mean_height.shape), (0, mean_height_error.shape), (1, projection_rz.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "mean_height").write_direct(
                np.asarray(mean_height, dtype=np.float64).reshape((num_radial_bins, num_locations)), np.s_[:, :], np.s_[:, frame, :]
            )
            get_dataset_from_hdf5(file, "mean_height_error").write_direct(
                np.asarray(mean_height_error, dtype=np.float64).reshape((num_radial_bins, num_locations)), np.s_[:, :], np.s_[:, frame, :]
            )
            get_dataset_from_hdf5(file, "projection_rz").write_direct(
                np.asarray(projection_rz, dtype=np.float64).reshape((num_height_bins, num_radial_bins, num_locations)), np.s_[:, :, :], np.s_[:, :, frame, :]
            )
        log.info("Successfully saved frame {frame} of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)


def load_v3(file: Hdf5File) -> CorrugationData:
    cls = CorrugationData
    cutoff_frequency = get_float_attr_from_hdf5(file, "cutoff_frequency")
    distance_error = get_float_attr_from_hdf5(file, "distance_error")
    inner_radius = get_float_attr_from_hdf5(file, "inner_radius")
    max_height = get_float_attr_from_hdf5(file, "max_height")
    max_longitude_deg = get_float_attr_from_hdf5(file, "max_longitude_deg")
    max_radius = get_float_attr_from_hdf5(file, "max_radius")
    min_longitude_deg = get_float_attr_from_hdf5(file, "min_longitude_deg")
    min_radius = get_float_attr_from_hdf5(file, "min_radius")
    name = get_str_attr_from_hdf5(file, "name")
    outer_radius = get_float_attr_from_hdf5(file, "outer_radius")
    mean_height = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_height", dtype=np.float32))
    mean_height_error = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_height_error", dtype=np.float32))
    projection_rz = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "projection_rz", dtype=np.float32))
    radii = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "radii", dtype=np.float32))
    num_frames = check_axis_length(
        ((1, mean_height.shape), (1, mean_height_error.shape), (2, projection_rz.shape),)
    )
    num_height_bins = check_axis_length(
        ((0, projection_rz.shape),)
    )
    num_locations = check_axis_length(
        ((2, mean_height.shape), (2, mean_height_error.shape), (3, projection_rz.shape),)
    )
    num_radial_bins = check_axis_length(
        ((0, mean_height.shape), (0, mean_height_error.shape), (1, projection_rz.shape), (0, radii.shape),)
    )

    return cls(
        cutoff_frequency=cutoff_frequency,
        distance_error=distance_error,
        inner_radius=inner_radius,
        max_height=max_height,
        max_longitude_deg=max_longitude_deg,
        max_radius=max_radius,
        min_longitude_deg=min_longitude_deg,
        min_radius=min_radius,
        name=name,
        outer_radius=outer_radius,
        mean_height=mean_height.astype(np.float64),
        mean_height_error=mean_height_error.astype(np.float64),
        projection_rz=projection_rz.astype(np.float64),
        radii=radii.astype(np.float64),
    )


def load_v4(file: Hdf5File) -> CorrugationData:
    cls = CorrugationData
    cutoff_frequency = get_float_attr_from_hdf5(file, "cutoff_frequency")
    distance_error = get_float_attr_from_hdf5(file, "distance_error")
    inner_radius = get_float_attr_from_hdf5(file, "inner_radius")
    max_height = get_float_attr_from_hdf5(file, "max_height")
    max_longitude_deg = get_float_attr_from_hdf5(file, "max_longitude_deg")
    max_radius = get_float_attr_from_hdf5(file, "max_radius")
    min_longitude_deg = get_float_attr_from_hdf5(file, "min_longitude_deg")
    min_radius = get_float_attr_from_hdf5(file, "min_radius")
    name = get_str_attr_from_hdf5(file, "name")
    outer_radius = get_float_attr_from_hdf5(file, "outer_radius")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    mean_height = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_height", dtype=np.float64))
    mean_height_error = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "mean_height_error", dtype=np.float64))
    projection_rz = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "projection_rz", dtype=np.float64))
    radii = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "radii", dtype=np.float64))
    num_frames = check_axis_length(
        ((1, mean_height.shape), (1, mean_height_error.shape), (2, projection_rz.shape),)
    )
    num_height_bins = check_axis_length(
        ((0, projection_rz.shape),)
    )
    num_locations = check_axis_length(
        ((2, mean_height.shape), (2, mean_height_error.shape), (3, projection_rz.shape),)
    )
    num_radial_bins = check_axis_length(
        ((0, mean_height.shape), (0, mean_height_error.shape), (1, projection_rz.shape), (0, radii.shape),)
    )

    return cls(
        cutoff_frequency=cutoff_frequency,
        distance_error=distance_error,
        inner_radius=inner_radius,
        max_height=max_height,
        max_longitude_deg=max_longitude_deg,
        max_radius=max_radius,
        min_longitude_deg=min_longitude_deg,
        min_radius=min_radius,
        name=name,
        outer_radius=outer_radius,
        completeness=completeness,
        mean_height=mean_height,
        mean_height_error=mean_height_error,
        projection_rz=projection_rz,
        radii=radii,
    )


_LOADERS: Mapping[int, _CorrugationDataLoader] = {
    3: load_v3,
    4: load_v4,
}


