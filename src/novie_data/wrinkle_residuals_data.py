from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias, override

import numpy as np
from h5py import File as Hdf5File
from novie_helpers import (
    check_axis_length,
    get_file_version,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_2d,
    verify_array_is_3d,
)
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_helpers import Array1D, Array2D, Array3D

    _Array1D_f32: TypeAlias = Array1D[np.float32]
    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array1D_f64: TypeAlias = Array1D[np.float64]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V3 = Version("3.0.0")
LATEST_VERSION_V4 = Version("4.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _WrinkleResidualsDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> WrinkleResidualsData: ...


class WrinkleResidualsData:
    DATA_FILE_TYPE: ClassVar[str] = "WrinkleResiduals"
    VERSION: ClassVar[Version] = LATEST_VERSION_V4


    def __init__(
        self,
        *,
        bin_values: _Array1D_f64,
        metric: _Array3D_f64,
        summary: _Array2D_f64,
        metric_name: str = "UNSET",
        name: str = "UNKNOWN",
    ) -> None:
        num_bins = check_axis_length(
            ((0, bin_values.shape), (0, metric.shape))
        )
        num_frames = check_axis_length(
            ((1, metric.shape), (0, summary.shape))
        )
        num_locations = check_axis_length(
            ((2, metric.shape), (1, summary.shape))
        )
        self.num_bins: int = num_bins
        self.num_frames: int = num_frames
        self.num_locations: int = num_locations
        self.metric_name: str = metric_name
        self.name: str = name
        self.bin_values: _Array1D_f64 = bin_values
        self.metric: _Array3D_f64 = metric
        self.summary: _Array2D_f64 = summary

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_bins == other.num_bins
        equality &= self.num_frames == other.num_frames
        equality &= self.num_locations == other.num_locations
        equality &= self.metric_name == other.metric_name
        equality &= self.name == other.name
        equality &= np.array_equal(self.bin_values, other.bin_values)
        equality &= np.array_equal(self.metric, other.metric)
        equality &= np.array_equal(self.summary, other.summary)
        return bool(equality)

    @classmethod
    def empty(cls, *, num_bins: int, num_frames: int, num_locations: int) -> Self:
        bin_values: _Array1D_f64 = np.zeros((num_bins,), dtype=np.float64)
        metric: _Array3D_f64 = np.zeros((num_bins, num_frames, num_locations), dtype=np.float64)
        summary: _Array2D_f64 = np.zeros((num_frames, num_locations), dtype=np.float64)
        return cls(
            bin_values=bin_values,
            metric=metric,
            summary=summary,
        )

    @staticmethod
    def load(path: Path) -> WrinkleResidualsData:
        path = path.expanduser()
        cls = WrinkleResidualsData
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
            file.attrs.create("metric_name", str(self.metric_name))
            file.attrs.create("name", str(self.name))
            _ = file.create_dataset("bin_values", data=self.bin_values)
            _ = file.create_dataset("metric", data=self.metric)
            _ = file.create_dataset("summary", data=self.summary)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())




def load_v3(file: Hdf5File) -> WrinkleResidualsData:
    cls = WrinkleResidualsData
    metric_name = get_str_attr_from_hdf5(file, "metric_name")
    name = get_str_attr_from_hdf5(file, "name")
    bin_values = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "bin_values", dtype=np.float32))
    metric = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "metric", dtype=np.float32))
    summary = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "summary", dtype=np.float32))

    return cls(
        metric_name=metric_name,
        name=name,
        bin_values=bin_values.astype(np.float64),
        metric=metric.astype(np.float64),
        summary=summary.astype(np.float64),
    )


def load_v4(file: Hdf5File) -> WrinkleResidualsData:
    cls = WrinkleResidualsData
    metric_name = get_str_attr_from_hdf5(file, "metric_name")
    name = get_str_attr_from_hdf5(file, "name")
    bin_values = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "bin_values", dtype=np.float64))
    metric = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "metric", dtype=np.float64))
    summary = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "summary", dtype=np.float64))

    return cls(
        metric_name=metric_name,
        name=name,
        bin_values=bin_values,
        metric=metric,
        summary=summary,
    )


_LOADERS: Mapping[int, _WrinkleResidualsDataLoader] = {
    3: load_v3,
    4: load_v4,
}


