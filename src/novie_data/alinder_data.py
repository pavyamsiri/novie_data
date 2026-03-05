from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, override

import numpy as np
from h5py import File as Hdf5File
from novie_helpers import (
    check_axis_length,
    get_file_version,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_2d,
    verify_array_is_4d,
)
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_helpers import Array2D, Array4D


LATEST_VERSION_V1 = Version("1.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _AlinderDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> AlinderData: ...


class AlinderData:
    DATA_FILE_TYPE: ClassVar[str] = "Alinder"
    VERSION: ClassVar[Version] = LATEST_VERSION_V1


    def __init__(
        self,
        *,
        alpha: Array2D[np.float32],
        b: Array2D[np.float32],
        background: Array4D[np.float32],
        c: Array2D[np.float32],
        rho: Array2D[np.float32],
        scale_factor: Array2D[np.float32],
        theta0: Array2D[np.float32],
        name: str = "UNKNOWN",
    ) -> None:
        num_frames = check_axis_length(
            ((0, alpha.shape), (0, b.shape), (2, background.shape), (0, c.shape), (0, rho.shape), (0, scale_factor.shape), (0, theta0.shape))
        )
        num_height_bins = check_axis_length(
            ((1, background.shape),)
        )
        num_locations = check_axis_length(
            ((1, alpha.shape), (1, b.shape), (3, background.shape), (1, c.shape), (1, rho.shape), (1, scale_factor.shape), (1, theta0.shape))
        )
        num_velocity_bins = check_axis_length(
            ((0, background.shape),)
        )
        self.num_frames: int = num_frames
        self.num_height_bins: int = num_height_bins
        self.num_locations: int = num_locations
        self.num_velocity_bins: int = num_velocity_bins
        self.name: str = name
        self.alpha: Array2D[np.float32] = alpha
        self.b: Array2D[np.float32] = b
        self.background: Array4D[np.float32] = background
        self.c: Array2D[np.float32] = c
        self.rho: Array2D[np.float32] = rho
        self.scale_factor: Array2D[np.float32] = scale_factor
        self.theta0: Array2D[np.float32] = theta0

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_frames == other.num_frames
        equality &= self.num_height_bins == other.num_height_bins
        equality &= self.num_locations == other.num_locations
        equality &= self.num_velocity_bins == other.num_velocity_bins
        equality &= self.name == other.name
        equality &= np.array_equal(self.alpha, other.alpha, equal_nan=True)
        equality &= np.array_equal(self.b, other.b, equal_nan=True)
        equality &= np.array_equal(self.background, other.background, equal_nan=True)
        equality &= np.array_equal(self.c, other.c, equal_nan=True)
        equality &= np.array_equal(self.rho, other.rho, equal_nan=True)
        equality &= np.array_equal(self.scale_factor, other.scale_factor, equal_nan=True)
        equality &= np.array_equal(self.theta0, other.theta0, equal_nan=True)
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
        alpha: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        b: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        background: Array4D[np.float32] = np.zeros((num_velocity_bins, num_height_bins, num_frames, num_locations), dtype=np.float32)
        c: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        rho: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        scale_factor: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        theta0: Array2D[np.float32] = np.zeros((num_frames, num_locations), dtype=np.float32)
        return cls(
            alpha=alpha,
            b=b,
            background=background,
            c=c,
            rho=rho,
            scale_factor=scale_factor,
            theta0=theta0,
        )

    @staticmethod
    def load(path: Path) -> AlinderData:
        path = path.expanduser()
        cls = AlinderData
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
            _ = file.create_dataset("alpha", data=self.alpha)
            _ = file.create_dataset("b", data=self.b)
            _ = file.create_dataset("background", data=self.background)
            _ = file.create_dataset("c", data=self.c)
            _ = file.create_dataset("rho", data=self.rho)
            _ = file.create_dataset("scale_factor", data=self.scale_factor)
            _ = file.create_dataset("theta0", data=self.theta0)
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




def load_v1(file: Hdf5File) -> AlinderData:
    cls = AlinderData
    name = get_str_attr_from_hdf5(file, "name")
    alpha = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "alpha", dtype=np.float32))
    b = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "b", dtype=np.float32))
    background = verify_array_is_4d(read_dataset_from_hdf5_with_dtype(file, "background", dtype=np.float32))
    c = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "c", dtype=np.float32))
    rho = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "rho", dtype=np.float32))
    scale_factor = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "scale_factor", dtype=np.float32))
    theta0 = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "theta0", dtype=np.float32))

    return cls(
        name=name,
        alpha=alpha,
        b=b,
        background=background,
        c=c,
        rho=rho,
        scale_factor=scale_factor,
        theta0=theta0,
    )


_LOADERS: Mapping[int, _AlinderDataLoader] = {
    1: load_v1,
}


