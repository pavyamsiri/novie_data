from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, override

import numpy as np
from h5py import File as Hdf5File
from novie_helpers import (
    check_axis_length,
    get_dataset_from_hdf5,
    get_file_version,
    get_float_attr_from_hdf5,
    get_str_attr_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_3d,
)
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from optype.numpy import Array1D, Array2D, Array3D


LATEST_VERSION_V1 = Version("1.0.0")
LATEST_VERSION_V2 = Version("2.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _VelocityGridDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> VelocityGridData: ...


class VelocityGridData:
    DATA_FILE_TYPE: ClassVar[str] = "VelocityGrid"
    VERSION: ClassVar[Version] = LATEST_VERSION_V2


    def __init__(
        self,
        *,
        vphi_xy: Array3D[np.float64],
        vr_xy: Array3D[np.float64],
        completeness: Array1D[np.bool_] | bool = True,
        extent: float = 1,
        name: str = "UNKNOWN",
        omega: float = 0,
        vz_xy: Array3D[np.float64] | bool = False,
        z_xy: Array3D[np.float64] | bool = False,
    ) -> None:
        num_bins = check_axis_length(
            ((0, vphi_xy.shape), (1, vphi_xy.shape), (0, vr_xy.shape), (1, vr_xy.shape))
        )
        num_frames = check_axis_length(
            ((2, vphi_xy.shape), (2, vr_xy.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((num_frames,), dtype=np.bool_)
            case False:
                completeness = np.zeros((num_frames,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
        match vz_xy:
            case True:
                vz_xy = np.ones((num_bins, num_bins, num_frames), dtype=np.float64)
            case False:
                vz_xy = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
            case _:
                pass
        assert vz_xy is not bool
        match z_xy:
            case True:
                z_xy = np.ones((num_bins, num_bins, num_frames), dtype=np.float64)
            case False:
                z_xy = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
            case _:
                pass
        assert z_xy is not bool
        self.num_bins: int = num_bins
        self.num_frames: int = num_frames
        self.extent: float = extent
        self.name: str = name
        self.omega: float = omega
        self.completeness: Array1D[np.bool_] = completeness
        self.vphi_xy: Array3D[np.float64] = vphi_xy
        self.vr_xy: Array3D[np.float64] = vr_xy
        self.vz_xy: Array3D[np.float64] = vz_xy
        self.z_xy: Array3D[np.float64] = z_xy

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_bins == other.num_bins
        equality &= self.num_frames == other.num_frames
        equality &= self.extent == other.extent
        equality &= self.name == other.name
        equality &= self.omega == other.omega
        equality &= np.array_equal(self.completeness, other.completeness, equal_nan=True)
        equality &= np.array_equal(self.vphi_xy, other.vphi_xy, equal_nan=True)
        equality &= np.array_equal(self.vr_xy, other.vr_xy, equal_nan=True)
        equality &= np.array_equal(self.vz_xy, other.vz_xy, equal_nan=True)
        equality &= np.array_equal(self.z_xy, other.z_xy, equal_nan=True)
        return bool(equality)

    @classmethod
    def empty(cls, *, num_bins: int, num_frames: int) -> Self:
        completeness: Array1D[np.bool_] = np.zeros((num_frames,), dtype=np.bool_)
        vphi_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        vr_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        vz_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        z_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        return cls(
            completeness=completeness,
            vphi_xy=vphi_xy,
            vr_xy=vr_xy,
            vz_xy=vz_xy,
            z_xy=z_xy,
        )

    @staticmethod
    def load(path: Path) -> VelocityGridData:
        path = path.expanduser()
        cls = VelocityGridData
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
            if file_version == LATEST_VERSION_V2:
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
            file.attrs.create("extent", self.extent, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            file.attrs.create("omega", self.omega, dtype=np.float64)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("vphi_xy", data=self.vphi_xy)
            _ = file.create_dataset("vr_xy", data=self.vr_xy)
            _ = file.create_dataset("vz_xy", data=self.vz_xy)
            _ = file.create_dataset("z_xy", data=self.z_xy)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def is_compatible(
        cls,
        path: Path,
        *,
        extent: float,
        name: str,
        omega: float,
    ) -> bool:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't check compatibility of {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        is_compatible: bool = True
        with Hdf5File(path, "r") as file:
            is_compatible &= extent == get_float_attr_from_hdf5(file, "extent")
            is_compatible &= name == get_str_attr_from_hdf5(file, "name")
            is_compatible &= omega == get_float_attr_from_hdf5(file, "omega")
        return is_compatible

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        extent: float,
        name: str,
        omega: float,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("extent", extent)
            file.attrs.modify("name", name)
            file.attrs.modify("omega", omega)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        vphi_xy: Array2D[np.float64],
        vr_xy: Array2D[np.float64],
        vz_xy: Array2D[np.float64],
        z_xy: Array2D[np.float64],
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        num_bins = check_axis_length(
            ((0, vphi_xy.shape), (1, vphi_xy.shape), (0, vr_xy.shape), (1, vr_xy.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "vphi_xy").write_direct(
                np.asarray(vphi_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "vr_xy").write_direct(
                np.asarray(vr_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "vz_xy").write_direct(
                np.asarray(vz_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "z_xy").write_direct(
                np.asarray(z_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
        log.info("Successfully saved frame %d of [cyan]%s[/cyan] to [magenta]%s[/magenta]", frame, cls.__name__, path)


def load_v1(file: Hdf5File) -> VelocityGridData:
    cls = VelocityGridData
    extent = get_float_attr_from_hdf5(file, "extent")
    name = get_str_attr_from_hdf5(file, "name")
    omega = get_float_attr_from_hdf5(file, "omega")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    vphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "vphi_xy", dtype=np.float64))
    vr_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "vr_xy", dtype=np.float64))

    return cls(
        extent=extent,
        name=name,
        omega=omega,
        completeness=completeness,
        vphi_xy=vphi_xy,
        vr_xy=vr_xy,
    )


def load_v2(file: Hdf5File) -> VelocityGridData:
    cls = VelocityGridData
    extent = get_float_attr_from_hdf5(file, "extent")
    name = get_str_attr_from_hdf5(file, "name")
    omega = get_float_attr_from_hdf5(file, "omega")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    vphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "vphi_xy", dtype=np.float64))
    vr_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "vr_xy", dtype=np.float64))
    vz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "vz_xy", dtype=np.float64))
    z_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "z_xy", dtype=np.float64))

    return cls(
        extent=extent,
        name=name,
        omega=omega,
        completeness=completeness,
        vphi_xy=vphi_xy,
        vr_xy=vr_xy,
        vz_xy=vz_xy,
        z_xy=z_xy,
    )


_LOADERS: Mapping[int, _VelocityGridDataLoader] = {
    1: load_v1,
    2: load_v2,
}


