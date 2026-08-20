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

    from novie_helpers import Array1D, Array2D, Array3D


LATEST_VERSION_V1 = Version("1.0.0")
LATEST_VERSION_V2 = Version("2.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _SquashDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> SquashData: ...


class SquashData:
    DATA_FILE_TYPE: ClassVar[str] = "Squash"
    VERSION: ClassVar[Version] = LATEST_VERSION_V2


    def __init__(
        self,
        *,
        jphi_xy: Array3D[np.float64],
        jr_xy: Array3D[np.float64],
        jz_xy: Array3D[np.float64],
        omegaphi_xy: Array3D[np.float64],
        omegar_xy: Array3D[np.float64],
        omegaz_xy: Array3D[np.float64],
        stdz_xy: Array3D[np.float64],
        completeness: Array1D[np.bool_] | bool = True,
        extent: float = 25,
        jz_cutoff: float = 1,
        jz_threshold: float = 2.5,
        min_rxy: float = 2,
        name: str = "UNKNOWN",
        z_xy: Array3D[np.float64] | bool = False,
    ) -> None:
        num_bins = check_axis_length(
            ((0, jphi_xy.shape), (1, jphi_xy.shape), (0, jr_xy.shape), (1, jr_xy.shape), (0, jz_xy.shape), (1, jz_xy.shape), (0, omegaphi_xy.shape), (1, omegaphi_xy.shape), (0, omegar_xy.shape), (1, omegar_xy.shape), (0, omegaz_xy.shape), (1, omegaz_xy.shape), (0, stdz_xy.shape), (1, stdz_xy.shape))
        )
        num_frames = check_axis_length(
            ((2, jphi_xy.shape), (2, jr_xy.shape), (2, jz_xy.shape), (2, omegaphi_xy.shape), (2, omegar_xy.shape), (2, omegaz_xy.shape), (2, stdz_xy.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((num_frames,), dtype=np.bool_)
            case False:
                completeness = np.zeros((num_frames,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
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
        self.jz_cutoff: float = jz_cutoff
        self.jz_threshold: float = jz_threshold
        self.min_rxy: float = min_rxy
        self.name: str = name
        self.completeness: Array1D[np.bool_] = completeness
        self.jphi_xy: Array3D[np.float64] = jphi_xy
        self.jr_xy: Array3D[np.float64] = jr_xy
        self.jz_xy: Array3D[np.float64] = jz_xy
        self.omegaphi_xy: Array3D[np.float64] = omegaphi_xy
        self.omegar_xy: Array3D[np.float64] = omegar_xy
        self.omegaz_xy: Array3D[np.float64] = omegaz_xy
        self.stdz_xy: Array3D[np.float64] = stdz_xy
        self.z_xy: Array3D[np.float64] = z_xy

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_bins == other.num_bins
        equality &= self.num_frames == other.num_frames
        equality &= self.extent == other.extent
        equality &= self.jz_cutoff == other.jz_cutoff
        equality &= self.jz_threshold == other.jz_threshold
        equality &= self.min_rxy == other.min_rxy
        equality &= self.name == other.name
        equality &= np.array_equal(self.completeness, other.completeness, equal_nan=True)
        equality &= np.array_equal(self.jphi_xy, other.jphi_xy, equal_nan=True)
        equality &= np.array_equal(self.jr_xy, other.jr_xy, equal_nan=True)
        equality &= np.array_equal(self.jz_xy, other.jz_xy, equal_nan=True)
        equality &= np.array_equal(self.omegaphi_xy, other.omegaphi_xy, equal_nan=True)
        equality &= np.array_equal(self.omegar_xy, other.omegar_xy, equal_nan=True)
        equality &= np.array_equal(self.omegaz_xy, other.omegaz_xy, equal_nan=True)
        equality &= np.array_equal(self.stdz_xy, other.stdz_xy, equal_nan=True)
        equality &= np.array_equal(self.z_xy, other.z_xy, equal_nan=True)
        return bool(equality)

    @classmethod
    def empty(cls, *, num_bins: int, num_frames: int) -> Self:
        completeness: Array1D[np.bool_] = np.zeros((num_frames,), dtype=np.bool_)
        jphi_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        jr_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        jz_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        omegaphi_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        omegar_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        omegaz_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        stdz_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        z_xy: Array3D[np.float64] = np.zeros((num_bins, num_bins, num_frames), dtype=np.float64)
        return cls(
            completeness=completeness,
            jphi_xy=jphi_xy,
            jr_xy=jr_xy,
            jz_xy=jz_xy,
            omegaphi_xy=omegaphi_xy,
            omegar_xy=omegar_xy,
            omegaz_xy=omegaz_xy,
            stdz_xy=stdz_xy,
            z_xy=z_xy,
        )

    @staticmethod
    def load(path: Path) -> SquashData:
        path = path.expanduser()
        cls = SquashData
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
            file.attrs.create("jz_cutoff", self.jz_cutoff, dtype=np.float64)
            file.attrs.create("jz_threshold", self.jz_threshold, dtype=np.float64)
            file.attrs.create("min_rxy", self.min_rxy, dtype=np.float64)
            file.attrs.create("name", str(self.name))
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("jphi_xy", data=self.jphi_xy)
            _ = file.create_dataset("jr_xy", data=self.jr_xy)
            _ = file.create_dataset("jz_xy", data=self.jz_xy)
            _ = file.create_dataset("omegaphi_xy", data=self.omegaphi_xy)
            _ = file.create_dataset("omegar_xy", data=self.omegar_xy)
            _ = file.create_dataset("omegaz_xy", data=self.omegaz_xy)
            _ = file.create_dataset("stdz_xy", data=self.stdz_xy)
            _ = file.create_dataset("z_xy", data=self.z_xy)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def is_compatible(
        cls,
        path: Path,
        *,
        extent: float,
        jz_cutoff: float,
        jz_threshold: float,
        min_rxy: float,
        name: str,
    ) -> bool:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't check compatibility of {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        is_compatible: bool = True
        with Hdf5File(path, "r") as file:
            is_compatible &= extent == get_float_attr_from_hdf5(file, "extent")
            is_compatible &= jz_cutoff == get_float_attr_from_hdf5(file, "jz_cutoff")
            is_compatible &= jz_threshold == get_float_attr_from_hdf5(file, "jz_threshold")
            is_compatible &= min_rxy == get_float_attr_from_hdf5(file, "min_rxy")
            is_compatible &= name == get_str_attr_from_hdf5(file, "name")
        return is_compatible

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        extent: float,
        jz_cutoff: float,
        jz_threshold: float,
        min_rxy: float,
        name: str,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("extent", extent)
            file.attrs.modify("jz_cutoff", jz_cutoff)
            file.attrs.modify("jz_threshold", jz_threshold)
            file.attrs.modify("min_rxy", min_rxy)
            file.attrs.modify("name", name)
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        jphi_xy: Array2D[np.float64],
        jr_xy: Array2D[np.float64],
        jz_xy: Array2D[np.float64],
        omegaphi_xy: Array2D[np.float64],
        omegar_xy: Array2D[np.float64],
        omegaz_xy: Array2D[np.float64],
        stdz_xy: Array2D[np.float64],
        z_xy: Array2D[np.float64],
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        num_bins = check_axis_length(
            ((0, jphi_xy.shape), (1, jphi_xy.shape), (0, jr_xy.shape), (1, jr_xy.shape), (0, jz_xy.shape), (1, jz_xy.shape), (0, omegaphi_xy.shape), (1, omegaphi_xy.shape), (0, omegar_xy.shape), (1, omegar_xy.shape), (0, omegaz_xy.shape), (1, omegaz_xy.shape), (0, stdz_xy.shape), (1, stdz_xy.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "jphi_xy").write_direct(
                np.asarray(jphi_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "jr_xy").write_direct(
                np.asarray(jr_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "jz_xy").write_direct(
                np.asarray(jz_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "omegaphi_xy").write_direct(
                np.asarray(omegaphi_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "omegar_xy").write_direct(
                np.asarray(omegar_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "omegaz_xy").write_direct(
                np.asarray(omegaz_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "stdz_xy").write_direct(
                np.asarray(stdz_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "z_xy").write_direct(
                np.asarray(z_xy, dtype=np.float64).reshape((num_bins, num_bins)), np.s_[:, :], np.s_[:, :, frame]
            )
        log.info("Successfully saved frame %d of [cyan]%s[/cyan] to [magenta]%s[/magenta]", frame, cls.__name__, path)


def load_v1(file: Hdf5File) -> SquashData:
    cls = SquashData
    extent = get_float_attr_from_hdf5(file, "extent")
    jz_cutoff = get_float_attr_from_hdf5(file, "jz_cutoff")
    jz_threshold = get_float_attr_from_hdf5(file, "jz_threshold")
    min_rxy = get_float_attr_from_hdf5(file, "min_rxy")
    name = get_str_attr_from_hdf5(file, "name")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    jphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jphi_xy", dtype=np.float64))
    jr_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jr_xy", dtype=np.float64))
    jz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jz_xy", dtype=np.float64))
    omegaphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegaphi_xy", dtype=np.float64))
    omegar_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegar_xy", dtype=np.float64))
    omegaz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegaz_xy", dtype=np.float64))
    stdz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "stdz_xy", dtype=np.float64))

    return cls(
        extent=extent,
        jz_cutoff=jz_cutoff,
        jz_threshold=jz_threshold,
        min_rxy=min_rxy,
        name=name,
        completeness=completeness,
        jphi_xy=jphi_xy,
        jr_xy=jr_xy,
        jz_xy=jz_xy,
        omegaphi_xy=omegaphi_xy,
        omegar_xy=omegar_xy,
        omegaz_xy=omegaz_xy,
        stdz_xy=stdz_xy,
    )


def load_v2(file: Hdf5File) -> SquashData:
    cls = SquashData
    extent = get_float_attr_from_hdf5(file, "extent")
    jz_cutoff = get_float_attr_from_hdf5(file, "jz_cutoff")
    jz_threshold = get_float_attr_from_hdf5(file, "jz_threshold")
    min_rxy = get_float_attr_from_hdf5(file, "min_rxy")
    name = get_str_attr_from_hdf5(file, "name")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    jphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jphi_xy", dtype=np.float64))
    jr_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jr_xy", dtype=np.float64))
    jz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "jz_xy", dtype=np.float64))
    omegaphi_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegaphi_xy", dtype=np.float64))
    omegar_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegar_xy", dtype=np.float64))
    omegaz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "omegaz_xy", dtype=np.float64))
    stdz_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "stdz_xy", dtype=np.float64))
    z_xy = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "z_xy", dtype=np.float64))

    return cls(
        extent=extent,
        jz_cutoff=jz_cutoff,
        jz_threshold=jz_threshold,
        min_rxy=min_rxy,
        name=name,
        completeness=completeness,
        jphi_xy=jphi_xy,
        jr_xy=jr_xy,
        jz_xy=jz_xy,
        omegaphi_xy=omegaphi_xy,
        omegar_xy=omegar_xy,
        omegaz_xy=omegaz_xy,
        stdz_xy=stdz_xy,
        z_xy=z_xy,
    )


_LOADERS: Mapping[int, _SquashDataLoader] = {
    1: load_v1,
    2: load_v2,
}



