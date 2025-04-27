from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias, override

import numpy as np
from h5py import File as Hdf5File
from novie_helpers import (
    check_axis_length,
    get_dataset_from_hdf5,
    get_file_version,
    get_float_attr_from_hdf5,
    get_str_attr_from_hdf5,
    get_string_sequence_from_hdf5,
    read_dataset_from_hdf5_with_dtype,
    verify_array_is_1d,
    verify_array_is_3d,
)
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from novie_helpers import Array1D, Array2D, Array3D

    _Array3D_f32: TypeAlias = Array3D[np.float32]
    _Array2D_f32: TypeAlias = Array2D[np.float32]
    _Array3D_u32: TypeAlias = Array3D[np.uint32]
    _Array2D_u32: TypeAlias = Array2D[np.uint32]
    _Array1D_b8: TypeAlias = Array1D[np.bool_]
    _Array3D_f64: TypeAlias = Array3D[np.float64]
    _Array2D_f64: TypeAlias = Array2D[np.float64]

LATEST_VERSION_V2 = Version("2.0.0")
LATEST_VERSION_V3 = Version("3.0.0")
LATEST_VERSION_V4 = Version("4.0.0")


log: logging.Logger = logging.getLogger(__name__)


class _SpiralArmCoverageDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> SpiralArmCoverageData: ...


class SpiralArmCoverageData:
    DATA_FILE_TYPE: ClassVar[str] = "SpiralArmCoverage"
    VERSION: ClassVar[Version] = LATEST_VERSION_V4


    def __init__(
        self,
        *,
        arm_names: tuple[str, ...],
        covered_arm_normalised_densities: _Array3D_f64,
        num_covered_arm_pixels: _Array3D_u32,
        num_total_arm_pixels: _Array3D_u32,
        completeness: _Array1D_b8 | bool = True,
        name: str = "UNKNOWN",
        omega: float = 0,
    ) -> None:
        num_arms = check_axis_length(
            ((0, (len(arm_names),)), (1, covered_arm_normalised_densities.shape), (1, num_covered_arm_pixels.shape), (1, num_total_arm_pixels.shape))
        )
        num_frames = check_axis_length(
            ((2, covered_arm_normalised_densities.shape), (2, num_covered_arm_pixels.shape), (2, num_total_arm_pixels.shape))
        )
        num_locations = check_axis_length(
            ((0, covered_arm_normalised_densities.shape), (0, num_covered_arm_pixels.shape), (0, num_total_arm_pixels.shape))
        )
        match completeness:
            case True:
                completeness = np.ones((num_frames,), dtype=np.bool_)
            case False:
                completeness = np.zeros((num_frames,), dtype=np.bool_)
            case _:
                pass
        assert completeness is not bool
        self.num_arms: int = num_arms
        self.num_frames: int = num_frames
        self.num_locations: int = num_locations
        self.name: str = name
        self.omega: float = omega
        self.arm_names: tuple[str, ...] = arm_names
        self.completeness: _Array1D_b8 = completeness
        self.covered_arm_normalised_densities: _Array3D_f64 = covered_arm_normalised_densities
        self.num_covered_arm_pixels: _Array3D_u32 = num_covered_arm_pixels
        self.num_total_arm_pixels: _Array3D_u32 = num_total_arm_pixels

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.num_arms == other.num_arms
        equality &= self.num_frames == other.num_frames
        equality &= self.num_locations == other.num_locations
        equality &= self.name == other.name
        equality &= self.omega == other.omega
        equality &= len(self.arm_names) == len(other.arm_names) and all(
            x == y for x, y in zip(self.arm_names, other.arm_names, strict=True)
        )
        equality &= np.array_equal(self.completeness, other.completeness, equal_nan=True)
        equality &= np.array_equal(self.covered_arm_normalised_densities, other.covered_arm_normalised_densities, equal_nan=True)
        equality &= np.array_equal(self.num_covered_arm_pixels, other.num_covered_arm_pixels, equal_nan=True)
        equality &= np.array_equal(self.num_total_arm_pixels, other.num_total_arm_pixels, equal_nan=True)
        return bool(equality)

    @classmethod
    def empty(cls, *, num_arms: int, num_frames: int, num_locations: int) -> Self:
        arm_names: tuple[str, ...] = tuple("UNSET" for _ in range(num_arms))
        completeness: _Array1D_b8 = np.zeros((num_frames,), dtype=np.bool_)
        covered_arm_normalised_densities: _Array3D_f64 = np.zeros((num_locations, num_arms, num_frames), dtype=np.float64)
        num_covered_arm_pixels: _Array3D_u32 = np.zeros((num_locations, num_arms, num_frames), dtype=np.uint32)
        num_total_arm_pixels: _Array3D_u32 = np.zeros((num_locations, num_arms, num_frames), dtype=np.uint32)
        return cls(
            arm_names=arm_names,
            completeness=completeness,
            covered_arm_normalised_densities=covered_arm_normalised_densities,
            num_covered_arm_pixels=num_covered_arm_pixels,
            num_total_arm_pixels=num_total_arm_pixels,
        )

    @staticmethod
    def load(path: Path) -> SpiralArmCoverageData:
        path = path.expanduser()
        cls = SpiralArmCoverageData
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
            file.attrs.create("name", str(self.name))
            file.attrs.create("omega", self.omega, dtype=np.float64)
            _ = file.create_dataset("arm_names", data=self.arm_names)
            _ = file.create_dataset("completeness", data=self.completeness)
            _ = file.create_dataset("covered_arm_normalised_densities", data=self.covered_arm_normalised_densities)
            _ = file.create_dataset("num_covered_arm_pixels", data=self.num_covered_arm_pixels)
            _ = file.create_dataset("num_total_arm_pixels", data=self.num_total_arm_pixels)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def save_init(
        cls,
        path: Path,
        *,
        arm_names: tuple[str, ...],
        name: str,
        omega: float,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            file.attrs.modify("name", name)
            file.attrs.modify("omega", omega)
            get_dataset_from_hdf5(file, "arm_names").write_direct(
                np.asarray(arm_names, dtype=np.object_), np.s_[:], np.s_[:]
            )
        log.info("Successfully saved attributes of [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path)

    @classmethod
    def save_frame(
        cls,
        path: Path,
        frame: int,
        *,
        covered_arm_normalised_densities: _Array2D_f64,
        num_covered_arm_pixels: _Array2D_u32,
        num_total_arm_pixels: _Array2D_u32,
    ) -> None:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't save initial data to {cls.__name__} as it doesn't exist!"
            raise ValueError(msg)

        num_arms = check_axis_length(
            ((1, covered_arm_normalised_densities.shape), (1, num_covered_arm_pixels.shape), (1, num_total_arm_pixels.shape))
        )
        num_locations = check_axis_length(
            ((0, covered_arm_normalised_densities.shape), (0, num_covered_arm_pixels.shape), (0, num_total_arm_pixels.shape))
        )
        cls.migrate(path)
        with Hdf5File(path, "a") as file:
            get_dataset_from_hdf5(file, "completeness").write_direct(
                np.asarray(True, dtype=np.bool_).reshape(1), np.s_[0], np.s_[frame]
            )
            get_dataset_from_hdf5(file, "covered_arm_normalised_densities").write_direct(
                np.asarray(covered_arm_normalised_densities, dtype=np.float64).reshape((num_locations, num_arms)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "num_covered_arm_pixels").write_direct(
                np.asarray(num_covered_arm_pixels, dtype=np.uint32).reshape((num_locations, num_arms)), np.s_[:, :], np.s_[:, :, frame]
            )
            get_dataset_from_hdf5(file, "num_total_arm_pixels").write_direct(
                np.asarray(num_total_arm_pixels, dtype=np.uint32).reshape((num_locations, num_arms)), np.s_[:, :], np.s_[:, :, frame]
            )
        log.info("Successfully saved frame %d of [cyan]%s[/cyan] to [magenta]%s[/magenta]", frame, cls.__name__, path)


def load_v2(file: Hdf5File) -> SpiralArmCoverageData:
    cls = SpiralArmCoverageData
    name = get_str_attr_from_hdf5(file, "name")
    arm_names = get_string_sequence_from_hdf5(file, "arm_names")
    covered_arm_normalised_densities = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "covered_arm_normalised_densities", dtype=np.float32))
    num_covered_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_covered_arm_pixels", dtype=np.uint32))
    num_total_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_total_arm_pixels", dtype=np.uint32))

    return cls(
        name=name,
        arm_names=arm_names,
        covered_arm_normalised_densities=covered_arm_normalised_densities.astype(np.float64),
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
    )


def load_v3(file: Hdf5File) -> SpiralArmCoverageData:
    cls = SpiralArmCoverageData
    name = get_str_attr_from_hdf5(file, "name")
    arm_names = get_string_sequence_from_hdf5(file, "arm_names")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    covered_arm_normalised_densities = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "covered_arm_normalised_densities", dtype=np.float64))
    num_covered_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_covered_arm_pixels", dtype=np.uint32))
    num_total_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_total_arm_pixels", dtype=np.uint32))

    return cls(
        name=name,
        arm_names=arm_names,
        completeness=completeness,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
    )


def load_v4(file: Hdf5File) -> SpiralArmCoverageData:
    cls = SpiralArmCoverageData
    name = get_str_attr_from_hdf5(file, "name")
    omega = get_float_attr_from_hdf5(file, "omega")
    arm_names = get_string_sequence_from_hdf5(file, "arm_names")
    completeness = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))
    covered_arm_normalised_densities = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "covered_arm_normalised_densities", dtype=np.float64))
    num_covered_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_covered_arm_pixels", dtype=np.uint32))
    num_total_arm_pixels = verify_array_is_3d(read_dataset_from_hdf5_with_dtype(file, "num_total_arm_pixels", dtype=np.uint32))

    return cls(
        name=name,
        omega=omega,
        arm_names=arm_names,
        completeness=completeness,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
    )


_LOADERS: Mapping[int, _SpiralArmCoverageDataLoader] = {
    2: load_v2,
    3: load_v3,
    4: load_v4,
}


