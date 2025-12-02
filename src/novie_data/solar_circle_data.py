from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version
from typing_extensions import override

from novie_helpers import (
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

    from novie_helpers import Array1D, Array2D, Array3D, Array4D


LATEST_VERSION_V0 = Version("0.1.0")


log: logging.Logger = logging.getLogger(__name__)


class _SolarCircleDataLoader(Protocol):
    def __call__(self, file: Hdf5File) -> SolarCircleData: ...


class SolarCircleData:
    DATA_FILE_TYPE: ClassVar[str] = "SolarCircle"
    VERSION: ClassVar[Version] = LATEST_VERSION_V0


    def __init__(
        self,
        *,
        name: str = "UNKNOWN",
        omega: float = 0,
        solar_radius: float = 8,
    ) -> None:
        self.name: str = name
        self.omega: float = omega
        self.solar_radius: float = solar_radius

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.name == other.name
        equality &= self.omega == other.omega
        equality &= self.solar_radius == other.solar_radius
        return bool(equality)

    @classmethod
    def empty(cls) -> Self:
        return cls(
        )

    @staticmethod
    def load(path: Path) -> SolarCircleData:
        path = path.expanduser()
        cls = SolarCircleData
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
            if file_version == LATEST_VERSION_V0:
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
            file.attrs.create("solar_radius", self.solar_radius, dtype=np.float64)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def is_compatible(
        cls,
        path: Path,
        *,
        name: str,
        omega: float,
        solar_radius: float,
    ) -> bool:
        path = path.expanduser()
        if not path.is_file():
            msg = f"Can't check compatibility of {cls.__name__} as {path} doesn't exist!"
            raise ValueError(msg)

        cls.migrate(path)
        is_compatible: bool = True
        with Hdf5File(path, "r") as file:
            is_compatible &= name == get_str_attr_from_hdf5(file, "name")
            is_compatible &= omega == get_float_attr_from_hdf5(file, "omega")
            is_compatible &= solar_radius == get_float_attr_from_hdf5(file, "solar_radius")
        return is_compatible




def load_v0(file: Hdf5File) -> SolarCircleData:
    cls = SolarCircleData
    name = get_str_attr_from_hdf5(file, "name")
    omega = get_float_attr_from_hdf5(file, "omega")
    solar_radius = get_float_attr_from_hdf5(file, "solar_radius")

    return cls(
        name=name,
        omega=omega,
        solar_radius=solar_radius,
    )


_LOADERS: Mapping[int, _SolarCircleDataLoader] = {
    0: load_v0,
}


