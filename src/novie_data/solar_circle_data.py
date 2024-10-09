"""Data to define the solar circle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

from h5py import File as Hdf5File
from packaging.version import Version

from .serde.accessors import get_float_attr_from_hdf5
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SolarCircleData:
    """Data to define the solar circle.

    Attributes
    ----------
    solar_radius : float
        The radius of the solar circle in kpc.
    omega : float
        The circular orbital frequency along the circle in radians/Myr.

    """

    solar_radius: float
    omega: float

    DATA_FILE_TYPE: ClassVar[str] = "Snapshot"
    VERSION: ClassVar[Version] = Version("0.1.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        if self.solar_radius <= 0:
            msg = f"Expected solar radius to be positive but got {self.solar_radius} kpc."
            raise ValueError(msg)

    def dump(self, path: Path) -> None:
        """Serialize data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        cls = type(self)
        with Hdf5File(path, "w") as file:
            # General
            file.attrs["type"] = cls.DATA_FILE_TYPE
            file.attrs["version"] = str(cls.VERSION)
            file.attrs["solar_radius"] = self.solar_radius
            file.attrs["omega"] = self.omega

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deerialize data from disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)
            solar_radius: float = get_float_attr_from_hdf5(file, "solar_radius")
            omega: float = get_float_attr_from_hdf5(file, "omega")

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            solar_radius=solar_radius,
            omega=omega,
        )
