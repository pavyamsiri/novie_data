"""Data to define the solar circle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from .serde.accessors import get_float_attr_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Sequence

    from h5py import File as Hdf5File

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

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        if self.solar_radius <= 0:
            msg = f"Expected solar radius to be positive but got {self.solar_radius} kpc."
            raise ValueError(msg)

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["solar_radius"] = self.solar_radius
        out_file.attrs["omega"] = self.omega
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", type(self).__name__, out_file.filename)

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize snapshot data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        solar_radius: float = get_float_attr_from_hdf5(in_file, "solar_radius")
        omega: float = get_float_attr_from_hdf5(in_file, "omega")

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, in_file.filename)
        return cls(
            solar_radius=solar_radius,
            omega=omega,
        )

    @staticmethod
    def all_same(data_list: Sequence[SolarCircleData]) -> bool:
        """Return whether the list of solar circle data are all the same.

        Parameters
        ----------
        data_list : Sequence[SolarCircleData]
            The list of solar circle data.

        Returns
        -------
        all_same : bool
            This is `True` if all solar circle data are equal otherwise `False`.

        """
        if len(data_list) == 0:
            return True
        reference_data = data_list[0]
        return all(current_data == reference_data for current_data in data_list)
