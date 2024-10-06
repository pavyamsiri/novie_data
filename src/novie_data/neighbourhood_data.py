"""Dataclasses that represent neighbourhoods."""

import logging
from dataclasses import dataclass
from typing import Self

from h5py import File as Hdf5File

from .serde.accessors import get_float_attr_from_hdf5, get_int_attr_from_hdf5

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class SphericalNeighbourhoodData:
    """Data to define spherical solar neighbourhoods.

    Attributes
    ----------
    num_spheres : int
        The number of spheres around the solar circle.
    radius : float
        The radius of each neighbourhood.

    """

    num_spheres: int
    radius: float

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        if self.num_spheres <= 0:
            msg = f"Expected the number of spheres to be positive but got {self.num_spheres}."
            raise ValueError(msg)
        if self.radius <= 0:
            msg = f"Expected the radius to be positive but got {self.radius} kpc."

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.attrs["num_spheres"] = self.num_spheres
        out_file.attrs["sphere_radius"] = self.radius
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", type(self).__name__, out_file.filename)

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize snapshot data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        num_spheres: int = get_int_attr_from_hdf5(in_file, "num_spheres")
        sphere_radius: float = get_float_attr_from_hdf5(in_file, "sphere_radius")

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]",
            cls.__name__,
            in_file.filename,
        )
        return cls(
            num_spheres=num_spheres,
            radius=sphere_radius,
        )
