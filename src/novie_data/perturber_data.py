"""The animation data class for side on view projections."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32
from packaging.version import Version

from novie_data.errors import verify_arrays_are_consistent, verify_arrays_have_correct_length

from .serde.accessors import get_str_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


class PerturberData:
    """The position, velocity and mass of a single perturber.

    Attributes
    ----------
    position : NDArray[float]
        The 3D position at every frame in kpc.
    velocity : NDArray[float]
        The 3D velocity at every frame in km/s.
    mass : NDArray[float]
        The mass at every frame in Msol.
    name : str
        The name of the dataset.

    """

    DATA_FILE_TYPE: ClassVar[str] = "Perturber"
    VERSION: ClassVar[Version] = Version("2.0.0")

    def __init__(self, *, name: str, position: NDArray[float32], velocity: NDArray[float32], mass: NDArray[float32]) -> None:
        """Perform post-initialisation verification.

        Parameters
        ----------
        name : str
            The name of the dataset.
        position : NDArray[float]
            The 3D position at every frame in kpc.
        velocity : NDArray[float]
            The 3D velocity at every frame in km/s.
        mass : NDArray[float]
            The mass at every frame in Msol.

        """
        self.name: str = name
        self.position: NDArray[float32] = position
        self.velocity: NDArray[float32] = velocity
        self.mass: NDArray[float32] = mass

        # Validate projection
        verify_arrays_have_correct_length(
            [(self.position, 0), (self.velocity, 0)], 3, msg="Expected the position/velocity vectors to have 3 rows."
        )
        verify_arrays_are_consistent(
            [(self.position, 1), (self.velocity, 1), (self.mass, 0)],
            msg="Expected position, velocity and mass to have the same number of frames!",
        )

    def __eq__(self, other: object, /) -> bool:
        """Compare for equality.

        Parameters
        ----------
        other : object
            The object to compare to.

        Returns
        -------
        bool
            `True` if the other object is equal to this object, `False` otherwise.

        Notes
        -----
        Equality means all fields are equal.

        """
        if not isinstance(other, type(self)):
            return False
        equality = True
        equality &= self.name == other.name
        equality &= np.all(self.position == other.position)
        equality &= np.all(self.velocity == other.velocity)
        equality &= np.all(self.mass == other.mass)
        return bool(equality)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize data from disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        PerturberData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            name: str = get_str_attr_from_hdf5(file, "name")

            # Projections
            position = read_dataset_from_hdf5_with_dtype(file, "position", dtype=float32)
            velocity = read_dataset_from_hdf5_with_dtype(file, "velocity", dtype=float32)
            mass = read_dataset_from_hdf5_with_dtype(file, "mass", dtype=float32)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            position=position,
            velocity=velocity,
            mass=mass,
            name=name,
        )

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
            file.attrs["name"] = self.name

            file.create_dataset("position", data=self.position)
            file.create_dataset("velocity", data=self.velocity)
            file.create_dataset("mass", data=self.mass)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.position.shape[1]
