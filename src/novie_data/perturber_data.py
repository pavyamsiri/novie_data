"""The animation data class for side on view projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

from h5py import File as Hdf5File
from packaging.version import Version

from .serde.accessors import read_dataset_from_hdf5
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import float32
    from numpy.typing import NDArray


log: logging.Logger = logging.getLogger(__name__)


@dataclass
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

    """

    position: NDArray[float32]
    velocity: NDArray[float32]
    mass: NDArray[float32]
    snapshot_data: SnapshotData

    DATA_FILE_TYPE: ClassVar[str] = "Perturber"
    VERSION: ClassVar[Version] = Version("1.0.0")

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Validate projection
        num_frames: int = self.snapshot_data.num_frames
        if self.position.shape[0] != 3 or self.position.shape[1] != num_frames:
            msg = f"Expected the position vector to have the shape (3, {num_frames}) but got {self.position.shape}."
            raise ValueError(msg)
        if self.velocity.shape[0] != 3 or self.velocity.shape[1] != num_frames:
            msg = f"Expected the velocity vector to have the shape (3, {num_frames}) but got {self.velocity.shape}."
            raise ValueError(msg)
        if self.mass.shape[0] != num_frames:
            msg = f"Expected the mass array to have the shape ({num_frames}) but got {self.mass.shape}."
            raise ValueError(msg)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize surface density data from file.

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

            # Projections
            position: NDArray[float32]
            position = read_dataset_from_hdf5(file, "position")
            velocity: NDArray[float32]
            velocity = read_dataset_from_hdf5(file, "velocity")
            mass: NDArray[float32]
            mass = read_dataset_from_hdf5(file, "mass")
            snapshot_data = SnapshotData.load_from(file)

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            position=position,
            velocity=velocity,
            mass=mass,
            snapshot_data=snapshot_data,
        )

    def dump(self, path: Path) -> None:
        """Serialize surface density data to disk.

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

            file.create_dataset("position", data=self.position)
            file.create_dataset("velocity", data=self.velocity)
            file.create_dataset("mass", data=self.mass)
            self.snapshot_data.dump_into(file)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.snapshot_data.num_frames
