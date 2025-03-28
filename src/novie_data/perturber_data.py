"""The animation data class for side on view projections."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, TypeAlias

import numpy as np
from h5py import File as Hdf5File
from packaging.version import Version

from novie_data._type_utils import Array1D, Array2D, verify_array_is_1d, verify_array_is_2d
from novie_data.errors import verify_arrays_are_consistent, verify_arrays_have_correct_length

from .serde.accessors import get_file_version, get_str_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


_Array1D_f32: TypeAlias = Array1D[np.float32]
_Array2D_f32: TypeAlias = Array2D[np.float32]
_Array1D_b8: TypeAlias = Array1D[np.bool_]


class _PerturberDataConverter(Protocol):
    """Description of a `PerturberData` file converter function."""

    def __call__(self, file: Hdf5File, *, is_complete: bool) -> None: ...


class _PerturberDataVerifier(Protocol):
    """Description of a `PerturberData` file verifier function."""

    def __call__(self, file: Hdf5File) -> int: ...


class _PerturberDataLoader(Protocol):
    """Description of a `PerturberData` file loader function."""

    def __call__(self, file: Hdf5File) -> PerturberData: ...


LATEST_VERSION_V2: Version = Version("2.0.0")
LATEST_VERSION_V3: Version = Version("3.0.0")

log: logging.Logger = logging.getLogger(__name__)


class PerturberData:
    """The position, velocity and mass of a single perturber.

    Attributes
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

    DATA_FILE_TYPE: ClassVar[str] = "Perturber"
    VERSION: ClassVar[Version] = LATEST_VERSION_V3

    def __init__(
        self,
        *,
        name: str,
        position: _Array2D_f32,
        velocity: _Array2D_f32,
        mass: _Array1D_f32,
        complete: _Array1D_b8 | bool = False,
    ) -> None:
        """Perform post-initialisation verification.

        Parameters
        ----------
        name : str
            The name of the dataset.
        position : Array2D[f32]
            The 3D position at every frame in kpc.
        velocity : Array2D[f32]
            The 3D velocity at every frame in km/s.
        mass : Array1D[f32]
            The mass at every frame in Msol.
        complete : Array1D[b8] | bool
            Set this flag to signify that the given data is complete or give an array of boolean values.

        """
        num_frames: int = len(mass)
        completeness: _Array1D_b8
        match complete:
            case True:
                completeness = np.ones(num_frames, dtype=np.bool_)
            case False:
                completeness = np.zeros(num_frames, dtype=np.bool_)
            case _:
                completeness = complete

        # Validate projection
        verify_arrays_have_correct_length(
            [(position, 0), (velocity, 0)], 3, msg="Expected the position/velocity vectors to have 3 rows."
        )
        verify_arrays_are_consistent(
            [(position, 1), (velocity, 1), (mass, 0), (completeness, 0)],
            msg="Expected position, velocity and mass to have the same number of frames!",
        )

        self.name: str = name
        self.position: _Array2D_f32 = position
        self.velocity: _Array2D_f32 = velocity
        self.mass: _Array1D_f32 = mass
        self.completeness: _Array1D_b8 = completeness

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
        equality &= np.array_equal(self.position, other.position)
        equality &= np.array_equal(self.velocity, other.velocity)
        equality &= np.array_equal(self.mass, other.mass)
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

            # v0
            file_version: Version = get_file_version(file)
            complete: _Array1D_b8 | bool
            if file_version.major == 2:
                log.debug("Loading %s v0", cls.__name__)
                complete = True
            else:
                verify_file_version_from_hdf5(file, cls.VERSION)
                complete = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "completeness", dtype=np.bool_))

            name: str = get_str_attr_from_hdf5(file, "name")

            # Projections
            position = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "position", dtype=np.float32))
            velocity = verify_array_is_2d(read_dataset_from_hdf5_with_dtype(file, "velocity", dtype=np.float32))
            mass = verify_array_is_1d(read_dataset_from_hdf5_with_dtype(file, "mass", dtype=np.float32))

        log.info("Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, path.absolute())
        return cls(
            position=position,
            velocity=velocity,
            mass=mass,
            name=name,
            complete=complete,
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
            file.create_dataset("completeness", data=self.completeness)

        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.position.shape[1]


_CONVERTERS: Mapping[tuple[int, int], _PerturberDataConverter] = {}
_VERIFIERS: Mapping[int, _PerturberDataVerifier] = {}
_LOADERS: Mapping[int, _PerturberDataLoader] = {}
