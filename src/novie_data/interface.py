"""Definition of the NovieData protocol."""

from pathlib import Path
from typing import ClassVar, Protocol, Self

from packaging.version import Version


class NovieData(Protocol):
    """Definition of novie data that can be directly read and written to disk."""

    DATA_FILE_TYPE: ClassVar[str]
    VERSION: ClassVar[Version]

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        NovieData
            The deserialized data.

        """
        ...

    def dump(self, path: Path) -> None:
        """Serialize data to disk.

        Parameters
        ----------
        path : Path
            The path to the data.

        """
        ...
