"""Data representing residuals and errors between observed and simulated corrugation data."""

from __future__ import annotations

import logging
from typing import ClassVar

from packaging.version import Version

from .signal_residuals_data import SignalResidualsData

log: logging.Logger = logging.getLogger(__name__)


class CorrugationResidualsData(SignalResidualsData):
    """Data class to store residuals and errors from corrugation data processing.

    Attributes
    ----------
    name : str
        The name of the data set.
    metric_name : str
        The name of the metric used.
    metric : Array3D[f32]
        The value of the metric per point, frame and location.
    summary : Array2D[f32]
        The summary statistic of the metric per frame and location.
    bin_values : Array1D[f32]
        The central bin value for each bin.

    """

    DATA_FILE_TYPE: ClassVar[str] = "CorrugationResiduals"
    VERSION: ClassVar[Version] = Version("2.0.0")

    @classmethod
    def get_data_file_type(cls) -> str:
        """Return the data file type string for this class.

        Returns
        -------
        file_type : str
            The data file type.

        """
        return cls.DATA_FILE_TYPE

    @classmethod
    def get_version(cls) -> Version:
        """Return the version for this class.

        Returns
        -------
        version : Version
            The file version.

        """
        return cls.VERSION
