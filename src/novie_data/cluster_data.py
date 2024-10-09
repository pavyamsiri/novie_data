"""Data representing clusters found by the SpArcFiRe algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np
from h5py import File as Hdf5File
from numpy import float32, int8, int16
from packaging.version import Version

from .serde.accessors import get_float_attr_from_hdf5, read_dataset_from_hdf5_with_dtype
from .serde.verification import verify_file_type_from_hdf5, verify_file_version_from_hdf5
from .snapshot_data import SnapshotData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


NUM_POINTS: int = 100
log: logging.Logger = logging.getLogger(__name__)


@dataclass
class GlobalSpiralData:
    """Spiral arm global to a galaxy.

    Attributes
    ----------
    overall_pitch_angles : NDArray[float]
        The overall pitch angle of a galaxy in degrees.
    winding_directions : NDArray[int8]
        The global winding direction or chirality for every frame.
        The only valid values are:
            -1 => counter-clockwise
             0 => no chirality
            +1 => clockwise

    """

    overall_pitch_angles: NDArray[float32]
    winding_directions: NDArray[int8]

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the arrays are the same size
        same_shape = self.overall_pitch_angles.shape == self.winding_directions.shape
        if not same_shape:
            pitch_angle_shape = self.overall_pitch_angles.shape
            winding_shape = self.winding_directions.shape
            msg = f"The pitch angle {pitch_angle_shape} and winding direction {winding_shape} array shapes are different!"
            raise ValueError(msg)
        # Verify that the winding directions are in the valid range
        is_valid = np.all(np.isin(self.winding_directions, [-1, 0, 1]))
        if not is_valid:
            msg = f"There are invalid chirality values! The unique elements are {np.unique(self.winding_directions)}"
            raise ValueError(msg)

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize snapshot data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        overall_pitch_angles = read_dataset_from_hdf5_with_dtype(in_file, "overall_pitch_angles", dtype=float32)
        winding_directions = read_dataset_from_hdf5_with_dtype(in_file, "winding_directions", dtype=int8)

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(overall_pitch_angles=overall_pitch_angles, winding_directions=winding_directions)

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.create_dataset("overall_pitch_angles", data=self.overall_pitch_angles)
        out_file.create_dataset("winding_directions", data=self.winding_directions)
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )


@dataclass
class ClusterData:
    """Local spiral arm data representing the fitted log spirals.

    Attributes
    ----------
    arc_bounds : NDArray[float32]
        The azimuthal bounds of each arc in radians.
    offset : NDArray[float32]
        The offset parameter of each arc in radians.
    growth_factor : NDArray[float32]
        The growth factor parameter of each arc.
    initial_radius : NDArray[float32]
        The initial radius parameter of each arc in physical units.
    is_two_revolution : NDArray[float32]
        The multi-revolutionness of each arc.
    errors : NDArray[float32]
        The fit errors of each cluster's fitted log spiral.
    num_clusters : NDArray[int16]
        The number of clusters found per frame.
    max_clusters : int
        The maximum number of clusters supported in a single frame.
    num_frames : int
        The number of frames.

    """

    arc_bounds: NDArray[float32]
    offset: NDArray[float32]
    growth_factor: NDArray[float32]
    initial_radius: NDArray[float32]
    is_two_revolution: NDArray[np.bool_]
    errors: NDArray[float32]
    num_clusters: NDArray[int16]

    def __post_init__(self) -> None:
        """Perform post-initialisation verification."""
        # Verify that the arrays the correct size
        if len(self.offset.shape) != 2:
            msg = f"Expected the offset array to be 2D but got {self.offset.shape}"
            raise ValueError(msg)
        num_frames: int = self.offset.shape[1]
        max_clusters: int = int(np.max(self.num_clusters))
        # Cut down the arrays
        self.offset = self.offset[:max_clusters, :]
        self.growth_factor = self.growth_factor[:max_clusters, :]
        self.initial_radius = self.initial_radius[:max_clusters, :]
        self.is_two_revolution = self.is_two_revolution[:max_clusters, :]
        self.errors = self.errors[:max_clusters, :]
        self.arc_bounds = self.arc_bounds[:, :max_clusters, :]

        common_shape: tuple[int, int] = (max_clusters, num_frames)
        if self.offset.shape != common_shape:
            msg = f"Expected the offset array to be 2D with shape {(max_clusters, num_frames)} but got {self.offset.shape}"
            raise ValueError(msg)
        if self.growth_factor.shape != common_shape:
            msg = f"Expected the growth_factor array to be 2D with shape {common_shape} but got {self.growth_factor.shape}"
            raise ValueError(msg)
        if self.initial_radius.shape != common_shape:
            msg = f"Expected the initial_radius array to be 2D with shape {common_shape} but got {self.initial_radius.shape}"
            raise ValueError(msg)
        if self.is_two_revolution.shape != common_shape:
            msg = (
                f"Expected the is_two_revolution array to be 2D with shape {common_shape} but got {self.is_two_revolution.shape}"
            )
            raise ValueError(msg)
        if self.errors.shape != common_shape:
            msg = f"Expected the errors array to be 2D with shape {common_shape} but got {self.is_two_revolution.shape}"
            raise ValueError(msg)
        expected_arc_bounds_shape = (2, max_clusters, num_frames)
        if self.arc_bounds.shape != expected_arc_bounds_shape:
            msg = f"Expected the arc bounds array to be 3D with shape {expected_arc_bounds_shape} but got {self.arc_bounds.shape}"
            raise ValueError(msg)
        self.num_frames: int = num_frames
        self.max_clusters = max_clusters

    @classmethod
    def load_from(cls, in_file: Hdf5File) -> Self:
        """Serialize snapshot data from file.

        Parameters
        ----------
        in_file : Hdf5File
            The HDF5 file to read from.

        """
        arc_bounds = read_dataset_from_hdf5_with_dtype(in_file, "arc_bounds", dtype=float32)
        offset = read_dataset_from_hdf5_with_dtype(in_file, "offset", dtype=float32)
        growth_factor = read_dataset_from_hdf5_with_dtype(in_file, "growth_factor", dtype=float32)
        initial_radius = read_dataset_from_hdf5_with_dtype(in_file, "initial_radius", dtype=float32)
        cluster_fit_errors = read_dataset_from_hdf5_with_dtype(in_file, "cluster_fit_errors", dtype=float32)
        is_two_revolution = read_dataset_from_hdf5_with_dtype(in_file, "is_two_revolution", dtype=np.bool_)
        num_clusters = read_dataset_from_hdf5_with_dtype(in_file, "num_clusters", dtype=int16)

        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]", cls.__name__, Path(in_file.filename).absolute()
        )
        return cls(
            arc_bounds=arc_bounds,
            offset=offset,
            growth_factor=growth_factor,
            errors=cluster_fit_errors,
            initial_radius=initial_radius,
            is_two_revolution=is_two_revolution,
            num_clusters=num_clusters,
        )

    def dump_into(self, out_file: Hdf5File) -> None:
        """Deserialize snapshot data to file.

        Parameters
        ----------
        out_file : Hdf5File
            The HDF5 file to write to.

        """
        # General
        out_file.create_dataset("arc_bounds", data=self.arc_bounds)
        out_file.create_dataset("offset", data=self.offset)
        out_file.create_dataset("growth_factor", data=self.growth_factor)
        out_file.create_dataset("initial_radius", data=self.initial_radius)
        out_file.create_dataset("cluster_fit_errors", data=self.errors)
        out_file.create_dataset("is_two_revolution", data=self.is_two_revolution)
        out_file.create_dataset("num_clusters", data=self.num_clusters)
        log.info(
            "Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]",
            type(self).__name__,
            Path(out_file.filename).absolute(),
        )


@dataclass
class SpiralClusterData:
    """Data class to store residuals and errors from corrugation data processing.

    Attributes
    ----------
    cluster_masks : NDArray[int16]
        The cluster masks. A cell is given the index of the cluster it belongs to,
        -1 indicating it does not belong to any cluster. The shape is (num_bins, num_bins, 4, num_frames).
    pixel_to_distance : float
        The unit conversion factor to go from pixels to physical units.
    global_spiral_data : GlobalSpiralData
        Global spiral arm properties.
    cluster_data : ClusterData
        The data pertaining to each found cluster.
    spiral_arm_error_data : SpiralArmErrorData
        The error between observed arms and found clusters.
    snapshot_data : SnapshotData
        The data describing each snapshot.

    """

    cluster_masks: NDArray[int16]
    # Unit conversion
    pixel_to_distance: float
    # Global properties
    global_spiral_data: GlobalSpiralData
    # Cluster data
    cluster_data: ClusterData
    # Arm data
    snapshot_data: SnapshotData

    DATA_FILE_TYPE: ClassVar[str] = "SpiralClusters"
    VERSION: ClassVar[Version] = Version("2.0.0")
    MAX_CLUSTERS: int = 256

    @classmethod
    def load(cls, path: Path) -> Self:
        """Deserialize phase spiral data from file.

        Parameters
        ----------
        path : Path
            The path to the data.

        Returns
        -------
        SpiralClusterData
            The deserialized data.

        """
        with Hdf5File(path, "r") as file:
            verify_file_type_from_hdf5(file, cls.DATA_FILE_TYPE)
            verify_file_version_from_hdf5(file, cls.VERSION)

            # Arrays
            cluster_masks = read_dataset_from_hdf5_with_dtype(file, "cluster_masks", dtype=int16)
            pixel_to_distance: float = get_float_attr_from_hdf5(file, "pixel_to_distance")

            snapshot_data = SnapshotData.load_from(file)
            global_spiral_data = GlobalSpiralData.load_from(file)
            cluster_data = ClusterData.load_from(file)
        log.info(
            "Successfully loaded [cyan]%s[/cyan] from [magenta]%s[/magenta]",
            cls.__name__,
            path.absolute(),
        )
        return cls(
            cluster_masks=cluster_masks,
            pixel_to_distance=pixel_to_distance,
            global_spiral_data=global_spiral_data,
            cluster_data=cluster_data,
            snapshot_data=snapshot_data,
        )

    def dump(self, path: Path) -> None:
        """Serialize phase spiral data to disk.

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
            file.attrs["pixel_to_distance"] = float(self.pixel_to_distance)

            file.create_dataset("cluster_masks", data=self.cluster_masks)
            self.snapshot_data.dump_into(file)
            self.global_spiral_data.dump_into(file)
            self.cluster_data.dump_into(file)
        log.info("Successfully dumped [cyan]%s[/cyan] to [magenta]%s[/magenta]", cls.__name__, path.absolute())

    # Convenience functions

    @property
    def num_frames(self) -> int:
        """int: The number of frames."""
        return self.snapshot_data.num_frames


@dataclass
class GlobalSpiralFrame:
    """Spiral arm global to a galaxy.

    Attributes
    ----------
    overall_pitch_angles : float
        The overall pitch angle of a galaxy in degrees.
    winding_directions : Literal[-1, 0, 1]
        The global winding direction or chirality for every frame.
        The only valid values are:
            -1 => counter-clockwise
             0 => no chirality
            +1 => clockwise

    """

    overall_pitch_angle: float
    winding_direction: Literal[-1, 0, 1]

    @staticmethod
    def create_empty_frame() -> GlobalSpiralFrame:
        """Create the default frame data.

        Returns
        -------
        default_frame : GlobalSpiralFrame
            The default frame.

        """
        return GlobalSpiralFrame(
            overall_pitch_angle=np.nan,
            winding_direction=0,
        )

    @staticmethod
    def combine(frames: Sequence[GlobalSpiralFrame]) -> GlobalSpiralData:
        """Combine multiple frames.

        Parameters
        ----------
        frames : Sequence[GlobalSpiralFrame]
            The frames to combine.

        Returns
        -------
        combined_data : GlobalSpiralData
            The combined frame data.

        """
        overall_pitch_angles: NDArray[float32] = np.asarray([frame.overall_pitch_angle for frame in frames], dtype=float32)
        winding_directions: NDArray[int8] = np.asarray([frame.winding_direction for frame in frames], dtype=int8)
        return GlobalSpiralData(
            overall_pitch_angles=overall_pitch_angles,
            winding_directions=winding_directions,
        )


@dataclass
class ClusterFrame:
    """Local spiral arm data representing the fitted log spirals.

    Attributes
    ----------
    arc_bounds : NDArray[float32]
        The azimuthal bounds of each arc in radians.
    offset : NDArray[float32]
        The offset parameter of each arc in radians.
    growth_factor : NDArray[float32]
        The growth factor parameter of each arc.
    initial_radius : NDArray[float32]
        The initial radius parameter of each arc in physical units.
    is_two_revolution : NDArray[bool]
        The multi-revolutionness of each arc.
    errors : NDArray[float32]
        The fit errors for every cluster.
    num_clusters : int
        The number of valid clusters in the frame.

    """

    arc_bounds: NDArray[float32]
    offset: NDArray[float32]
    growth_factor: NDArray[float32]
    initial_radius: NDArray[float32]
    is_two_revolution: NDArray[np.bool_]
    errors: NDArray[float32]
    num_clusters: int

    @staticmethod
    def create_empty_frame(max_clusters: int) -> ClusterFrame:
        """Create an empty frame.

        Parameters
        ----------
        max_clusters : int
            The maximum number of clusters allowed per frame.

        Returns
        -------
        empty_frame : ClusterFrame
            An empty frame.

        """
        arc_bounds: NDArray[float32] = np.full((2, max_clusters), np.nan, dtype=float32)
        offset: NDArray[float32] = np.full(max_clusters, np.nan, dtype=float32)
        growth_factor: NDArray[float32] = np.full(max_clusters, np.nan, dtype=float32)
        initial_radius: NDArray[float32] = np.full(max_clusters, np.nan, dtype=float32)
        is_two_revolution: NDArray[np.bool_] = np.zeros(max_clusters, dtype=np.bool_)
        cluster_fit_errors: NDArray[float32] = np.full(max_clusters, np.nan, dtype=float32)
        return ClusterFrame(
            arc_bounds=arc_bounds,
            offset=offset,
            growth_factor=growth_factor,
            initial_radius=initial_radius,
            errors=cluster_fit_errors,
            is_two_revolution=is_two_revolution,
            num_clusters=0,
        )

    @staticmethod
    def combine(frames: Sequence[ClusterFrame]) -> ClusterData:
        """Combine multiple frames.

        Parameters
        ----------
        frames : Sequence[ClusterFrame]
            The frames to combine.

        Returns
        -------
        combined_data : ClusterData
            The combined frame data.

        """
        arc_bounds: NDArray[float32] = np.stack([current_frame.arc_bounds for current_frame in frames], axis=-1, dtype=float32)
        offset: NDArray[float32] = np.stack([current_frame.offset for current_frame in frames], axis=-1, dtype=float32)
        growth_factor: NDArray[float32] = np.stack(
            [current_frame.growth_factor for current_frame in frames], axis=-1, dtype=float32
        )
        initial_radius: NDArray[float32] = np.stack(
            [current_frame.initial_radius for current_frame in frames], axis=-1, dtype=float32
        )
        is_two_revolution: NDArray[np.bool_] = np.stack(
            [current_frame.is_two_revolution for current_frame in frames], axis=-1, dtype=np.bool_
        )
        errors: NDArray[float32] = np.stack([current_frame.errors for current_frame in frames], axis=-1, dtype=float32)
        num_clusters: NDArray[int16] = np.stack([current_frame.num_clusters for current_frame in frames], axis=-1, dtype=int16)
        return ClusterData(
            arc_bounds=arc_bounds,
            offset=offset,
            growth_factor=growth_factor,
            initial_radius=initial_radius,
            is_two_revolution=is_two_revolution,
            errors=errors,
            num_clusters=num_clusters,
        )


@dataclass
class ProcessedFrameData:
    """The processed data of each frame."""

    mask: NDArray[int16]
    global_spiral_data: GlobalSpiralFrame
    cluster_data: ClusterFrame

    @staticmethod
    def create_empty_frame(
        num_rows: int,
        num_columns: int,
        *,
        max_clusters: int,
    ) -> ProcessedFrameData:
        """Create a frame with no clusters.

        Parameters
        ----------
        num_rows : int
            The number of rows in the input array.
        num_columns : int
            The number of columns in the input array.
        max_clusters : int
            The maximum amount of clusters allowed to be registered.

        Returns
        -------
        empty_frame : ProcessedFrameData
            The empty frame.

        """
        shape = (num_rows, num_columns)
        mask: NDArray[int16] = np.full(shape, -1, dtype=int16)
        global_spiral_data: GlobalSpiralFrame = GlobalSpiralFrame.create_empty_frame()
        cluster_data: ClusterFrame = ClusterFrame.create_empty_frame(max_clusters)

        return ProcessedFrameData(
            mask=mask,
            global_spiral_data=global_spiral_data,
            cluster_data=cluster_data,
        )

    @staticmethod
    def combine(
        frames: Sequence[ProcessedFrameData], snapshot_data: SnapshotData, *, pixel_to_distance: float
    ) -> SpiralClusterData:
        """Combine processed frames.

        Parameters
        ----------
        frames : Sequence[ProcessedFrameData]
            The frames to combine.
        snapshot_data : SnapshotData
            The snapshot frame data.
        pixel_to_distance : float
            The unit conversion factor to go from pixels to physical units.

        Returns
        -------
        combined_data : SpiralClusterData
            The combined data.

        """
        cluster_masks: NDArray[int16] = np.stack([current_frame.mask for current_frame in frames], axis=-1)
        global_spiral_data: GlobalSpiralData = GlobalSpiralFrame.combine(
            [current_frame.global_spiral_data for current_frame in frames]
        )
        cluster_data: ClusterData = ClusterFrame.combine([current_frame.cluster_data for current_frame in frames])
        return SpiralClusterData(
            cluster_masks=cluster_masks,
            pixel_to_distance=pixel_to_distance,
            global_spiral_data=global_spiral_data,
            cluster_data=cluster_data,
            snapshot_data=snapshot_data,
        )
