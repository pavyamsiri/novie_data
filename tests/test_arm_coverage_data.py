"""Test arm coverage data."""

from pathlib import Path

import numpy as np

from novie_data.arm_coverage_data import SpiralArmCoverageData
from novie_data.interface import NovieData


def test_arm_coverage_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SpiralArmCoverageData, NovieData)


def test_arm_coverage_data_init() -> None:
    """Test the constructor."""
    num_neighbourhoods: int = 7
    num_arms: int = 4
    num_frames: int = 20

    num_covered_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    num_total_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    covered_arm_normalised_densities = np.full((num_neighbourhoods, num_arms, num_frames), np.nan, dtype=np.float32)
    arm_names = [f"arm {num}" for num in range(num_arms)]

    s = SpiralArmCoverageData(
        name="test",
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        arm_names=arm_names,
    )
    assert len(s.arm_names) == num_arms


def test_arm_coverage_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "snapshot.hdf5"
    num_neighbourhoods: int = 7
    num_arms: int = 4
    num_frames: int = 20

    num_covered_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    num_total_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    covered_arm_normalised_densities = np.full((num_neighbourhoods, num_arms, num_frames), np.nan, dtype=np.float32)
    arm_names = [f"arm {num}" for num in range(num_arms)]

    s = SpiralArmCoverageData(
        name="test",
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        arm_names=arm_names,
    )
    s.dump(output_path)
    t = SpiralArmCoverageData.load(output_path)
    assert s == t
