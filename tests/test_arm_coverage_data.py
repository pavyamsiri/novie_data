"""Test arm coverage data."""

from pathlib import Path

import numpy as np

from novie_data.arm_coverage_data import SpiralArmCoverageData
from novie_data.interface import NovieData


def test_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SpiralArmCoverageData, NovieData)


def test_init() -> None:
    """Test the constructor."""
    num_neighbourhoods: int = 7
    num_arms: int = 4
    num_frames: int = 20

    num_covered_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    num_total_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    covered_arm_normalised_densities = np.full((num_neighbourhoods, num_arms, num_frames), np.nan, dtype=np.float64)
    arm_names = tuple(f"arm {num}" for num in range(num_arms))

    s = SpiralArmCoverageData(
        name="test",
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        arm_names=arm_names,
    )
    assert len(s.arm_names) == num_arms


def test_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    num_neighbourhoods: int = 7
    num_arms: int = 4
    num_frames: int = 20

    num_covered_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    num_total_arm_pixels = np.full((num_neighbourhoods, num_arms, num_frames), 0, dtype=np.uint32)
    covered_arm_normalised_densities = np.full((num_neighbourhoods, num_arms, num_frames), 4.45, dtype=np.float64)
    arm_names = tuple(f"arm {num}" for num in range(num_arms))

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


def test_deserialization_v2() -> None:
    """Test deserialization of v2."""
    num_neighbourhoods: int = 7
    num_arms: int = 4
    num_frames: int = 20

    input_path = Path("test_data/arm_coverage_v2.hdf5")
    t = SpiralArmCoverageData.load(input_path)

    assert t.num_covered_arm_pixels.shape == (num_neighbourhoods, num_arms, num_frames)


def test_incremental_serde(tmp_path: Path) -> None:
    """Test incremental serialization per frame.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"

    num_frames = 10
    num_arms = 3
    num_locations = 6

    s = SpiralArmCoverageData.empty(
        num_frames=num_frames,
        num_arms=num_arms,
        num_locations=num_locations,
    )
    s.dump(output_path)

    s = SpiralArmCoverageData.load(output_path)
    assert s.name == "UNKNOWN"
    assert s.num_frames == 10
    assert np.all(~s.completeness)

    arm_names = tuple(f"arm {i}" for i in range(num_arms))
    SpiralArmCoverageData.save_init(
        output_path,
        arm_names=arm_names,
        omega=23.0,
        name="test",
    )
    s = SpiralArmCoverageData.load(output_path)
    assert s.arm_names == arm_names
    assert s.name == "test"
    assert s.num_frames == 10
    assert s.omega == 23.0
    assert np.all(~s.completeness)

    num_covered_arm_pixels = np.full((num_locations, num_arms), 34, dtype=np.uint32)
    num_total_arm_pixels = np.full((num_locations, num_arms), 63, dtype=np.uint32)
    covered_arm_normalised_densities = np.full((num_locations, num_arms), 90871.233, dtype=np.float64)
    SpiralArmCoverageData.save_frame(
        output_path,
        0,
        covered_arm_normalised_densities=covered_arm_normalised_densities,
        num_covered_arm_pixels=num_covered_arm_pixels,
        num_total_arm_pixels=num_total_arm_pixels,
    )
    s = SpiralArmCoverageData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    assert np.array_equal(s.num_covered_arm_pixels[:, :, 0], num_covered_arm_pixels)
    assert np.array_equal(s.num_total_arm_pixels[:, :, 0], num_total_arm_pixels)
    assert np.array_equal(s.covered_arm_normalised_densities[:, :, 0], covered_arm_normalised_densities)

    for i in range(1, 5):
        current_time = 0.74 * i + 598
        num_covered_arm_pixels = np.full((num_locations, num_arms), current_time + 34, dtype=np.uint32)
        num_total_arm_pixels = np.full((num_locations, num_arms), 411.0 * current_time - 123.0, dtype=np.uint32)
        covered_arm_normalised_densities = np.full((num_locations, num_arms), 1233.0 * current_time, dtype=np.float64)
        SpiralArmCoverageData.save_frame(
            output_path,
            i,
            covered_arm_normalised_densities=covered_arm_normalised_densities,
            num_covered_arm_pixels=num_covered_arm_pixels,
            num_total_arm_pixels=num_total_arm_pixels,
        )
    s = SpiralArmCoverageData.load(output_path)
    assert s.name == "test"
    assert s.num_frames == 10
    for i in range(1, 5):
        current_time = 0.74 * i + 598
        num_covered_arm_pixels = np.full((num_locations, num_arms), current_time + 34, dtype=np.uint32)
        num_total_arm_pixels = np.full((num_locations, num_arms), 411.0 * current_time - 123.0, dtype=np.uint32)
        covered_arm_normalised_densities = np.full((num_locations, num_arms), 1233.0 * current_time, dtype=np.float64)

        assert np.array_equal(s.num_covered_arm_pixels[:, :, i], num_covered_arm_pixels)
        assert np.array_equal(s.num_total_arm_pixels[:, :, i], num_total_arm_pixels)
        assert np.array_equal(s.covered_arm_normalised_densities[:, :, i], covered_arm_normalised_densities)
        assert s.completeness[i]
