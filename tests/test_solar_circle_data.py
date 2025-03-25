"""Test SolarCircle data."""

from pathlib import Path

import pytest

from novie_data.errors import NonPositiveValueError
from novie_data.interface import NovieData
from novie_data.solar_circle_data import SolarCircleData


def test_solar_circle_data_protocol() -> None:
    """Test that the protocol is adhered to."""
    assert isinstance(SolarCircleData, NovieData)


def test_solar_circle_data_init() -> None:
    """Test the constructor."""
    s = SolarCircleData(name="test", omega=0.0, solar_radius=1.0)
    assert s.omega == 0.0


def test_solar_circle_data_init_negative_solar_radius() -> None:
    """Test the constructor raises an error on negative solar radii values."""
    with pytest.raises(NonPositiveValueError, match="solar radius"):
        _ = SolarCircleData(name="test", omega=0.0, solar_radius=-1.0)


def test_solar_circle_data_serde(tmp_path: Path) -> None:
    """Test serialization and deserialization.

    Parameters
    ----------
    tmp_path : Path
        The temporary directory to write to.

    """
    output_path = tmp_path / "test.hdf5"
    s = SolarCircleData(name="test", omega=0.0, solar_radius=1.0)
    s.dump(output_path)
    t = SolarCircleData.load(output_path)
    assert s == t


def test_solar_circle_data_deserialization_v0() -> None:
    """Test deserialization of v0."""
    solar_radius: float = 1.0
    omega: float = 0.0

    input_path = Path("test_data/solar_circle_v0.hdf5")
    t = SolarCircleData.load(input_path)

    assert t.solar_radius == solar_radius
    assert t.omega == omega
