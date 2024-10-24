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
    output_path = tmp_path / "SolarCircle.hdf5"
    s = SolarCircleData(name="test", omega=0.0, solar_radius=1.0)
    s.dump(output_path)
    t = SolarCircleData.load(output_path)
    assert s == t
