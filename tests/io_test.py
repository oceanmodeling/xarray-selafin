import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.spatial import Delaunay

TIDAL_FLATS = pytest.mark.parametrize(
    "slf_in",
    [
        pytest.param("tests/data/r3d_tidal_flats.slf", id="3D"),
        pytest.param("tests/data/r2d_tidal_flats.slf", id="2D"),
    ],
)

DIMS = pytest.mark.parametrize(
    "slf_in",
    [
        pytest.param("tests/data/r3d_tidal_flats.slf", id="3D"),
        pytest.param("tests/data/r2d_tidal_flats.slf", id="2D"),
        pytest.param("tests/data/r1d_tomsail.slf", id="1D"),
    ],
)


def write_netcdf(ds, nc_out):
    # Remove dict and multi-dimensional arrays not supported in netCDF
    del ds.attrs["variables"]
    del ds["face"]
    try:
        del ds.attrs["volume"]
    except KeyError:
        pass
    # Write netCDF file
    ds.to_netcdf(nc_out)


@TIDAL_FLATS
def test_open_dataset(slf_in):
    ds = xr.open_dataset(slf_in, engine="selafin")
    assert isinstance(ds, xr.Dataset)
    repr(ds)

    # Dimensions
    assert ds.sizes["time"] == 17
    assert ds.sizes["node"] == 648
    if "r3d" in slf_in:
        assert ds.sizes["plan"] == 21

    # Coordinates
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords

    # Attributes
    assert ds.attrs["endian"] == ">"
    assert ds.attrs["date_start"] == (1900, 1, 1, 0, 0, 0)
    assert "ipobo" in ds.attrs
    assert "face" in ds.variables
    if "r3d_" in slf_in:
        assert "volume" in ds.variables
    else:
        assert "volume" not in ds.variables


@TIDAL_FLATS
def test_to_netcdf(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test.nc"
    write_netcdf(ds_slf, nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slf)


@TIDAL_FLATS
def test_to_selafin(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, engine="selafin")

    # Remove some data which is rebuilt
    del ds_slf.attrs["date_start"]

    slf_out = tmp_path / "test.slf"
    ds_slf.selafin.write(slf_out)
    ds_slf2 = xr.open_dataset(slf_out, engine="selafin")
    assert ds_slf2.equals(ds_slf)

    # Compare binary files
    with open(slf_in, "rb") as in_slf1, open(slf_out, "rb") as in_slf2:
        assert in_slf1.read() == in_slf2.read()


@TIDAL_FLATS
def test_to_selafin_eager_mode(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, lazy_loading=False, engine="selafin")

    # Remove some data which is rebuilt
    del ds_slf.attrs["date_start"]

    slf_out = tmp_path / "test.slf"
    ds_slf.selafin.write(slf_out)
    ds_slf2 = xr.open_dataset(slf_out, engine="selafin")
    assert ds_slf2.equals(ds_slf)

    # Compare binary files
    with open(slf_in, "rb") as in_slf1, open(slf_out, "rb") as in_slf2:
        assert in_slf1.read() == in_slf2.read()


@TIDAL_FLATS
def test_slice(tmp_path, slf_in):
    # simple selection
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test1.nc"
    ds_slice = ds_slf.isel(time=0)
    write_netcdf(ds_slice, nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slice)
    # simple range
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test2.nc"
    ds_slice = ds_slf.isel(time=slice(0, 10))
    write_netcdf(ds_slice, nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slice)
    if "r3d" in slf_in:
        # multiple slices
        ds_slf = xr.open_dataset(slf_in, engine="selafin")
        nc_out = tmp_path / "test3.nc"
        ds_slice = ds_slf.isel(time=slice(0, 10), plan=0)
        write_netcdf(ds_slice, nc_out)
        ds_nc = xr.open_dataset(nc_out)
        assert ds_nc.equals(ds_slice)
        # multiple range slices
        ds_slf = xr.open_dataset(slf_in, engine="selafin")
        nc_out = tmp_path / "test4.nc"
        ds_slice = ds_slf.isel(time=slice(0, 10), plan=slice(5, 10))
        write_netcdf(ds_slice, nc_out)
        ds_nc = xr.open_dataset(nc_out)
        assert ds_nc.equals(ds_slice)


def test_from_scratch(tmp_path):
    slf_out = tmp_path / "test.slf"
    x = np.random.rand(100)
    y = np.random.rand(100)

    ikle = Delaunay(np.vstack((x, y)).T).simplices + 1  # IKLE tables are 1-indexed

    # Creating a minimal dataset
    ds = xr.Dataset(
        {
            "S": (("time", "node"), np.random.rand(10, 100)),
            # Add other variables as needed
        },
        coords={
            "x": ("node", x),
            "y": ("node", y),
            "time": pd.date_range("2020-01-01", periods=10),
            "face": (("face_node_connectivity", "face_dimension"), ikle),
        },
    )

    # Writing to a SELAFIN file
    ds.selafin.write(slf_out)


@DIMS
def test_dim(slf_in):
    ds = xr.open_dataset(slf_in, engine="selafin")
    repr(ds)
