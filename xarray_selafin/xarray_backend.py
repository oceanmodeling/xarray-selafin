import os
from datetime import datetime
from datetime import timedelta
from operator import attrgetter

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.backends import BackendEntrypoint
from xarray.core import indexing

from xarray_selafin import Serafin


DEFAULT_DATE_START = (1900, 1, 1, 0, 0, 0)


def compute_duration_between_datetime(t0, time_serie):
    return (time_serie - t0).astype("timedelta64[s]").astype(float)


def read_serafin(f, lang):
    resin = Serafin.Read(f, lang)
    resin.__enter__()
    resin.read_header()
    resin.get_time()
    return resin


def write_serafin(fout, ds):
    # Title
    try:
        title = ds.attrs["title"]
    except KeyError:
        title = "Converted with array-serafin"

    slf_header = Serafin.SerafinHeader(title)

    # File precision
    try:
        float_size = ds.attrs["float_size"]
    except KeyError:
        float_size = 4  # Default: single precision
    if float_size == 4:
        slf_header.to_single_precision()
    elif float_size == 8:
        slf_header.to_double_precision()
    else:
        raise NotImplementedError

    try:
        slf_header.endian = ds.attrs["endian"]
    except KeyError:
        pass  # Default: ">"

    try:
        slf_header.nb_frames = ds.time.size
    except AttributeError:
        slf_header.nb_frames = 0

    try:
        slf_header.date = ds.attrs["date_start"]
    except KeyError:
        # Retrieve starting date from first time
        if slf_header.nb_frames == 0:
            first_time = ds.time
        else:
            first_time = ds.time[0]
        first_date_str = first_time.values.astype(str)  # "1900-01-01T00:00:00.000000000"
        first_date_str = first_date_str.rstrip("0") + "0"  # "1900-01-01T00:00:00.0"
        try:
            date = datetime.strptime(first_date_str, "%Y-%m-%dT%H:%M:%S.%f")
            slf_header.date = attrgetter("year", "month", "day", "hour", "minute", "second")(date)
        except ValueError:
            slf_header.date = DEFAULT_DATE_START

    # Variables
    try:
        slf_header.language = ds.attrs["language"]
    except KeyError:
        slf_header.language = Serafin.LANG
    for var in ds.data_vars:
        try:
            name, unit = ds.attrs["variables"][var]
            slf_header.add_variable_str(var, name, unit)
        except KeyError:
            try:
                slf_header.add_variable_from_ID(var)
            except Serafin.SerafinRequestError:
                slf_header.add_variable_str(var, var, "?")
    slf_header.nb_var = len(slf_header.var_IDs)

    if "plan" in ds.dims:  # 3D
        is_2d = False
        nplan = len(ds.plan)
        slf_header.nb_nodes_per_elem = 6
        slf_header.nb_elements = len(ds.attrs["ikle2"]) * (nplan - 1)
    else:  # 2D
        is_2d = True
        nplan = 1  # just to do a multiplication
        slf_header.nb_nodes_per_elem = ds.attrs["ikle2"].shape[1]
        slf_header.nb_elements = len(ds.attrs["ikle2"])

    slf_header.nb_nodes = ds.sizes["node"] * nplan
    slf_header.nb_nodes_2d = ds.sizes["node"]

    x = ds.coords["x"].values
    y = ds.coords["y"].values
    if not is_2d:
        x = np.tile(x, nplan)
        y = np.tile(y, nplan)
    slf_header.x = x
    slf_header.y = y
    slf_header.mesh_origin = (0, 0)  # Should be integers
    slf_header.x_stored = x - slf_header.mesh_origin[0]
    slf_header.y_stored = y - slf_header.mesh_origin[1]
    slf_header.ikle_2d = ds.attrs["ikle2"]
    if is_2d:
        slf_header.ikle = slf_header.ikle_2d.flatten()
    else:
        try:
            slf_header.ikle = ds.attrs["ikle3"]
        except KeyError:
            # Rebuild IKLE from 2D
            slf_header.ikle = slf_header.compute_ikle(len(ds.plan), slf_header.nb_nodes_per_elem)

    try:
        slf_header.ipobo = ds.attrs["ipobo"]
    except KeyError:
        # Rebuild IPOBO
        slf_header.build_ipobo()

    if "plan" in ds.dims:  # 3D
        slf_header.nb_planes = len(ds.plan)
        slf_header.is_2d = False
        shape = (slf_header.nb_var, slf_header.nb_planes, slf_header.nb_nodes_2d)
    else:  # 2D (converted if required)
        # if ds.attrs["type"] == "3D":
        #     slf_header.is_2d = False  # to enable conversion from 3D
        #     slf_header = slf_header.copy_as_2d()
        slf_header.is_2d = True
        shape = (slf_header.nb_var, slf_header.nb_nodes_2d)

    try:
        slf_header.params = ds.attrs["params"]
    except KeyError:
        slf_header.build_params()

    resout = Serafin.Write(fout, slf_header.language, overwrite=True)
    resout.__enter__()
    resout.write_header(slf_header)

    t0 = np.datetime64(datetime(*slf_header.date))

    try:
        time_serie = compute_duration_between_datetime(t0, ds.time.values)
    except AttributeError:
        return  # no time (header only is written)
    if isinstance(time_serie, float):
        time_serie = [time_serie]
    for it, t_ in enumerate(time_serie):
        temp = np.empty(shape, dtype=slf_header.np_float_type)
        for iv, var in enumerate(slf_header.var_IDs):
            if slf_header.nb_frames == 1:
                temp[iv] = ds[var].values
            else:
                temp[iv] = ds.isel(time=it)[var].values
            if slf_header.nb_planes > 1:
                temp[iv] = np.reshape(np.ravel(temp[iv]), (slf_header.nb_planes, slf_header.nb_nodes_2d))
        resout.write_entire_frame(
            slf_header,
            t_,
            np.reshape(temp, (slf_header.nb_var, slf_header.nb_nodes)),
        )


class SelafinLazyArray(BackendArray):
    # not implemented yet: too slow
    def __init__(self, slf_reader, var, dtype, shape):
        self.slf_reader = slf_reader
        self.var = var
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        if isinstance(key, tuple):
            if len(key) == 3:
                time_key, plan_key, node_key  = key
            elif len(key) == 2:
                time_key, node_key = key
                plan_key = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Convert time_key and node_key to ranges to handle steps and to list indices for SELAFIN reader
        if isinstance(time_key, slice):
            time_indices = range(*time_key.indices(self.shape[0]))
        elif isinstance(time_key, int):
            time_indices = [time_key]
        else:
            raise ValueError("time_key must be an integer or slice")

        if isinstance(node_key, slice):
            node_indices = range(*node_key.indices(self.shape[-1]))
        elif isinstance(node_key, int):
            node_indices = [node_key]
        else:
            raise ValueError("node_key must be an integer or slice")

        if plan_key is not None:
            if isinstance(plan_key, slice):
                plan_indices = range(*plan_key.indices(self.shape[1]))
            elif isinstance(plan_key, int):
                plan_indices = [plan_key]
            else:
                raise ValueError("plan_key must be an integer or slice")
            data_shape = (len(time_indices), len(plan_indices), len(node_indices))
        else:
            data_shape = (len(time_indices), len(node_indices))

        # Initialize data array to hold the result
        data = np.empty(data_shape, dtype=self.dtype)

        # Iterate over the time indices to read the required time steps
        for it, t in enumerate(time_indices):
            temp = self.slf_reader.read_var_in_frame(t, self.var)  # shape = (nb_nodes_3d,)
            temp = np.reshape(temp, self.shape[1:])  # shape = (nb_planes, nb_nodes_2d)
            if node_key == slice(None) and plan_key == slice(None):  # speedup if not selection
                data[it] = temp
            else:
                if plan_key is None:
                    data[it] = temp[node_indices]
                else:
                    values = temp[plan_indices][:, node_indices]
                    data[it] = np.reshape(values, (len(plan_indices), len(node_indices)))

        # Remove dimension if key was an integer
        if isinstance(node_key, int):
            if plan_key is None:
                data = data[:, 0]
            else:
                data = data[:, :, 0]
        if isinstance(plan_key, int):
            data = data[:, 0, :]
        if isinstance(time_key, int):
            data = data[0, ...]
        return data


class SelafinBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
        # Below are custom arguments
        lazy_loading=True,
        lang=Serafin.LANG,
    ):
        # Initialize SELAFIN reader
        slf = read_serafin(filename_or_obj, lang)
        is_2d = slf.header.is_2d

        # Prepare dimensions, coordinates, and data variables
        if slf.header.date is None:
            slf.header.date = DEFAULT_DATE_START
        times = [datetime(*slf.header.date) + timedelta(seconds=t) for t in slf.time]
        npoin2 = slf.header.nb_nodes_2d
        ndp3 = slf.header.nb_nodes_per_elem
        nplan = slf.header.nb_planes
        x = slf.header.x
        y = slf.header.y
        vars = slf.header.var_IDs

        # Create data variables
        data_vars = {}
        dtype = np.dtype(slf.header.np_float_type)

        if nplan == 0:
            shape = (len(times), npoin2)
            dims = ["time", "node"]
        else:
            shape = (len(times), nplan, npoin2)
            dims = ["time", "plan", "node"]

        for var in vars:
            if lazy_loading:
                lazy_array = SelafinLazyArray(slf, var, dtype, shape)
                data = indexing.LazilyIndexedArray(lazy_array)
                data_vars[var] = xr.Variable(dims=dims, data=data)
            else:
                data = np.empty(shape, dtype=dtype)
                for time_index, t in enumerate(times):
                    values = slf.read_var_in_frame(time_index, var)
                    if is_2d:
                        data[time_index, :] = values
                    else:
                        data[time_index, :, :] = np.reshape(values, (nplan, npoin2))
                data_vars[var] = xr.Variable(dims=dims, data=data)

        coords = {
            "x": ("node", x[:npoin2]),
            "y": ("node", y[:npoin2]),
            "time": times,
            # Consider how to include IPOBO (with node and plan dimensions?)
            # if it's essential for your analysis
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        ds.attrs["title"] = slf.header.title.decode(Serafin.SLF_EIT).strip()
        ds.attrs["language"] = slf.header.language
        ds.attrs["float_size"] = slf.header.float_size
        ds.attrs["endian"] = slf.header.endian
        ds.attrs["params"] = slf.header.params
        ds.attrs["ipobo"] = slf.header.ipobo
        ds.attrs["ikle2"] = slf.header.ikle_2d
        if not is_2d:
            ds.attrs["ikle3"] = np.reshape(slf.header.ikle, (slf.header.nb_elements, ndp3))
        ds.attrs["variables"] = {
            var_ID: (name.decode(Serafin.SLF_EIT).rstrip(), unit.decode(Serafin.SLF_EIT).rstrip())
            for var_ID, name, unit in slf.header.iter_on_all_variables()
        }
        ds.attrs["date_start"] = slf.header.date

        return ds

    @staticmethod
    def guess_can_open(filename_or_obj):
        try:
            _, ext = os.path.splitext(str(filename_or_obj))
        except TypeError:
            return False
        return ext.lower() in {".slf"}

    description = "A SELAFIN file format backend for Xarray"
    url = "https://www.example.com/selafin_backend_documentation"


@xr.register_dataset_accessor("selafin")
class SelafinAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def write(self, filepath, **kwargs):
        """
        Write data from an Xarray dataset to a SELAFIN file.
        Parameters:
        - filename: String with the path to the output SELAFIN file.
        """
        ds = self._obj
        write_serafin(filepath, ds)
