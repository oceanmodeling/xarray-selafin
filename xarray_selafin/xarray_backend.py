"""
Documentation on how to implement a new backend in xarray
* https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html
* https://tutorial.xarray.dev/advanced/backends/2.Backend_with_Lazy_Loading.html
"""
import os
from datetime import datetime
from datetime import timedelta
import numpy as np
from operator import attrgetter
import threading
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

from xarray_selafin import Serafin


try:
    import dask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


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

    with Serafin.Write(fout, slf_header.language, overwrite=True) as resout:
        resout.write_header(slf_header)

        t0 = np.datetime64(datetime(*slf_header.date))

        try:
            time_serie = compute_duration_between_datetime(t0, ds.time.values)
        except AttributeError:
            return  # no time (header only is written)
        if isinstance(time_serie, float):
            time_serie = [time_serie]
        for time_index, time in enumerate(time_serie):
            temp = np.empty(shape, dtype=slf_header.np_float_type)
            for iv, var in enumerate(slf_header.var_IDs):
                if slf_header.nb_frames == 1:
                    temp[iv] = ds[var].values
                else:
                    temp[iv] = ds.isel(time=time_index)[var].values
                if slf_header.nb_planes > 1:
                    temp[iv] = np.reshape(
                        np.ravel(temp[iv]), (slf_header.nb_planes, slf_header.nb_nodes_2d)
                    )
            resout.write_entire_frame(
                slf_header,
                time,
                np.reshape(temp, (slf_header.nb_var, slf_header.nb_nodes)),
            )


class SelafinLazyArray(BackendArray):

    def __init__(self, filename_or_obj, shape, dtype, lock, var):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        # Below are other backend specific keyword arguments
        self.var = var

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        with self.lock:
            ndim = len(self.shape)
            if ndim not in (2, 3):
                raise NotImplementedError(f"Unsupported SELAFIN shape {self.shape}")

            if not isinstance(key, tuple):
                raise NotImplementedError("SELAFIN access must use tuple indexing")

            # Pad key with slices to match array dimensions
            ndim = len(self.shape)
            if len(key) < ndim:
                key = key + (slice(None),) * (ndim - len(key))

            # --- Parse keys
            if ndim == 3:
                # 3D file
                if len(key) == 3:
                    time_key, plan_key, node_key = key
                elif len(key) == 2:
                    time_key, node_key = key
                    plan_key = slice(None)
                else:
                    raise NotImplementedError("Only (time, plan, node) or (time, node) supported for 3D files")
            else:
                # 2D file
                if len(key) == 2:
                    time_key, node_key = key
                elif len(key) == 1:
                    time_key = key[0]
                    node_key = slice(None)
                else:
                    raise NotImplementedError("Only (time, node) supported for 2D files")

            # --- helper
            def _range_from_key(k, n):
                if isinstance(k, slice):
                    return range(*k.indices(n))
                elif isinstance(k, int):
                    return [k]
                else:
                    raise ValueError("index must be int or slice")

            time_indices = _range_from_key(time_key, self.shape[0])

            if ndim == 3:
                plan_indices = _range_from_key(plan_key, self.shape[1])
                node_indices = _range_from_key(node_key, self.shape[2])
                data_shape = (len(time_indices), len(plan_indices), len(node_indices))
            else:  # ndim = 2
                node_indices = _range_from_key(node_key, self.shape[1])
                data_shape = (len(time_indices), len(node_indices))

            data = np.empty(data_shape, dtype=self.dtype)

            for data_index, time_index in enumerate(time_indices):
                temp = self.filename_or_obj.read_var_in_frame(time_index, self.var)  # np.ndarray

                if ndim == 3:
                    temp = np.reshape(temp, (self.shape[1], self.shape[2]))  # (nplan, nnode)
                    data[data_index, :, :] = temp[np.ix_(plan_indices, node_indices)]
                else:  # ndim = 2
                    data[data_index, :] = temp[node_indices]

            return data.squeeze()


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
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
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

        if DASK_AVAILABLE:
            file_lock = dask.utils.SerializableLock()
        else:
            file_lock = threading.Lock()

        for var in vars:
            if lazy_loading:
                lazy_array = SelafinLazyArray(
                    filename_or_obj=slf,
                    shape=shape,
                    dtype=dtype,
                    lock=file_lock,
                    var=var)
                data = indexing.LazilyIndexedArray(lazy_array)
                data_vars[var] = xr.Variable(dims=dims, data=data)
            else:
                data = np.empty(shape, dtype=dtype)
                for time_index, _ in enumerate(times):
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

        # Avoid a ResourceWarning (unclosed file)
        def close():
            slf.__exit__()

        ds.set_close(close)

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
