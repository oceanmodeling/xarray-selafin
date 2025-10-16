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


def range_from_key(k, n):
    if isinstance(k, slice):
        return range(*k.indices(n))
    elif isinstance(k, int):
        return range(k, k + 1)
    else:
        raise ValueError("index must be int or slice")


def compute_duration_between_datetime(t0, time_serie):
    return (time_serie - t0).astype("timedelta64[s]").astype(float)


def read_serafin(filepath, lang):
    resin = Serafin.Read(filepath, lang)
    resin.__enter__()
    resin.read_header()
    resin.get_time()
    return resin


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

    def _raw_indexing_method_unlocked(self, key):
        ndim = self.ndim  # = len(self.shape)
        if ndim not in (2, 3):
            raise NotImplementedError(f"Unsupported SELAFIN shape {ndim}")
        assert len(key) == len(self.shape)

        if not isinstance(key, tuple):
            raise NotImplementedError("SELAFIN access must use tuple indexing")

        nb_nodes_2d = self.shape[-1]  # last dimension

        # Parse keys
        if ndim == 3:  # (3D)
            time_key, plan_key, node_key = key
        else:  # ndim = 2 (2D)
            time_key, node_key = key
            plan_key = None

        # Build indices
        time_range = range_from_key(time_key, self.shape[0])
        if ndim == 3:  # (3D)
            nb_planes = self.shape[1]
            plan_range = range_from_key(plan_key, nb_planes)
            node_range = range_from_key(node_key, nb_nodes_2d)
            ds_shape = (len(time_range), len(plan_range), len(node_range))
        else:  # ndim = 2 (2D)
            nb_planes = 1
            node_range = range_from_key(node_key, nb_nodes_2d)
            ds_shape = (len(time_range), len(node_range))
            plan_range = None

        res_array = np.empty(ds_shape, dtype=self.dtype)

        for ds_index_time, time_index in enumerate(time_range):
            flatten_values = self.filename_or_obj.read_var_in_frame(time_index, self.var)  # flatten np.ndarray
            if ndim == 3:
                all_values = flatten_values.reshape((nb_planes, nb_nodes_2d))
                if plan_range == slice(None, None, None) and node_key == slice(None, None, None):  # avoid a subset to speedup
                    res_array[ds_index_time, :, :] = all_values
                else:
                    res_array[ds_index_time, :, :] = all_values[np.ix_(plan_range, node_range)]
            else:
                if node_key == slice(None, None, None):  # avoid a subset to speedup
                    res_array[ds_index_time, :] = flatten_values
                else:
                    res_array[ds_index_time, :] = flatten_values[node_range]

        # Remove some dimensions if it was selected by integer
        squeeze_dims = []
        if isinstance(time_key, int):
            squeeze_dims.append(0)
        if isinstance(plan_key, int):
            squeeze_dims.append(1)
        if isinstance(node_key, int):
            squeeze_dims.append(-1)
        if squeeze_dims:
            res_array = np.squeeze(res_array, axis=tuple(squeeze_dims))

        return res_array

    def _raw_indexing_method(self, key):
        if self.lock is None:
            return self._raw_indexing_method_unlocked(key)
        else:
            with self.lock:
                return self._raw_indexing_method_unlocked(key)


class SelafinBackendEntrypoint(BackendEntrypoint):

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
        # Below are custom arguments
        disable_lock=False,
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

        if disable_lock:
            file_lock = None
        else:
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
                    var=var,
                )
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
    url = "https://github.com/oceanmodeling/xarray-selafin/"


@xr.register_dataset_accessor("selafin")
class SelafinAccessor:

    def __init__(self, xarray_obj):
        self._ds = xarray_obj
        self._header = None

    def _build_header(self):
        ds = self._ds

        # Title
        title = ds.attrs.get("title", "Converted with array-serafin")
        header = Serafin.SerafinHeader(title)

        # File precision
        float_size = ds.attrs.get("float_size", 4)  # Default: single precision
        if float_size == 4:
            header.to_single_precision()
        elif float_size == 8:
            header.to_double_precision()
        else:
            raise NotImplementedError

        header.endian = ds.attrs.get("endian", ">")  # Default: ">"

        header.nb_frames = ds.sizes.get("time", 0)

        try:
            header.date = ds.attrs["date_start"]
        except KeyError:
            # Retrieve starting date from first time
            if header.nb_frames == 0:
                first_time = ds.time
            else:
                first_time = ds.time[0]
            first_date_str = first_time.values.astype(str)  # "1900-01-01T00:00:00.000000000"
            first_date_str = first_date_str.rstrip("0") + "0"  # "1900-01-01T00:00:00.0"
            try:
                date = datetime.strptime(first_date_str, "%Y-%m-%dT%H:%M:%S.%f")
                header.date = attrgetter("year", "month", "day", "hour", "minute", "second")(date)
            except ValueError:
                header.date = DEFAULT_DATE_START

        # Variables
        header.language = ds.attrs.get("language", Serafin.LANG)
        for var in ds.data_vars:
            try:
                name, unit = ds.attrs["variables"][var]
                header.add_variable_str(var, name, unit)
            except KeyError:
                try:
                    header.add_variable_from_ID(var)
                except Serafin.SerafinRequestError:
                    header.add_variable_str(var, var, "?")
        header.nb_var = len(header.var_IDs)

        if "plan" in ds.dims:  # 3D
            header.is_2d = False
            nplan = len(ds.plan)
            header.nb_nodes_per_elem = 6
            header.nb_elements = len(ds.attrs["ikle2"]) * (nplan - 1)
            header.nb_planes = nplan
        else:  # 2D
            header.is_2d = True
            nplan = 1  # just to do a multiplication below
            header.nb_nodes_per_elem = ds.attrs["ikle2"].shape[1]
            header.nb_elements = len(ds.attrs["ikle2"])
            header.nb_planes = 0  # convention in Selafin for 2D files

        header.nb_nodes_2d = ds.sizes["node"]
        header.nb_nodes = header.nb_nodes_2d * nplan

        x = ds.coords["x"].values
        y = ds.coords["y"].values
        if not header.is_2d:
            x = np.tile(x, nplan)
            y = np.tile(y, nplan)
        header.x = x
        header.y = y
        header.mesh_origin = (0, 0)  # Should be integers
        header.x_stored = x - header.mesh_origin[0]
        header.y_stored = y - header.mesh_origin[1]
        header.ikle_2d = ds.attrs["ikle2"]
        if header.is_2d:
            header.ikle = header.ikle_2d.flatten()
        else:
            try:
                header.ikle = ds.attrs["ikle3"]
            except KeyError:
                # Rebuild IKLE from 2D
                header.ikle = header.compute_ikle(nplan, header.nb_nodes_per_elem)

        try:
            header.ipobo = ds.attrs["ipobo"]
        except KeyError:
            # Rebuild IPOBO
            header.build_ipobo()

        try:
            header.params = ds.attrs["params"]
        except KeyError:
            header.build_params()

        self._header = header

    def _write_all(self, filepath):
        header = self._header

        with Serafin.Write(filepath, header.language, overwrite=True) as resout:
            resout.write_header(header)

            t0 = np.datetime64(datetime(*header.date))

            try:
                time_serie = compute_duration_between_datetime(t0, self._ds.time.values)
            except AttributeError:
                return  # no time (header only is written)
            if isinstance(time_serie, float):
                time_serie = [time_serie]

            shape = (header.nb_var, header.nb_nodes)
            values = np.empty(shape, dtype=header.np_float_type)
            for time_index, time in enumerate(time_serie):
                for var_index, var in enumerate(header.var_IDs):
                    if header.nb_frames == 1:
                        array = self._ds[var].values
                    else:
                        array = self._ds.isel(time=time_index)[var].values
                    if header.is_2d:
                        values[var_index, :] = array
                    else:
                        values[var_index, :] = array.ravel()

                resout.write_entire_frame(
                    header,
                    time,
                    values,
                )

    def write(self, filepath, **kwargs):
        """
        Write data from an Xarray dataset to a SELAFIN file.
        Parameters:
        - filename: String with the path to the output SELAFIN file.
        """
        self._build_header()
        self._write_all(filepath)
