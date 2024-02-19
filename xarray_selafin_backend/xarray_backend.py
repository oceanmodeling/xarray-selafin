from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core.indexing import LazilyIndexedArray
from xarray.core import indexing
import dask.array as da
import numpy as np
import xarray as xr
import os 
from .selafin_io_pp import *
import logging


class SelafinLazyArray(BackendArray):
    def __init__(self, selafin_reader, variable_name, dtype, shape):
        self.selafin_reader = selafin_reader
        self.variable_name = variable_name
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        logging.debug('Raw indexing method called')
        # time_key, node_key = key
        if isinstance(key, tuple):
            time_key, node_key = key
        else:
            # Assuming if only one key is provided, it's for the time dimension
            time_key = key
            # And we want to select all nodes
            node_key = slice(None)
        
        logging.info(key)
        print(key)

        # Convert time_key and node_key to ranges to handle steps and to list indices for SELAFIN reader
        time_indices = range(*time_key.indices(self.shape[0])) if isinstance(time_key, slice) else [time_key]
        node_indices = range(*node_key.indices(self.shape[1])) if isinstance(node_key, slice) else [node_key]

        # Initialize data array to hold the result
        data_shape = (len(time_indices), len(node_indices))
        data = np.empty(data_shape, dtype=self.dtype)

        # Iterate over the time indices to read the required time steps
        for i, t_idx in enumerate(time_indices):
            self.selafin_reader.readVariables(t_idx)
            temp = self.selafin_reader.getVarValues()
            variable_index = self.selafin_reader.getVarNames().index(self.variable_name)

            # Handling case where node_key is an integer (selecting a single node across time)
            if isinstance(node_key, int):
                data[i] = temp[variable_index, node_key]
            else:
                # For each time step, extract the required nodes based on node_key
                for j, n_idx in enumerate(node_indices):
                    data[i, j] = temp[variable_index, n_idx]

        # Return the fetched data, reshaping if only a single dimension was accessed
        if isinstance(key, slice) or isinstance(key, int):
            return data.squeeze()  # Remove single-dimensional entries from the array shape
        return data

class SelafinBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
    ):
        # Initialize SELAFIN reader
        slf = ppSELAFIN(filename_or_obj)
        slf.readHeader()
        slf.readTimes()
        
        # Prepare dimensions, coordinates, and data variables
        x = slf.getMeshX()
        y = slf.getMeshY()
        times = slf.getTimes()
        ikle = slf.getIKLE()  # Adjust if necessary for 0-based indexing
        ipobo = slf.getIPOBO()  # Boundary points
        
        # Note: Consider how you wish to handle or transform these for use in Xarray
        nelem = slf.getNELEM()
        npoin = slf.getNPOIN()
        ndp = slf.NDP  # Number of points per element
        
        # Create data variables using Dask arrays for the variables
        data_vars = {}
        for name in slf.getVarNames():
            dtype = np.float32  # Adjust based on SELAFIN precision
            shape = (len(times), len(x))
            lazy_array = SelafinLazyArray(slf, name.strip(), dtype, shape)
            dask_array = da.from_array(lazy_array, chunks=(1, shape[1]))  # Define chunks
            data_vars[name.strip()] = (["time", "node"], dask_array)
        
        # Including essential parameters directly in the dataset
        coords = {
            "x": ("node", x),
            "y": ("node", y),
            "time": times,
            # Adding IKLE as a coordinate or data variable for mesh connectivity
            "ikle": (("nelem", "nnode"), ikle),
            # Consider how to include IPOBO if it's essential for your analysis
        }
        
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Adding additional metadata as attributes
        ds.attrs["nelem"] = nelem
        ds.attrs["npoin"] = npoin
        ds.attrs["nnode"] = ndp
        ds.attrs["IPARAM"] = slf.IPARAM
        # IPOBO can be added as a non-dimension coordinate if it's significant
        ds["ipobo"] = ("node", ipobo)
        
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
        # Assuming ds is your Xarray Dataset
        ds = self._obj
        
        # Simplified example of writing logic (details need to be implemented):
        slf_writer = ppSELAFIN(filepath)
        slf_writer.setTitle("Converted from Xarray")
        
        # Set mesh information from dataset coordinates or attributes
        # This is a simplified example; adjust according to your data structure
        nelem, npoin, ndp = ds.attrs['nelem'], ds.attrs['npoin'], ds.attrs['nnode']
        ikle = ds['ikle'].values 
        x, y = ds['x'].values, ds['y'].values
        slf_writer.IPARAM = list(ds.attrs['IPARAM'])
        
        slf_writer.setMesh(nelem, npoin, ndp, ikle, np.zeros(npoin, dtype=np.int32), x, y)
        
        # Prepare and write data variables
        slf_writer.writeHeader()
        for time_step in ds.time:
            temp = np.array([ds[var].sel(time=time_step).values for var in ds.data_vars])
            slf_writer.writeVariables(time_step.values, temp)
        
        slf_writer.close()