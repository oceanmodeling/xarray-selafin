[![Available on pypi](https://img.shields.io/pypi/v/xarray-selafin.svg)](https://pypi.python.org/pypi/xarray-selafin/)
[![CI](https://github.com/oceanmodeling/xarray-selafin/actions/workflows/run_tests.yml/badge.svg)](https://github.com/oceanmodeling/xarray-selafin/actions/workflows/run_tests.yml)

# xarray backend for Selafin formats

Supports lazy loading by default.

## Install

To have the backend working in xarray, you need to follow these steps:

```
pip install xarray-selafin
```

or, if you are using conda/mamba:

```
conda install -c conda-forge xarray_selafin
```

## Read Selafin

```python
import xarray as xr

with xr.open_dataset("tests/data/r3d_tidal_flats.slf", engine="selafin") as ds:
    print(ds)  # do something with `ds`...
    # `ds.close()` not necessary

ds = xr.open_dataset("tests/data/r3d_tidal_flats.slf", lang="fr", engine="selafin")  # if variables are in French
print(ds)  # do something with `ds`...
ds.close()  # avoid a ResourceWarning (unclosed file)
```

```
<xarray.Dataset> Size: 5MB
Dimensions:  (time: 17, plan: 21, node: 648)
Coordinates:
    x        (node) float32 3kB ...
    y        (node) float32 3kB ...
  * time     (time) datetime64[ns] 136B 1900-01-01 ... 1900-01-02T20:26:40
Dimensions without coordinates: plan, node
Data variables:
    Z        (time, plan, node) float32 925kB ...
    U        (time, plan, node) float32 925kB ...
    V        (time, plan, node) float32 925kB ...
    W        (time, plan, node) float32 925kB ...
    MUD      (time, plan, node) float32 925kB ...
Attributes:
    title:       Sloped flume Rouse profile test
    language:    en
    float_size:  4
    endian:      >
    params:      (1, 0, 0, 0, 0, 0, 21, 5544, 0, 1)
    ipobo:       [   1  264  263 ... 5411 5412 5413]
    ikle2:       [[155 153 156]\n [310 307 305]\n [308 310 305]\n ...\n [537 ...
    ikle3:       [[  155   153   156   803   801   804]\n [  310   307   305 ...
    variables:   {'Z': ('ELEVATION Z', 'M'), 'U': ('VELOCITY U', 'M/S'), 'V':...
    date_start:  (1900, 1, 1, 0, 0, 0)
```

## Indexing

```python
ds_last = ds.isel(time=-1)  # last frame
```

## Manipulate variables

```python
ds = ds.assign(UTIMES100=lambda x: x.U * 100)  # Add a new variable
# ds.attrs["variables"]["UTIMES100"] = ("UTIMES100", "My/Unit")  # To provide variable name and unit (optional)
ds.drop_vars(["W"])  # Remove variable `VELOCITY W`
```

## Extracting a specific layer from a 3D DataSet

```python
ds_bottom = ds.selafin.get_dataset_as_2d(plan=0)  # bottom layer
```

## Write Selafin

```python
ds.selafin.write("output_file.slf")
```

## DataSet content

### Dimensions
In 2D:
1. time
2. node

in 3D:
1. time
2. plan
3. node

### Coordinates

| Coordinate | Description            |
|------------|------------------------|
| x          | East mesh coordinates  |
| y          | North mesh coordinates |
| time       | Datetime serie         |

### Attributes

All attributes are optional except `ikle2`:

| Attribute  | Description                                                             | Default value                  |
|------------|-------------------------------------------------------------------------|--------------------------------|
| title      | Serafin title                                                           | "Converted with array-serafin" |
| language   | Language for variable detection                                         | "en"                           |
| float_size | Float size                                                              | 4 (single precision)           |
| endian     | File endianness                                                         | ">"                            |
| params     | Table of integer parameters                                             | (can be rebuilt)               |
| ikle2      | Connectivity table in 2D (1-indexed)                                    | -                              |
| ikle3      | Connectivity table in 3D (1-indexed, only in 3D, optional)              | (can be rebuilt from 2D)       |
| variables  | Dictionary with variable names and units (key is variable abbreviation) | -                              |
| date_start | Starting date with integers (year to seconds)                           | (from first time serie)        |
