# xarray-selafin

[![Available on pypi](https://img.shields.io/pypi/v/xarray-selafin.svg)](https://pypi.python.org/pypi/xarray-selafin/)
[![CI](https://github.com/oceanmodeling/xarray-selafin/actions/workflows/run_tests.yml/badge.svg)](https://github.com/oceanmodeling/xarray-selafin/actions/workflows/run_tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oceanmodeling/xarray-selafin/blob/main/workshop/01_getting_started_colab.ipynb)

**An xarray backend for SELAFIN/SERAFIN file formats used by TELEMAC and other hydraulic solvers.**

`xarray-selafin` provides seamless integration between [xarray](https://docs.xarray.dev/) and the SELAFIN binary format, enabling lazy loading, slicing, and standard xarray operations on hydraulic simulation results.

## Features

- **Lazy loading** by default -- only reads data when accessed
- **Full read/write** support for 2D and 3D SELAFIN files
- **Bilingual** variable support (French and English)
- **Dask-compatible** for parallel and out-of-core computation
- **Layer extraction** from 3D to 2D datasets
- **Round-trip fidelity** -- read, modify, and write back without data loss

## Quick Example

```python
import xarray as xr

ds = xr.open_dataset("simulation.slf", engine="selafin")
print(ds)
```

```
<xarray.Dataset> Size: 226kB
Dimensions:  (time: 17, node: 648)
Coordinates:
  * time     (time) datetime64[us] 136B 1900-01-01 ... 1900-01-02T20:26:40
    x        (node) float32 3kB ...
    y        (node) float32 3kB ...
Dimensions without coordinates: node
Data variables:
    U        (time, node) float32 44kB ...
    V        (time, node) float32 44kB ...
    H        (time, node) float32 44kB ...
    S        (time, node) float32 44kB ...
    B        (time, node) float32 44kB ...
```

## Navigation

| | |
|---|---|
| **[Getting Started](getting-started.md)** | Installation, reading, writing, slicing |
| **[Examples](examples.md)** | Interactive plots: 2D, TOMAWAC, 3D, create from scratch |
| **[API Reference](api.md)** | Dataset structure, attributes, export requirements |
