# Concrete examples with holoviews

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oceanmodeling/xarray-selafin/blob/main/workshop/01_getting_started_colab.ipynb)

The workshop notebooks can be run interactively in your browser via Google Colab, no installation required.

| Notebook | Colab |
|---|---|
| 01 — Getting started | [open](https://colab.research.google.com/github/oceanmodeling/xarray-selafin/blob/main/workshop/01_getting_started_colab.ipynb) |
| 02 — Advanced: ERA5 atmospheric forcing | [open](https://colab.research.google.com/github/oceanmodeling/xarray-selafin/blob/main/workshop/02_advanced.ipynb) |

!!! note
    Notebook 01 has a dedicated **Colab version** (`01_getting_started_colab.ipynb`).
    Colab does not support kernel-backed widgets, so interactive sliders (e.g. along
    `time`) are replaced by single frames (`isel(time=0)`) there, while the local
    version keeps fully interactive plots. The Colab version starts with a bootstrap
    cell that installs the dependencies (`pip install xarray-selafin ...`), downloads
    the sample data, and calls `output.enable_custom_widget_manager()` once.

## Plotting SELAFIN data

The unstructured triangular mesh is stored in `ds.attrs["ikle2"]`. We use `holoviews.TriMesh` to render filled triangulations directly:

```python
import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import HoverTool

def plot_selafin(ds, var, geo=False, **kwargs):
    if geo:
        import geoviews as lib
    else:
        import holoviews as lib
    simplices = pd.DataFrame(ds.attrs["ikle2"] - 1, columns=["v0", "v1", "v2"])
    nodes = np.column_stack([ds.x.values, ds.y.values, ds[var].values])
    nodes = lib.Points(nodes, vdims="z")
    trimesh = lib.TriMesh((simplices, nodes), name=var)
    hover = HoverTool(tooltips=[("x", "$x"), ("y", "$y"), (var, "@z")])
    return trimesh.opts(
        lib.opts.TriMesh(
            filled=True, colorbar=True, edge_color="z",
            inspection_policy="edges", node_size=0, tools=[hover],
            **kwargs,
        )
    )
```

!!! warning "Use [Thalassa](https://github.com/ec-jrc/Thalassa) for big meshes"
    The above function is just given as an example. We strongly recommend leveraging the use of `datashader` rendering.
    For large datasets, consider using the [Thalassa](https://github.com/ec-jrc/Thalassa) library, which uses `datashader` for efficient big data visualisations.

---

## 2D: Tidal Flats

```python
import xarray as xr

ds = xr.open_dataset("tests/data/r2d_tidal_flats.slf", engine="selafin")
```

### Free surface

```python
plot_selafin(ds.isel(time=-1), "S", cmap="coolwarm", width=1000, height=200)
```

<iframe src="./2d_free_surface.html" width="100%" height="200" style="border:none;"></iframe>

### Bathymetry

```python
plot_selafin(ds.isel(time=-1), "B", cmap="viridis", width=1000, height=200)
```

<iframe src="./2d_bathymetry.html" width="100%" height="200" style="border:none;"></iframe>

### Time series at a node

```python
ds.isel(node=0).hvplot(x="time", y="S")
```

<iframe src="./2d_timeseries.html" width="100%" height="410" style="border:none;"></iframe>

---

## TOMAWAC: Waves (geo-referenced)

For geo-referenced data, pass `geo=True` to use `geoviews` with a tile background:

```python
ds = xr.open_dataset("tests/data/tom_manche.slf", engine="selafin", lang="fr")
plot_selafin(ds.isel(time=-1), "HAUTEUR_HM0", geo=True, cmap="rainbow4", width=800, height=800)
```

<iframe src="./tomawac_hm0.html" width="100%" height="710" style="border:none;"></iframe>

---

## 3D: Bump (hydraulic jump)

3D data is not natively rendered as a volume with hvplot. You can view **horizontal sections** by indexing on the `plan` dimension:

```python
ds = xr.open_dataset("tests/data/r3d_bump.slf", engine="selafin")
plot_selafin(ds.isel(time=-1, plan=-1), "Z", cmap="coolwarm", width=1000, height=200)
```

<iframe src="./3d_bump_z.html" width="100%" height="310" style="border:none;"></iframe>

You can also use `hvplot.scatter` for **vertical cross-sections**:

```python
ds.isel(time=-1).hvplot.scatter(x="x", y="Z", c="U", cmap="coolwarm")
```

<iframe src="./3d_bump_scatter.html" width="100%" height="510" style="border:none;"></iframe>

---

## Creating a SELAFIN file from scratch

To write a SELAFIN file you need two critical attributes on the dataset:

1. **`ikle2`** -- the triangular connectivity table, **1-indexed**
2. **`variables`** -- a dict mapping variable short names to `(long_name, unit)` tuples

```python
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay

# Create a mesh
xx = np.arange(10)
yy = np.arange(10)
x, y = np.meshgrid(xx, yy)
x = x.ravel() + 0.3*np.random.rand(100)
y = y.ravel() + 0.3*np.random.rand(100)
ikle = Delaunay(np.vstack((x, y)).T).simplices  # 0-indexed from scipy

# Create data
ds = xr.Dataset(
    {
        "WINDX": (("time", "node"), np.matlib.repmat(np.arange(0,10,0.1), 5, 1)),
        "WINDY": (("time", "node"), np.matlib.repmat(np.arange(0,20,0.2), 5, 1)),
        "PATM":  (("time", "node"), np.full((5, 100), 101325.0)),
    },
    coords={
        "x": ("node", x),
        "y": ("node", y),
        "time": pd.date_range("2024-01-01", periods=5, freq="6h"),
    },
)

# --- CRITICAL: set attributes for SELAFIN export ---
ds.attrs["ikle2"] = ikle + 1  # convert to 1-based indexing

ds.attrs["variables"] = {
    "WINDX": ("WINDX", "M/S"),
    "WINDY": ("WINDY", "M/S"),
    "PATM":  ("PATM", "PASCAL"),
}
```

!!! warning "1-based connectivity"
    SELAFIN uses **1-based indexing** for `ikle2`. If you get your triangulation from scipy or meshio (which are 0-based), you **must** add 1 before assigning.

!!! warning "Variables dictionary"
    Without `ds.attrs["variables"]`, the export will either fail or produce a file with empty variable names. Each key must match a data variable name, and the value is a tuple of `(long_name, unit)`.

Now write:

```python
from xarray_selafin.xarray_backend import SelafinAccessor  # registers .selafin accessor

ds.selafin.write("wind_forcing.slf")
```

## Visualize vector field

```python
import holoviews as hv
import numpy as np

x = ds.isel(time=-1).x.values
y = ds.isel(time=-1).y.values
u = ds.isel(time=-1)["WINDX"].values
v = ds.isel(time=-1)["WINDY"].values

angle = np.arctan2(v, u)
mag = np.sqrt(u**2 + v**2)

hv.VectorField((x, y, angle, mag)).opts(
        magnitude="Magnitude",
        color="Magnitude",
        colorbar=True,
        cmap="rainbow4",
        tools=["hover"],
        width = 1000,
        height = 600
    )
```

<iframe src="./scratch_wind.html" width="100%" height="610" style="border:none;"></iframe>
