# API Reference

## Opening a SELAFIN file

```python
import xarray as xr

ds = xr.open_dataset(filename, engine="selafin", lang="en", lazy_loading=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` or path | -- | Path to the `.slf` file |
| `engine` | `str` | -- | Must be `"selafin"` |
| `lang` | `str` | `"en"` | Variable language: `"en"` or `"fr"` |
| `lazy_loading` | `bool` | `True` | If `True`, data is read on demand |
| `disable_lock` | `bool` | `False` | Disable thread-safe file locking |
| `chunks` | `dict` or `None` | `None` | Dask chunk sizes (e.g. `{"time": -1, "node": 50}`) |

## Dataset structure

### Dimensions

| Dimension | Description | Present in |
|-----------|-------------|------------|
| `time` | Time steps | 2D and 3D |
| `node` | Mesh nodes | 2D and 3D |
| `plan` | Vertical layers (sigma levels) | 3D only |

### Coordinates

| Coordinate | Dimension | Type | Description |
|------------|-----------|------|-------------|
| `x` | `(node,)` | `float32` | Easting / x mesh coordinates |
| `y` | `(node,)` | `float32` | Northing / y mesh coordinates |
| `time` | `(time,)` | `datetime64` | Time stamps |

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `"Converted with xarray-selafin"` | File title (up to 80 chars) |
| `language` | `str` | `"en"` | Language for variable detection |
| `float_size` | `int` | `4` | Precision: `4` (single) or `8` (double) |
| `endian` | `str` | `">"` | Byte order: `">"` (big) or `"<"` (little) |
| `params` | `tuple` | *(rebuilt)* | Integer parameter table (10 values) |
| `ipobo` | `ndarray` | *(rebuilt)* | Boundary node pointer array |
| `ikle2` | `ndarray` | **required** | 2D connectivity, shape `(n_elements, 3)`, **1-indexed** |
| `ikle3` | `ndarray` | *(rebuilt)* | 3D connectivity, shape `(n_prisms, 6)`, 1-indexed |
| `variables` | `dict` | *(auto-detected)* | Maps short names to `(long_name, unit)` tuples |
| `date_start` | `tuple` | *(from first time)* | `(year, month, day, hour, minute, second)` |

---

## Writing a SELAFIN file

```python
from xarray_selafin.xarray_backend import SelafinAccessor  # registers .selafin accessor

ds.selafin.write("output.slf")
```

### Required attributes for export

!!! danger "Export will fail without these"
    The two attributes below **must** be set correctly before calling `ds.selafin.write()`:

**`ikle2`** -- triangular connectivity table, **1-based indexing**:

```python
from scipy.spatial import Delaunay

ikle = Delaunay(np.vstack((x, y)).T).simplices  # 0-based from scipy
ds.attrs["ikle2"] = ikle + 1                    # SELAFIN requires 1-based
```

**`variables`** -- dictionary mapping each data variable name to a `(long_name, unit)` tuple:

```python
ds.attrs["variables"] = {
    "WINDX": ("WINDX", "M/S"),
    "WINDY": ("WINDY", "M/S"),
    "PATM":  ("PATM", "PASCAL"),
    "TAIR":  ("TAIR", "DEGREES C"),
}
```

Each key **must match** a variable name in the dataset. Without this dict, variable metadata in the output file will be empty or the write will crash.

---

## Variable definitions

Built-in TELEMAC variable definitions (French and English):

### 2D variables

| ID | French name | English name | Unit |
|----|-------------|--------------|------|
| `U` | VITESSE U | VELOCITY U | M/S |
| `V` | VITESSE V | VELOCITY V | M/S |
| `C` | CELERITE | CELERITY | M/S |
| `H` | HAUTEUR D'EAU | WATER DEPTH | M |
| `S` | SURFACE LIBRE | FREE SURFACE | M |
| `B` | FOND | BOTTOM | M |
| `F` | FROUDE | FROUDE NUMBER | |
| `Q` | DEBIT SCALAIRE | SCALAR FLOWRATE | M2/S |
| `Q` | DEBIT Q | FLOWRATE Q | M2/S |
| `I` | DEBIT SUIVANT X | FLOWRATE ALONG X | M2/S |
| `I` | DEBIT QX | FLOWRATE QX | M2/S |
| `J` | DEBIT SUIVANT Y | FLOWRATE ALONG Y | M2/S |
| `J` | DEBIT QY | FLOWRATE QX | M2/S |
| `E` | DISSIPATION | DISSIPATION | WATT/KG |
| `D` | VISCOSITE TURB. | VISCOSITY | M2/S |
| `K` | ENERGIE TURBUL. | TURBULENT ENERG. | JOULE/KG |
| `P` | PRESSION ATMOS. | AIR PRESSURE | PASCAL |
| `M` | VITESSE SCALAIRE | SCALAR VELOCITY | M/S |
| `X` | VENT X | WIND ALONG X | M/S |
| `Y` | VENT Y | WIND ALONG Y | M/S |
| `W` | FROTTEMENT | BOTTOM FRICTION | |
| `W` | COEFT FROTTEMENT | FRICTION COEFT | |
| `A` | DERIVE EN X | DRIFT ALONG X | M |
| `G` | DERIVE EN Y | DRIFT ALONG Y | M |
| `L` | NBRE DE COURANT | COURANT NUMBER | |
| `MAXZ` | COTE MAXIMUM | HIGH WATER MARK | M |
| `TMXZ` | TEMPS COTE MAXI | HIGH WATER TIME | S |
| `MAXV` | VITESSE MAXIMUM | HIGHEST VELOCITY | M/S |
| `TMXV` | T VITESSE MAXI | TIME OF HIGH VEL | S |
| `US` | VITESSE DE FROT. | FRICTION VEL. | M/S |
| `US` | VITESSE DE FROT | FRICTION VELOCIT | M/S |
| `TAU` | CONTRAINTE | BED SHEAR STRESS | PASCAL |
| `TAU` | FROTTEMENT TOB | BED SHEAR STRESS | PASCAL |
| `DMAX` | DIAMETRE | DIAMETER | MM |
| `WH` | HAUTEUR HM0 | WAVE HEIGHT HM0 | M |
| `PPT` | PERIODE PIC TPR5 | PEAK PERIOD TPR5 | S |
| `RB` | FOND RIGIDE | RIGID BED | M |
| `THETAW` | DIRECTION MOY | MEAN DIRECTION | DEG |
| `QS` | DEBIT SOLIDE | SOLID DISCH | M2/S |
| `QSX` | DEBIT SOLIDE X | SOLID DISCH X | M2/S |
| `QSY` | DEBIT SOLIDE Y | SOLID DISCH Y | M2/S |
| `QS` | DEBIT SOLIDE | SOLID DISCHARGE | M2/S |
| `QSX` | DEBIT SOL EN X | SOLID DIS IN X | M2/S |
| `QSY` | DEBIT SOL EN Y | SOLID DIS IN Y | M2/S |
| `EV` | EVOLUTION | EVOLUTION | M |
| `EV` | EVOLUTION FOND | BED EVOLUTION | M |
| `KS` | RUGOSITE TOTALE. | RUGOSITE TOTALE | M |
| `KS` | RUGOSITE TOTALE | RUGOSITE TOTALE | M |
| `MU` | CORR FROTT PEAU | FROT. PEAU MU | |
| `D50` | DIAMETRE MOYEN | MEAN DIAMETER | M |
| `UWB` | VITESSE FOND | BOTTOM VELOCITY | M/S |
| `QSBL` | QS CHARRIAGE | QS BEDLOAD | M2/S |
| `QSBLX` | QS CHARRIAGE X | QS BEDLOAD X | M2/S |
| `QSBLY` | QS CHARRIAGE Y | QS BEDLOAD Y | M2/S |
| `QSSUSP` | QS SUSPENSION | QS SUSPENSION | M2/S |
| `QSSUSPX` | QS SUSPENSION X | QS SUSPENSION X | M2/S |
| `QSSUSPY` | QS SUSPENSION Y | QS SUSPENSION Y | M2/S |
| `HD` | EPAISSEUR DU LIT | BED THICKNESS | M |
| `EF` | FLUX D'EROSION | EROSION FLUX | KG/M2/S |
| `DF` | FLUX DE DEPOT | DEPOSITION FLUX | KG/M2/S |
| `FROTP` | FROT. PEAU | FROT. PEAU | PASCAL |

Note: some IDs appear multiple times with different label variants, as
TELEMAC has used different names for the same variable across versions.

### 3D variables

| ID | French name | English name | Unit |
|----|-------------|--------------|------|
| `Z` | COTE Z | ELEVATION Z | M |
| `U` | VITESSE U | VELOCITY U | M/S |
| `V` | VITESSE V | VELOCITY V | M/S |
| `W` | VITESSE W | VELOCITY W | M/S |
| `NUX` | NUX POUR VITESSE | NUX FOR VELOCITY | M2/S |
| `NUY` | NUY POUR VITESSE | NUY FOR VELOCITY | M2/S |
| `NUZ` | NUZ POUR VITESSE | NUZ FOR VELOCITY | M2/S |
| `K` | ENERGIE TURBULEN | TURBULENT ENERGY | JOULE/KG |
| `EPS` | DISSIPATION | DISSIPATION | WATT/KG |
| `RI` | NB DE RICHARDSON | RICHARDSON NUMB | |
| `RHO` | DENSITE RELATIVE | RELATIVE DENSITY | |
| `DP` | PRESSION DYNAMIQ | DYNAMIC PRESSURE | PA |
| `PH` | PRESSION HYDROST | HYDROSTATIC PRES | PA |
| `UCONV` | U CONVECTION | U ADVECTION | M/S |
| `VCONV` | V CONVECTION | V ADVECTION | M/S |
| `WCONV` | W CONVECTION | W ADVECTION | M/S |
| `DM1` | DM1 | DM1 | |
| `DHHN` | DHHN | DHHN | M |
| `UCONVC` | UCONVC | UCONVC | M/S |
| `VCONVC` | VCONVC | VCONVC | M/S |
| `UD` | UD | UD | M/S |
| `VD` | VD | VD | M/S |
| `WD` | WD | WD | M/S |
| `M` | VITESSE SCALAIRE | SCALAR VELOCITY | M/S |
| `NU` | NU POUR VITESSE | NU FOR VELOCITY | M2/S |

Source tables: `Serafin_var2D.csv` and `Serafin_var3D.csv`, shipped in the
`data/` folder of the underlying [`python-serafin`](https://pypi.org/project/python-serafin/) package.
