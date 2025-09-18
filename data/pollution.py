"""
pollution.py

Synthetic 2-D advection–diffusion dataset generator with:
- Known, spatially varying source term (brick kilns, industries, population, traffic).
- Time-varying wind (Vx, Vy) derived from hourly data and linearly upsampled.
- Open/radiation boundary conditions via an Orlanski-style update (optional sponge).
- Initial condition from kriging of heterogeneous sensor network observations.
- Save routine that mirrors the SWE periodic generator (fields, coords, sensors, meta).

Notes:
- The PDE is solved with scikit-fdiff (skfdiff) using a ROS2 scheme.
- Sources and initial conditions are percentile-normalized (99th) for stability.
- The final NPZ layout matches `sew_periodic.py` style (keys and shapes).
- Variable `bc` is referenced when building the Model; ensure it is defined
  appropriately (e.g., empty dict for open BC handled in the hook), or supply
  skfdiff-style boundary conditions.
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import skfdiff
import time
import pandas as pd
import pytz
from pykrige.uk import UniversalKriging


# In[2]:


def interpolate_array(array, num_interpolations):
    """Linearly upsample a 1D array by inserting evenly-spaced interpolants
    between each pair of consecutive samples.

    Parameters
    ----------
    array : (N,) array-like
        Original samples (e.g., hourly wind components).
    num_interpolations : int
        Number of new points to insert *between* each adjacent pair.

    Returns
    -------
    np.ndarray
        New array with length (N - 1) * (num_interpolations + 1) + 1, where the
        original values are preserved and evenly spaced interpolants are added.

    Rationale
    ---------
    Used to densify Vx/Vy time series so the PDE hook sees smoother winds
    without requiring a smaller physical time step.
    """
    new_array = []
    for i in range(len(array) - 1):
        # Add the current element
        new_array.append(array[i])
        # Generate interpolated values
        interpolated_values = np.linspace(array[i], array[i + 1], num_interpolations + 2)[1:-1]
        new_array.extend(interpolated_values)
    # Add the last element
    new_array.append(array[-1])
    return np.array(new_array)


# In[3]:


def process_row(idx, row):
    """Kriging-based spatial interpolation from irregular sensors to a fixed grid.

    Parameters
    ----------
    idx : Any
        Unused index from DataFrame iteration (kept for signature consistency).
    row : pd.Series
        Sensor readings at a particular timestamp, indexed by monitor IDs.

    Returns
    -------
    vals_grid : np.ma.MaskedArray
        Interpolated grid values over the target lon/lat mesh (masked array).

    Notes
    -----
    - Uses Universal Kriging with a spherical variogram.
    - Filters out NaNs before fitting.
    - Returns a masked array aligned to gridx/gridy defined below.
    """
    x = locs.loc[df.columns]['Longitude'].values
    y = locs.loc[df.columns]['Latitude'].values
    z = row.values

    cols = np.array(df.columns)[~np.isnan(z)]
    x = x[~np.isnan(z)]
    y = y[~np.isnan(z)]
    z = z[~np.isnan(z)]

    UK = UniversalKriging(
        x,
        y,
        z,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
        exact_values = True
    )

    gridx = np.arange(77.01, 77.40, 0.01)
    gridy = np.arange(28.39, 28.78, 0.01)
    vals_grid, ss_grid = UK.execute("grid", gridx, gridy)
    return(vals_grid)


# In[4]:


# --- Wind data (read, align, densify) ----------------------------------------
df_ws = pd.read_csv('wind_speeds.csv', parse_dates=True)
df_ws = df_ws.sort_values(['Timestamp']).reset_index(drop=True)
df_ws = df_ws.set_index(pd.DatetimeIndex(df_ws['Timestamp']))
df_ws = df_ws[['u-component', 'v-component']].groupby('Timestamp').mean()
Vx_array = df_ws['u-component'].to_numpy()
Vy_array = df_ws['v-component'].to_numpy()
Vx_array = interpolate_array(Vx_array, 5)
Vy_array = interpolate_array(Vy_array, 5)
Vx_array = Vx_array.astype(np.float32, copy=False)
Vy_array = Vy_array.astype(np.float32, copy=False)
UPSAMPLE = 5  # keep in sync with interpolate_array()
IST = pytz.FixedOffset(330)
idx = pd.DatetimeIndex(df_ws.index)
if idx.tz is None:
    # assume incoming timestamps are UTC; change if your source is different
    idx = idx.tz_localize("UTC")
idx_ist = idx.tz_convert(IST)

# Coarse (hourly) hours, then densify to match the upsampled wind arrays
hours_coarse = idx_ist.hour.to_numpy()
H_array = np.repeat(hours_coarse[:-1], UPSAMPLE + 1)
H_array = np.concatenate([H_array, [hours_coarse[-1]]])  # length = (N-1)*(UPSAMPLE+1)+1
H_array = H_array[:len(Vx_array)]  # harden against any tiny length mismatch


# In[5]:


# --- Static source layers (combined and cropped to simulation window) ---------
src_dir = './'
brick_kilns = np.load(src_dir+'brick_kilns_intensity_80x80.npy')
industries = np.load(src_dir+'industries_intensity_80x80.npy')
population_density = np.load(src_dir+'population_density_intensity_80x80.npy')
traffic_06 = np.load(src_dir+'traffic_06_intensity_80x80.npy')
traffic_12 = np.load(src_dir+'traffic_12_intensity_80x80.npy')
traffic_18 = np.load(src_dir+'traffic_18_intensity_80x80.npy')
traffic_00 = np.load(src_dir+'traffic_00_intensity_80x80.npy')
traffic = (traffic_06 + traffic_12 + traffic_18 + traffic_00)/4
known_source = (brick_kilns+industries+population_density+traffic)[21:61,16:56]
S_scale = np.percentile(known_source, 99)
S_norm = known_source / (S_scale + 1e-12)
S_norm = S_norm.astype(np.float32, copy=False)


# In[6]:


# --- Observation ingestion (Kaiterra + Govt), timezone handling ---------------
sensor = 'pm25'
res_time = '1H'
filepath_root = './'

filepath_data_gov = filepath_root+'govdata_{}_current.csv'.format(res_time)
filepath_locs_gov = filepath_root+'govdata_locations.csv'

locs = pd.read_csv(filepath_locs_gov, index_col=[0])
data = pd.read_csv(filepath_data_gov, index_col=[0,1], parse_dates=True)[sensor]
data.replace(0,np.nan,inplace=True)

# Normalize timestamps to IST and clip to [start_dt, end_dt]
start_dt = data.index.levels[1][0]
end_dt = data.index.levels[1][-1]

if start_dt.tzname != 'IST':
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('UTC')
        start_dt = start_dt.tz_convert(pytz.FixedOffset(330))

if end_dt.tzname != 'IST':
    if end_dt.tzinfo is None: 
        end_dt = end_dt.tz_localize('UTC')
    end_dt = end_dt.tz_convert(pytz.FixedOffset(330))

# now, filter through the start and end dates
data.sort_index(inplace=True)
data = data.loc[(slice(None), slice(start_dt, end_dt))]


df = data.unstack(level=0)

# --- Misc preprocessing: dropping problematic station --------------
df = df.drop(['Pusa_IMD'], axis=1)


# In[7]:


## Simulation Parameters

# Spatial grid (lon/lat aligned to Delhi window). 40x40 grid from ranges below.
x = np.arange(77.01, 77.40, 0.01)
y = np.arange(28.39, 28.78, 0.01)
X, Y = np.meshgrid(x, y, indexing='ij')
grid_shape = (40, 40)
diffusivity = 5.0  # scalar k in PDE
# right after you define x, y (or inside the solver right at the start)
dx_deg = float(x[1] - x[0])
dy_deg = float(y[1] - y[0])
lat_mean = float((y[0] + y[-1]) * 0.5)

DEG2M_LAT = 111_320.0
DEG2M_LON = 111_320.0 * np.cos(np.deg2rad(lat_mean))

dx_m = dx_deg * DEG2M_LON
dy_m = dy_deg * DEG2M_LAT
dx = dx_m
dy = dy_m
# Map sensor lon/lat to nearest grid indices; de-duplicate final set.
sensor_coords = list(zip(locs['Longitude'].to_numpy(), locs['Latitude'].to_numpy()))
sensor_indices = []
for xc, yc in sensor_coords:
    idx_x = int(round((xc - x[0]) / 0.01))  # Column index
    idx_y = int(round((yc - y[0]) / 0.01))  # Row index
    sensor_indices.append((idx_y, idx_x))

sensor_locs = list(set(sensor_indices))

# Build initial condition from kriging on the first available timestamp
for idx,row in df[0:1].iterrows():
    vals_grid = process_row(idx, row)
initial_conditions = vals_grid.data
IC_scale = np.percentile(initial_conditions, 99)
initial_conditions_norm = initial_conditions / (IC_scale + 1e-12)
initial_conditions_norm = initial_conditions_norm.astype(np.float32, copy=False)


# In[8]:


def apply_open_bc_orlanski(U, U_prev, Vx, Vy, dx, dy, dt, sponge_width=0, sponge_sigma=0.0, U_bg=0.0):
    """Apply Orlanski-style radiation/open boundary conditions to a 2-D field.

    Strategy
    --------
    - Diagnose phase speed along each boundary using c = (∂U/∂t) / (∂U/∂n).
    - Allow only outward transport (V·n > 0); otherwise fall back to zero-gradient.
    - Optional "sponge" layer damps residual reflections near edges.

    Parameters
    ----------
    U : (Ny, Nx) ndarray
        Field at current time step t^n.
    U_prev : (Ny, Nx) ndarray
        Field at previous time step t^{n-1}.
    Vx, Vy : (Ny, Nx) ndarray
        Velocity components at cell centers.
    dx, dy : float
        Grid spacing in x, y.
    dt : float
        Time step.
    sponge_width : int, optional
        Number of edge cells to absorb toward U_bg (0 disables).
    sponge_sigma : float, optional
        Per-step damping factor in sponge region (typ. 0..0.3).
    U_bg : float, optional
        Background toward which sponge relaxes.

    Returns
    -------
    U_bc : (Ny, Nx) ndarray
        Field with updated boundary values.

    Notes
    -----
    This update is invoked inside the skfdiff `hook`, so skfdiff sees open
    boundaries each step without needing explicit BCs in the Model.
    """
    """
    Orlanski-style radiation/open BC:
      U_t + c_n * dU/dn = 0 at the boundary, with c_n diagnosed from interior.
    - Outflow (V·n > 0): use radiation update with c_n >= 0  -> waves exit.
    - Inflow (V·n < 0): no injection; hold value (or zero-gradient fallback).

    Args:
      U       : (Ny, Nx) current field at time n
      U_prev  : (Ny, Nx) previous field at time n-1 (for time derivative)
      Vx, Vy  : (Ny, Nx) velocities at cell centers
      dx, dy  : grid spacings
      dt      : time step
      sponge_width : optional absorbing layer width (cells). 0 disables.
      sponge_sigma : per-step damping strength in sponge (0..0.3 typical)
      U_bg    : background toward which sponge relaxes

    Returns:
      U_bc    : (Ny, Nx) field with open BC applied on the outermost cells
    """
    Ny, Nx = U.shape
    U_bc = U.copy()
    # Fast broadcast if Vx/Vy are scalars
    if np.isscalar(Vx):
        Vx = float(Vx)
        Vx_left  = -Vx
        Vx_right =  Vx
    else:
        Vx = np.asarray(Vx)
        Vx_left, Vx_right = -Vx[:, 0], Vx[:, -1]

    if np.isscalar(Vy):
        Vy = float(Vy)
        Vy_bot  = -Vy
        Vy_top  =  Vy
    else:
        Vy = np.asarray(Vy)
        Vy_bot, Vy_top = -Vy[0, :], Vy[-1, :]

    eps = 1e-12

    # --- helpers to compute Orlanski update on a 1D boundary line ---
    def orlanski_update(Ub, Ubm1, Uint, dn, dt, Vn):
        """
        Ub   : boundary line (length M) at t^n  (U at boundary)
        Ubm1 : boundary line at t^{n-1}
        Uint : neighbor just inside boundary at t^n
        dn   : grid spacing in outward normal direction (dx or dy)
        Vn   : normal velocity at boundary (positive = outflow)
        """
        # diagnose phase speed: c = (∂U/∂t)/(∂U/∂n)
        dUdt = (Ub - Ubm1) / dt
        dUdn = (Ub - Uint) / dn
        c_est = np.divide(dUdt, dUdn, out=np.zeros_like(dUdt), where=np.abs(dUdn) > 1e-12)
        c_eff = np.where(Vn > 0.0, np.clip(c_est, 0.0, np.abs(Vn)), 0.0)
        # only allow outward transport; clip negatives and NaNs
        # c_eff = np.where(Vn > 0.0, np.maximum(0.0, c_est), 0.0)
        # c_eff = np.nan_to_num(c_eff, nan=0.0, posinf=0.0, neginf=0.0)

        # radiation step: Ub^{n+1} ≈ Ub^n - dt * c_eff * (Ub - Uint)/dn
        return Ub - dt * c_eff * (Ub - Uint) / dn

    # LEFT (x=0): outward normal is -x, but we still use neighbor at x=1
    Ub  = U_bc[:, 0]
    Ubm1= U_prev[:, 0]
    Uint= U_bc[:, 1]
    Vn = Vx_left
    U_bc[:, 0] = np.where(Vn > 0.0, orlanski_update(U_bc[:, 0], U_prev[:, 0], U_bc[:, 1], dx, dt, Vn), U_bc[:, 1])

    # RIGHT (x=Nx-1): outward normal is +x, neighbor at x=Nx-2
    Ub  = U_bc[:, -1]
    Ubm1= U_prev[:, -1]
    Uint= U_bc[:, -2]
    Vn = Vx_right
    U_bc[:, -1] = np.where(Vn > 0.0, orlanski_update(U_bc[:, -1], U_prev[:, -1], U_bc[:, -2], dx, dt, Vn), U_bc[:, -2])


    # BOTTOM (y=0): outward normal is -y, neighbor at y=1
    Ub  = U_bc[0, :]
    Ubm1= U_prev[0, :]
    Uint= U_bc[1, :]
    Vn = Vy_bot
    U_bc[0, :] = np.where(Vn > 0.0, orlanski_update(U_bc[0, :], U_prev[0, :], U_bc[1, :], dy, dt, Vn), U_bc[1, :])

    # TOP (y=Ny-1): outward normal is +y, neighbor at y=Ny-2
    Ub  = U_bc[-1, :]
    Ubm1= U_prev[-1, :]
    Uint= U_bc[-2, :]
    Vn = Vy_top
    U_bc[-1, :] = np.where(Vn > 0.0, orlanski_update(U_bc[-1, :], U_prev[-1, :], U_bc[-2, :], dy, dt, Vn), U_bc[-2, :])

    # Optional sponge/absorber (helps kill residual reflections)
    if sponge_width > 0 and sponge_sigma > 0.0:
        w = sponge_width
        ramp_x = np.linspace(1, 0, w)  # 1 at edge -> 0 inside
        ramp_y = np.linspace(1, 0, w)
        # left/right
        U_bc[:, :w]    -= sponge_sigma * ramp_x[np.newaxis, :] * (U_bc[:, :w]    - U_bg)
        U_bc[:, -w:]   -= sponge_sigma * ramp_x[::-1][np.newaxis, :] * (U_bc[:, -w:]   - U_bg)
        # bottom/top
        U_bc[:w, :]    -= sponge_sigma * ramp_y[:, np.newaxis] * (U_bc[:w, :]    - U_bg)
        U_bc[-w:, :]   -= sponge_sigma * ramp_y[::-1][:, np.newaxis] * (U_bc[-w:, :]   - U_bg)

    return U_bc


# In[9]:


# ---------- NumPy finite-difference helpers ----------

def _neighbors_lr_tb(U):
    """Left/Right/Top/Bottom neighbors with Neumann-style edge copy."""
    # shape (Ny, Nx)
    L = np.empty_like(U); L[:, 1:] = U[:, :-1]; L[:, 0]  = U[:, 0]
    R = np.empty_like(U); R[:, :-1] = U[:, 1:]; R[:, -1] = U[:, -1]
    T = np.empty_like(U); T[1:, :] = U[:-1, :]; T[0, :]  = U[0, :]
    B = np.empty_like(U); B[:-1, :] = U[1:, :]; B[-1, :] = U[-1, :]
    return L, R, T, B

def _laplacian(U, dx, dy):
    L, R, T, B = _neighbors_lr_tb(U)
    return (R - 2.0*U + L) / (dx*dx) + (B - 2.0*U + T) / (dy*dy)

def _advection_upwind(U, Vx, Vy, dx, dy):
    """Conservative upwind gradients using one-sided diffs at edges."""
    L, R, T, B = _neighbors_lr_tb(U)
    # backward/forward differences
    dxb = (U - L) / dx; dxf = (R - U) / dx
    dyb = (U - T) / dy; dyf = (B - U) / dy

    if np.isscalar(Vx):
        adv_x = -(Vx * (dxb if Vx >= 0.0 else dxf))
    else:
        Vx = np.asarray(Vx, dtype=U.dtype)
        adv_x = -(np.where(Vx >= 0.0, Vx*dxb, Vx*dxf))

    if np.isscalar(Vy):
        adv_y = -(Vy * (dyb if Vy >= 0.0 else dyf))
    else:
        Vy = np.asarray(Vy, dtype=U.dtype)
        adv_y = -(np.where(Vy >= 0.0, Vy*dyb, Vy*dyf))

    return adv_x + adv_y

def _rhs(U, Vx, Vy, k, S, dx, dy):
    """RHS = -V·∇U + k∇²U + S (advection upwind, diffusion central)."""
    return _advection_upwind(U, Vx, Vy, dx, dy) + k * _laplacian(U, dx, dy) + S


# In[10]:


def numpy_solve_pde_and_collect_data(diffusivity, T, Nt, sponge_width=4, sponge_sigma=0.08):
    """
    Heun (RK2) explicit integrator with Orlanski open BC at every substep.
    Returns:
        U_time_major : (Nt, Ny, Nx)
        sensors_clean: (S, Nt)
    """
    DTYPE = np.float32
    Ny, Nx = initial_conditions_norm.shape
    t = np.linspace(0.0, T, Nt, dtype=DTYPE); dt = float(t[1] - t[0])

    # --- quick explicit CFL-ish guard (heuristic units) ---
    Vmax = float(max(np.max(np.abs(Vx_array)), np.max(np.abs(Vy_array)), 1e-8))
    adv_dt  = 0.5 * min(dx, dy) / Vmax
    diff_dt = 0.24 * min(dx*dx, dy*dy) / max(diffusivity, 1e-8)
    dt_safe = min(adv_dt, diff_dt)
    if dt > dt_safe:
        print(f"[warn] dt={dt:.3e} > dt_safe={dt_safe:.3e}; consider lowering Nt or T.")

    # --- state ---
    U     = initial_conditions_norm.astype(DTYPE, copy=True)
    Uprev = U.copy()
    U_time_major = np.empty((Nt, Ny, Nx), dtype=DTYPE)
    U_time_major[0] = U

    # precompute negatives for S schedule
    S_NEG = (-np.ones_like(S_norm, dtype=DTYPE))

    # vectorized sensor extraction indices
    ij = np.array(sensor_locs, dtype=np.int32)
    iy, ix = ij[:, 0], ij[:, 1]
    sensors_clean = np.empty((len(sensor_locs), Nt), dtype=DTYPE)
    sensors_clean[:, 0] = U[iy, ix]

    # --- time integration ---
    for n in range(Nt - 1):
        t_n = t[n]
        # map physical time to wind index (covers full arrays)
        H = len(Vx_array)  # your prebuilt per-hour or upsampled array
        w_idx = (n * (H - 1)) // (Nt - 1)     # integer 0..H-1
        Vx_t  = float(Vx_array[w_idx]); Vy_t = float(Vy_array[w_idx])
        hour  = int(H_array[w_idx])           # from your hours array
        S_t   = S_NEG if (11 <= hour <= 16) else S_norm
        S_t = S_t.astype(DTYPE, copy=False)

        # Stage A: apply open BC to U, compute f(U)
        U_bc = apply_open_bc_orlanski(U, Uprev, Vx_t, Vy_t, dx, dy, dt,
                                      sponge_width=sponge_width, sponge_sigma=sponge_sigma, U_bg=0.0)
        f0 = _rhs(U_bc, Vx_t, Vy_t, diffusivity, S_t, dx, dy)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)

        # predictor
        U_star = U + dt * f0

        # Stage B: apply open BC to predictor, compute f(U*)
        U_star_bc = apply_open_bc_orlanski(U_star, U, Vx_t, Vy_t, dx, dy, dt,
                                           sponge_width=sponge_width, sponge_sigma=sponge_sigma, U_bg=0.0)
        f1 = _rhs(U_star_bc, Vx_t, Vy_t, diffusivity, S_t, dx, dy)
        f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

        # corrector
        U_next = U + 0.5 * dt * (f0 + f1)
        U_next = np.nan_to_num(U_next, nan=0.0, posinf=0.0, neginf=0.0)

        # roll state
        np.copyto(Uprev, U)
        np.copyto(U, U_next)

        # record
        U_time_major[n+1] = U
        sensors_clean[:, n+1] = U[iy, ix]

    return U_time_major, sensors_clean


# In[11]:


def generate_dataset(save_path='pollution_dataset.npz'):
    """Wrapper to run the solver and save outputs in SWE-style NPZ format.

    Parameters
    ----------
    save_path : str
        Default save path.

    Side Effects
    ------------
    Writes a compressed NPZ to `save_path` (variable referenced below).

    Dataset Layout (keys)
    ---------------------
    - U : (Nx, Ny, Nt) float32      — concentration field (transposed for SWE parity)
    - x, y, t, X, Y                 — coordinates (1D and 2D mesh)
    - params, param_names, bc       — meta
    - rng_seed                      — reproducibility token
    - S                             — normalized source (float32)
    - sensors_idx, sensors_xy       — grid indices and (x,y) of used sensors
    - U_sensor_clean, U_sensor_noisy— sensor time series (clean/noisy)
    - noise_mode, noise_div         — noise config
    """
    # Generate random parameters and solve once
    T = 5.0
    Nt = 2500
    DTYPE = np.float32
    t = np.linspace(0.0, T, Nt, dtype=DTYPE)
    dt = float(t[1] - t[0])

    # ---- Noise settings (like sew_periodic) ----
    NOISE_MODE = "max"   # or "std"
    NOISE_DIV  = 10.0

    # ---- PDE run (get full field + sensor time series) ----
    # Note: time_steps in the solver is number of *steps*, which is Nt-1 for Nt frames
    # U_full_time_major, sensors_clean = solve_pde_and_collect_data(
    #     diffusivity=diffusivity, dt=dt, time_steps=Nt - 1
    # )  # U_full_time_major: (Nt, Ny, Nx) or (Nt+1,...). Above code returns (Nt,Ny,Nx) == len(t).
    U_full_time_major, sensors_clean = numpy_solve_pde_and_collect_data(
    diffusivity=diffusivity, T=T, Nt=Nt, sponge_width=4, sponge_sigma=0.08)

    # ---- Reorder to match SWE field layout (Nx, Ny, Nt) ----
    # SWE saves fields as (Nx, Ny, Nt) with X,Y built as indexing='ij'.
    # In pollution, X,Y came from np.meshgrid(x, y) default (indexing='xy') -> shape (Ny, Nx).
    # U_full is time-major (Nt, Ny, Nx). Convert to (Nx, Ny, Nt).
    U_full_xyz = np.transpose(U_full_time_major, (2, 1, 0)).astype(DTYPE)  # (Nx, Ny, Nt)

    # ---- Sensors: indices & xy (consistent with earlier computed sensor_locs) ----
    # sensor_locs: list of unique (i_y, j_x) indices on the (Ny, Nx) grid
    # Convert to array and also build xy positions
    sensors_idx = np.array(sensor_locs, dtype=np.int32)           # (SENSORS, 2) as (iy, ix)
    sensors_xy  = np.stack([x[sensors_idx[:,1]], y[sensors_idx[:,0]]], axis=1).astype(DTYPE)  # (SENSORS, 2)

    # sensors_clean currently shape (SENSORS, Nt); cast dtype
    sensors_clean = sensors_clean.astype(DTYPE)

    # ---- Add noise (like SWE) ----
    if NOISE_MODE == "std":
        sigma = np.std(sensors_clean)
    elif NOISE_MODE == "max":
        sigma = np.max(np.abs(sensors_clean))
    else:
        raise ValueError("NOISE_MODE must be 'std' or 'max'")
    noise_std = DTYPE(float(sigma) / float(NOISE_DIV))
    rng = np.random.default_rng(42)
    sensors_noisy = sensors_clean + rng.normal(scale=float(noise_std), size=sensors_clean.shape).astype(DTYPE)

    # ---- Params/meta (mirror SWE style) ----
    # Keep key names close in spirit. SWE has: g,H,T,Nx,Ny,Nt,Lx,Ly,dx,dy,dt
    # For pollution, store: k (diffusivity), T, Nt, dx, dy, dt, plus domain extents
    dx = DTYPE(x[1] - x[0])
    dy = DTYPE(y[1] - y[0])
    Lx = DTYPE(x[-1] - x[0] + dx)  # approximate domain span
    Ly = DTYPE(y[-1] - y[0] + dy)
    params = np.array([diffusivity, T, len(x), len(y), Nt, Lx, Ly, dx_deg, dy_deg, dx_m, dy_m, dt], dtype=np.float64)
    param_names = np.array(["k", "T", "Nx", "Ny", "Nt", "Lx", "Ly", "dx_deg", "dy_deg", "dx_m", "dy_m", "dt"], dtype="<U16")

    # ---- Save (npz) using SWE-like keys ----
    # SWE uses: eta, u, v; here we at least save U (concentration).
    # SWE also saves X,Y and various meta.
    np.savez_compressed(
        save_path,
        # fields
        U=U_full_xyz,               # (Nx, Ny, Nt), float32
        # coords
        x=x.astype(DTYPE), y=y.astype(DTYPE), t=t,
        X=X.astype(DTYPE), Y=Y.astype(DTYPE),
        # meta
        params=params,
        param_names=param_names,
        bc=np.array(["open_orlanski"]),
        rng_seed=np.array([42], dtype=np.int64),
        # sources/vel (optional: include for analysis)
        S=S_norm.astype(DTYPE),     # normalized source you built earlier
        # sensors
        sensors_idx=sensors_idx,            # (SENSORS, 2) (iy, ix)
        sensors_xy=sensors_xy,              # (SENSORS, 2) (x,y)
        U_sensor_clean=sensors_clean,       # (SENSORS, Nt)
        U_sensor_noisy=sensors_noisy,       # (SENSORS, Nt)
        noise_mode=np.array(["max"]),
        noise_div=np.array([NOISE_DIV], dtype=np.float32),
    )

    print(f"[SAVE] Wrote dataset to: {save_path}")
    print(f"      U shape: {U_full_xyz.shape}, sensors: {sensors_clean.shape[0]}, noise σ ≈ {float(noise_std):.4g}")
    return


# In[12]:


generate_dataset()