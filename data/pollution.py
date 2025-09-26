#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pytz
from pykrige.uk import UniversalKriging


# In[2]:


# -------------------------------
# Configuration
# -------------------------------
# Domain in lon/lat for Delhi box (used only for IC and sensor mapping)
LON_MIN, LON_MAX = 77.01, 77.40
LAT_MIN, LAT_MAX = 28.39, 28.78

# Grid and time (normalized domain [0,1] x [0,1])
Nx, Ny = 40, 40
T = 20.0                     # 5 sim-seconds = 1 real day
Nt = 10000                   # frames
DTYPE = np.float32

# Wind base (s^-1 on unit box) for 5s=1day scaling
Vx0, Vy0 = 1.12, 0.984      # from 3.5 m/s NE mapped & timescaled

# Diffusivity choice for advection-dominated but numerically friendly regime
# (derived from Kh ~ 20–100 m^2/s mapped and 5s=1day scaling)
diffusivity = 3e-4

# Data sources
sensor = "pm25"
res_time = "1H"
root = "./"
filepath_data_gov = f"{root}govdata_{res_time}_current.csv"
filepath_locs_gov = f"{root}govdata_locations.csv"


# In[3]:


# -------------------------------
# Utilities: wind variations and AR(1)
# -------------------------------

def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ar1(n: int, dt_sim: float, tau_sim: float, sigma: float) -> np.ndarray:
    """AR(1) with e-folding time tau_sim (sim seconds) and stationary std sigma."""
    if tau_sim <= 0 or sigma <= 0 or n <= 1:
        return np.zeros(n, dtype=DTYPE)
    rho = float(np.exp(-dt_sim / tau_sim))
    eps = np.random.normal(size=n).astype(DTYPE)
    out = np.empty(n, dtype=DTYPE)
    out[0] = sigma / np.sqrt(max(1e-12, 1 - rho**2)) * eps[0]
    s = sigma * np.sqrt(max(0.0, 1 - rho**2))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + s * eps[i]
    return out


def monsoon_variations_on_base(
    t_seconds: np.ndarray,
    Vx_base: float = Vx0,
    Vy_base: float = Vy0,
    sim_seconds_per_day: float = 5.0,
    diurnal_amp_frac: float = 0.5,     # ±50% around base magnitude
    speed_noise_frac: float = 0.12,    # AR(1) std as fraction of base speed
    speed_noise_tau_hours: float = 2.0,
    dir_wobble_deg: float = 8.0,
    dir_noise_sigma_deg: float = 5.0,
    dir_noise_tau_hours: float = 3.0,
    peak_hour_local: float = 14.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Vx_steps, Vy_steps) with diurnal and small stochastic variability.
    Keeps mean direction near NE and speed nonnegative.
    """
    t = np.asarray(t_seconds, dtype=np.float64)
    Nt_local = t.size
    if Nt_local == 0:
        return np.array([], DTYPE), np.array([], DTYPE)

    hour_to_simsec = sim_seconds_per_day / 24.0
    dt_sim = float(t[1] - t[0]) if Nt_local > 1 else hour_to_simsec

    V0 = float(np.hypot(Vx_base, Vy_base))
    th0 = float(np.arctan2(Vy_base, Vx_base))

    hours_local = (t / sim_seconds_per_day) * 24.0

    diel = np.cos(2 * np.pi * (hours_local - peak_hour_local) / 24.0).astype(DTYPE)
    speed_det = V0 * (1.0 + diurnal_amp_frac * diel)

    tau_speed_sim = speed_noise_tau_hours * hour_to_simsec
    spd_noise = _ar1(Nt_local, dt_sim, tau_speed_sim, sigma=speed_noise_frac * V0)
    speed = np.maximum(0.0, (speed_det + spd_noise).astype(DTYPE))

    wobble = np.deg2rad(dir_wobble_deg) * np.sin(
        2 * np.pi * (hours_local - (peak_hour_local + 1)) / 24.0
    ).astype(DTYPE)
    tau_dir_sim = dir_noise_tau_hours * hour_to_simsec
    dir_noise = _ar1(Nt_local, dt_sim, tau_dir_sim, sigma=np.deg2rad(dir_noise_sigma_deg))
    theta = _wrap_pi(th0 + wobble + dir_noise).astype(DTYPE)

    Vx = (speed * np.cos(theta)).astype(DTYPE)
    Vy = (speed * np.sin(theta)).astype(DTYPE)
    return Vx, Vy


# In[4]:


# -------------------------------
# Initial condition (IC) from sensors via Universal Kriging
# -------------------------------

def process_row_to_grid(row: pd.Series, nx: int, ny: int) -> np.ndarray:
    """Kriging from irregular sensors to an (nx,ny) lon/lat grid covering the box."""
    x = locs.loc[row.index]["Longitude"].to_numpy()
    y = locs.loc[row.index]["Latitude"].to_numpy()
    z = row.to_numpy()

    # Drop NaNs
    mask = ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]

    UK = UniversalKriging(x, y, z, variogram_model="spherical", verbose=False,
                          enable_plotting=False, exact_values=True)

    gridx = np.linspace(LON_MIN, LON_MAX, nx)
    gridy = np.linspace(LAT_MIN, LAT_MAX, ny)
    vals_grid, _ = UK.execute("grid", gridx, gridy)
    return vals_grid.data.astype(DTYPE)

# Load sensor data and build IC
locs = pd.read_csv(filepath_locs_gov, index_col=0)
raw = pd.read_csv(filepath_data_gov, index_col=[0, 1], parse_dates=True)[sensor]
raw.replace(0, np.nan, inplace=True)

# Normalize to IST and clip to data range
start_dt = raw.index.levels[1][0]
end_dt = raw.index.levels[1][-1]
if start_dt.tzinfo is None:
    start_dt = start_dt.tz_localize("UTC")
start_dt = start_dt.tz_convert(pytz.FixedOffset(330))
if end_dt.tzinfo is None:
    end_dt = end_dt.tz_localize("UTC")
end_dt = end_dt.tz_convert(pytz.FixedOffset(330))

data = raw.sort_index().loc[(slice(None), slice(start_dt, end_dt))]
df = data.unstack(level=0)
# Drop a problematic station (optional)
df = df.drop(["Pusa_IMD"], axis=1, errors="ignore")


# In[7]:


df.iloc[745]


# In[8]:


# First timestamp → IC
first_ts = df.iloc[745]
initial_conditions = process_row_to_grid(first_ts, Nx, Ny)
IC_scale = np.percentile(initial_conditions, 99)
initial_conditions_norm = (initial_conditions / (IC_scale + 1e-12)).astype(DTYPE, copy=False)


# In[9]:


# -------------------------------
# Sources (static, normalized)
# -------------------------------
src_dir = "./"
brick_kilns = np.load(src_dir + "brick_kilns_intensity_80x80.npy")
industries = np.load(src_dir + "industries_intensity_80x80.npy")
population_density = np.load(src_dir + "population_density_intensity_80x80.npy")
traffic_06 = np.load(src_dir + "traffic_06_intensity_80x80.npy")
traffic_12 = np.load(src_dir + "traffic_12_intensity_80x80.npy")
traffic_18 = np.load(src_dir + "traffic_18_intensity_80x80.npy")
traffic_00 = np.load(src_dir + "traffic_00_intensity_80x80.npy")
traffic = (traffic_06 + traffic_12 + traffic_18 + traffic_00) / 4
known_source = (brick_kilns + industries + population_density + traffic)[21:61, 16:56]
S_scale = np.percentile(known_source, 99)
S_norm = (known_source / (S_scale + 1e-12)).astype(DTYPE, copy=False)


# In[10]:


# -------------------------------
# Grid, sensors, wind, and time
# -------------------------------
x = np.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = np.linspace(0.0, 1.0, Ny, dtype=DTYPE)
X, Y = np.meshgrid(x, y, indexing="ij")
dx = 1.0 / (Nx - 1)
dy = 1.0 / (Ny - 1)

t = np.linspace(0.0, T, Nt, dtype=DTYPE)
dt = float(t[1] - t[0])
Vx_steps, Vy_steps = monsoon_variations_on_base(t, Vx_base=Vx0, Vy_base=Vy0)

# Map sensor lon/lat → nearest grid indices
sensor_indices = []
for xc, yc in zip(locs["Longitude"].to_numpy(), locs["Latitude"].to_numpy()):
    x_norm = (xc - LON_MIN) / (LON_MAX - LON_MIN)
    y_norm = (yc - LAT_MIN) / (LAT_MAX - LAT_MIN)
    ix = max(0, min(Nx - 1, int(round(x_norm * (Nx - 1)))))
    iy = max(0, min(Ny - 1, int(round(y_norm * (Ny - 1)))))
    sensor_indices.append((iy, ix))
# de-duplicate
sensor_locs = list(set(sensor_indices))


# In[11]:


# -------------------------------
# Numerical operators
# -------------------------------

def _neighbors_lr_tb(U: np.ndarray):
    L = np.empty_like(U); L[:, 1:] = U[:, :-1]; L[:, 0] = U[:, 0]
    R = np.empty_like(U); R[:, :-1] = U[:, 1:]; R[:, -1] = U[:, -1]
    T = np.empty_like(U); T[1:, :] = U[:-1, :]; T[0, :] = U[0, :]
    B = np.empty_like(U); B[:-1, :] = U[1:, :]; B[-1, :] = U[-1, :]
    return L, R, T, B


def _laplacian(U: np.ndarray, dx: float, dy: float) -> np.ndarray:
    L, R, T, B = _neighbors_lr_tb(U)
    return (R - 2.0 * U + L) / (dx * dx) + (B - 2.0 * U + T) / (dy * dy)


def _advection_upwind(U: np.ndarray, Vx, Vy, dx: float, dy: float) -> np.ndarray:
    L, R, T, B = _neighbors_lr_tb(U)
    dxb = (U - L) / dx; dxf = (R - U) / dx
    dyb = (U - T) / dy; dyf = (B - U) / dy

    if np.isscalar(Vx):
        adv_x = -(Vx * (dxb if Vx >= 0.0 else dxf))
    else:
        Vx = np.asarray(Vx, dtype=U.dtype)
        adv_x = -(np.where(Vx >= 0.0, Vx * dxb, Vx * dxf))

    if np.isscalar(Vy):
        adv_y = -(Vy * (dyb if Vy >= 0.0 else dyf))
    else:
        Vy = np.asarray(Vy, dtype=U.dtype)
        adv_y = -(np.where(Vy >= 0.0, Vy * dyb, Vy * dyf))

    return adv_x + adv_y


def _rhs(U: np.ndarray, Vx, Vy, k: float, S: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return _advection_upwind(U, Vx, Vy, dx, dy) + k * _laplacian(U, dx, dy) + S



# In[12]:


# -------------------------------
# Open (Orlanski) boundary update
# -------------------------------

def apply_open_bc_orlanski(U: np.ndarray, U_prev: np.ndarray, Vx, Vy,
                            dx: float, dy: float, dt: float,
                            sponge_width: int = 0, sponge_sigma: float = 0.0, U_bg: float = 0.0) -> np.ndarray:
    """Apply Orlanski radiation BCs on all edges; optional sponge absorber."""
    U_bc = U.copy()

    if np.isscalar(Vx):
        Vx = float(Vx); Vx_left, Vx_right = -Vx, Vx
    else:
        Vx = np.asarray(Vx); Vx_left, Vx_right = -Vx[:, 0], Vx[:, -1]

    if np.isscalar(Vy):
        Vy = float(Vy); Vy_bot, Vy_top = -Vy, Vy
    else:
        Vy = np.asarray(Vy); Vy_bot, Vy_top = -Vy[0, :], Vy[-1, :]

    def _orlanski_line(Ub, Ubm1, Uint, dn, dt_loc, Vn):
        dUdt = (Ub - Ubm1) / dt_loc
        dUdn = (Ub - Uint) / dn
        c_est = np.divide(dUdt, dUdn, out=np.zeros_like(dUdt), where=np.abs(dUdn) > 1e-12)
        c_eff = np.where(Vn > 0.0, np.clip(c_est, 0.0, np.abs(Vn)), 0.0)
        return Ub - dt_loc * c_eff * (Ub - Uint) / dn

    # left/right
    U_bc[:, 0]  = np.where(Vx_left > 0.0, _orlanski_line(U_bc[:, 0],  U_prev[:, 0],  U_bc[:, 1],  dx, dt, Vx_left), U_bc[:, 1])
    U_bc[:, -1] = np.where(Vx_right > 0.0, _orlanski_line(U_bc[:, -1], U_prev[:, -1], U_bc[:, -2], dx, dt, Vx_right), U_bc[:, -2])
    # bottom/top
    U_bc[0, :]  = np.where(Vy_bot > 0.0, _orlanski_line(U_bc[0, :],  U_prev[0, :],  U_bc[1, :],  dy, dt, Vy_bot), U_bc[1, :])
    U_bc[-1, :] = np.where(Vy_top > 0.0, _orlanski_line(U_bc[-1, :], U_prev[-1, :], U_bc[-2, :], dy, dt, Vy_top), U_bc[-2, :])

    if sponge_width > 0 and sponge_sigma > 0.0:
        w = sponge_width
        rx = np.linspace(1, 0, w); ry = np.linspace(1, 0, w)
        U_bc[:, :w]  -= sponge_sigma * rx[np.newaxis, :]       * (U_bc[:, :w]  - U_bg)
        U_bc[:, -w:] -= sponge_sigma * rx[::-1][np.newaxis, :] * (U_bc[:, -w:] - U_bg)
        U_bc[:w, :]  -= sponge_sigma * ry[:, np.newaxis]       * (U_bc[:w, :]  - U_bg)
        U_bc[-w:, :] -= sponge_sigma * ry[::-1][:, np.newaxis] * (U_bc[-w:, :] - U_bg)

    return U_bc


# In[13]:


# -------------------------------
# Time integration (Heun / RK2)
# -------------------------------

def numpy_solve_pde_and_collect_data(sponge_width: int = 4, sponge_sigma: float = 0.08):
    """Advance U with Heun RK2, applying open BCs each stage; return field & sensors."""
    Vmax = float(max(np.max(np.abs(Vx_steps)), np.max(np.abs(Vy_steps)), 1e-8))
    adv_dt_safe = 0.5 * min(dx, dy) / Vmax
    diff_dt_safe = 0.24 * (min(dx, dy) ** 2) / max(diffusivity, 1e-12)
    dt_safe = min(adv_dt_safe, diff_dt_safe)
    if dt > dt_safe:
        print(f"[warn] dt={dt:.3e} > safe ~{dt_safe:.3e} (adv:{adv_dt_safe:.3e}, diff:{diff_dt_safe:.3e})")

    U = initial_conditions_norm.astype(DTYPE, copy=True)
    Uprev = U.copy()
    U_time_major = np.empty((Nt, Ny, Nx), dtype=DTYPE)
    U_time_major[0] = U

    ij = np.array(sensor_locs, dtype=np.int32)
    iy, ix = ij[:, 0], ij[:, 1]
    sensors_clean = np.empty((len(sensor_locs), Nt), dtype=DTYPE)
    sensors_clean[:, 0] = U[iy, ix]

    for n in range(Nt - 1):
        Vx_t = float(Vx_steps[n]); Vy_t = float(Vy_steps[n])
        S_t = S_norm.astype(DTYPE, copy=False)

        U_bc = apply_open_bc_orlanski(U, Uprev, Vx_t, Vy_t, dx, dy, dt,
                                      sponge_width=sponge_width, sponge_sigma=sponge_sigma, U_bg=0.0)
        f0 = _rhs(U_bc, Vx_t, Vy_t, diffusivity, S_t, dx, dy)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)

        U_star = U + dt * f0
        U_star_bc = apply_open_bc_orlanski(U_star, U, Vx_t, Vy_t, dx, dy, dt,
                                           sponge_width=sponge_width, sponge_sigma=sponge_sigma, U_bg=0.0)
        f1 = _rhs(U_star_bc, Vx_t, Vy_t, diffusivity, S_t, dx, dy)
        f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

        U_next = U + 0.5 * dt * (f0 + f1)
        U_next = np.nan_to_num(U_next, nan=0.0, posinf=0.0, neginf=0.0)

        np.copyto(Uprev, U)
        np.copyto(U, U_next)

        U_time_major[n + 1] = U
        sensors_clean[:, n + 1] = U[iy, ix]

    return U_time_major, sensors_clean


# In[14]:


# -------------------------------
# Save (SWE-like NPZ)
# -------------------------------

def generate_dataset(save_path: str = "pollution_dataset.npz") -> None:
    """Run solver and save NPZ with SWE-style keys."""
    NOISE_MODE = "max"  # or "std"
    NOISE_DIV = 10.0

    U_time_major, sensors_clean = numpy_solve_pde_and_collect_data(sponge_width=4, sponge_sigma=0.08)

    # Time-major (Nt,Ny,Nx) -> SWE layout (Nx,Ny,Nt)
    U_xyz = np.transpose(U_time_major, (2, 1, 0)).astype(DTYPE)

    sensors_idx = np.array(sensor_locs, dtype=np.int32)  # (S,2) (iy,ix)
    sensors_xy = np.stack([x[sensors_idx[:, 1]], y[sensors_idx[:, 0]]], axis=1).astype(DTYPE)
    sensors_clean = sensors_clean.astype(DTYPE)

    # Noise on sensors (SWE-like)
    sigma = np.std(sensors_clean) if NOISE_MODE == "std" else np.max(np.abs(sensors_clean))
    noise_std = DTYPE(float(sigma) / float(NOISE_DIV))
    rng = np.random.default_rng(42)
    sensors_noisy = sensors_clean + rng.normal(scale=float(noise_std), size=sensors_clean.shape).astype(DTYPE)

    # Meta (SWE spirit): k, T, Nx, Ny, Nt, Lx, Ly, dx, dy, dt
    Lx_meta = DTYPE(1.0)
    Ly_meta = DTYPE(1.0)
    params = np.array([diffusivity, T, Nx, Ny, Nt, Lx_meta, Ly_meta, dx, dy, dt], dtype=np.float64)
    param_names = np.array(["k", "T", "Nx", "Ny", "Nt", "Lx", "Ly", "dx", "dy", "dt"], dtype="<U16")

    np.savez_compressed(
        save_path,
        U=U_xyz,
        x=x.astype(DTYPE), y=y.astype(DTYPE), t=t,
        X=X.astype(DTYPE), Y=Y.astype(DTYPE),
        params=params, param_names=param_names,
        bc=np.array(["open_orlanski"]), rng_seed=np.array([42], dtype=np.int64),
        S=S_norm.astype(DTYPE),
        sensors_idx=sensors_idx, sensors_xy=sensors_xy,
        U_sensor_clean=sensors_clean, U_sensor_noisy=sensors_noisy,
        noise_mode=np.array(["max"]), noise_div=np.array([NOISE_DIV], dtype=np.float32),
    )

    print(f"[SAVE] Wrote dataset to: {save_path}")
    print(f"      U shape: {U_xyz.shape}, sensors: {sensors_clean.shape[0]}, noise σ ≈ {float(noise_std):.4g}")


# In[15]:


generate_dataset()


# In[ ]:




