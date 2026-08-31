"""Public-release documentation. Scientific logic and parameters are unchanged."""
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
NFACTOR_DIR: Path = PROJECT_ROOT.parent / 'outputs' / 'n_factor_outputs' / '0'
KAPPA_TINIT_DIR: Path = PROJECT_ROOT.parent / 'outputs' / 'kappa_Tinit'
DATA_DIR_BH: Path = PROJECT_ROOT.parent / 'data' / 'borehole_temperature'
BOREHOLE_FILES: Dict[str, str] = {'1C': 'yanshiping_rts_borehole_bh1_ground_temperature_20250719_20251017.csv', '2C': 'yanshiping_rts_borehole_bh2_ground_temperature_20250719_20251017.csv', '3C': 'yanshiping_rts_borehole_bh3_ground_temperature_20250719_20251017.csv', '4C': 'yanshiping_rts_borehole_bh4_ground_temperature_20250719_20251017.csv', '5C': 'yanshiping_rts_borehole_bh5_ground_temperature_20250719_20251016.csv'}
PINN_OUT_DIR: Path = PROJECT_ROOT.parent / 'outputs' / 'PINN_THERMAL_10A'
PINN_OUT_DIR.mkdir(parents=True, exist_ok=True)
BOREHOLES: List[str] = ['1C', '2C', '3C', '4C', '5C']
Z_TOP_M: float = 0.0
Z_BOTTOM_M: float = 3.0
TIME_START_STR: Optional[str] = None
TIME_END_STR: Optional[str] = None
USE_BH_TEMP_DATA: bool = True
N_BH_DATA_POINTS: int = 2000
PINN_INPUT_DIM: int = 2
PINN_OUTPUT_DIM: int = 1
PINN_HIDDEN_LAYERS: int = 5
PINN_HIDDEN_UNITS: int = 64
NUM_EPOCHS: int = 3000
LR: float = 0.001
WEIGHT_PDE: float = 1.0
WEIGHT_IC: float = 10.0
WEIGHT_BC_TOP: float = 10.0
WEIGHT_BC_BOTTOM: float = 5.0
WEIGHT_DATA: float = 5.0
N_COLLOC_POINTS: int = 5000
N_IC_POINTS: int = 300
N_BC_TOP_POINTS: int = 400
N_BC_BOTTOM_POINTS: int = 400
USE_LBFGS: bool = False
LBFGS_MAX_ITER: int = 500
USE_GPU: bool = True
LOG_INTERVAL: int = 100
ENABLE_PLOTTING: bool = False
PLOT_NUM_DEPTH: int = 60
ENABLE_PLOTTING_OBS = False
PLOT_NUM_DEPTH = 80
PLOT_NUM_LEVELS = 16
HIGHLIGHT_ZERO_ISOTHERM = True
CONTOUR_LABEL_FONTSIZE = 7
COLORBAR_MARGIN = 0.5
TEMP_QC_MIN: float = -30.0
TEMP_QC_MAX: float = 30.0
MATERIAL_PROPS: Dict[str, Dict[str, float]] = {'clay': {'C': 2800000.0, 'lambda': 1.3}, 'mudstone': {'C': 2400000.0, 'lambda': 2.0}, 'mudstone_frags_ice': {'C': 3000000.0, 'lambda': 2.5}, 'ice_rich_clay': {'C': 3100000.0, 'lambda': 2.3}, 'pure_ice': {'C': 1900000.0, 'lambda': 2.2}}
SIMPLE_LAYERS: Dict[str, List[Tuple[float, float, str]]] = {'1C': [(0.0, 2.3, 'clay'), (2.3, 3.5, 'mudstone_frags_ice'), (3.5, 15.0, 'mudstone')], '2C': [(0.0, 2.5, 'clay'), (2.5, 3.4, 'mudstone'), (3.4, 4.7, 'ice_rich_clay'), (4.7, 15.0, 'clay')], '3C': [(0.0, 2.6, 'clay'), (2.6, 5.3, 'mudstone_frags_ice'), (5.3, 5.5, 'pure_ice'), (5.5, 15.0, 'mudstone_frags_ice')], '4C': [(0.0, 1.8, 'clay'), (1.8, 4.5, 'mudstone_frags_ice'), (4.5, 15.0, 'mudstone_frags_ice')], '5C': [(0.0, 2.0, 'clay'), (2.0, 2.5, 'pure_ice'), (2.5, 4.6, 'ice_rich_clay'), (4.6, 7.0, 'clay'), (7.0, 15.0, 'mudstone_frags_ice')]}

def load_Tinit_profile(bh_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    csv_path = KAPPA_TINIT_DIR / f'Tinit_profiles_{bh_name}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path)
    if 'depth_m' not in df.columns or 'T_init_C' not in df.columns:
        raise ValueError(f'Public-release status message.{csv_path}Public-release status message.')
    depth = df['depth_m'].values.astype(float)
    T_init = df['T_init_C'].values.astype(float)
    mask = (depth >= Z_TOP_M) & (depth <= Z_BOTTOM_M)
    depth = depth[mask]
    T_init = T_init[mask]
    order = np.argsort(depth)
    depth_sorted = depth[order]
    T_init_sorted = T_init[order]
    return (depth_sorted, T_init_sorted)

def load_surface_temperature(bh_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    csv_path = NFACTOR_DIR / f'Ta_Ts_degree_days_{bh_name}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path)
    if 'date' not in df.columns or 'Ts_C_daily' not in df.columns:
        raise ValueError(f'Public-release status message.{csv_path}Public-release status message.')
    df['date'] = pd.to_datetime(df['date'])
    if TIME_START_STR is not None:
        df = df[df['date'] >= pd.to_datetime(TIME_START_STR)]
    if TIME_END_STR is not None:
        df = df[df['date'] <= pd.to_datetime(TIME_END_STR)]
    df = df.sort_values('date').reset_index(drop=True)
    dates = df['date'].values.astype('datetime64[D]')
    Ts = df['Ts_C_daily'].values.astype(float)
    return (dates, Ts)

def build_time_scaling(dates: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if len(dates) < 2:
        raise ValueError('Public-release status message.')
    days = dates.astype('datetime64[D]').astype('int64').astype(float)
    t0 = float(days.min())
    t1 = float(days.max())
    if t1 <= t0:
        t1 = t0 + 1.0
    t_scaled = (days - t0) / (t1 - t0)
    return (t_scaled.astype(np.float32), t0, t1)

def depth_to_scaled(z: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    return ((z - Z_TOP_M) / (Z_BOTTOM_M - Z_TOP_M)).astype(np.float32)

def scaled_to_depth(z_scaled: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    return Z_TOP_M + z_scaled * (Z_BOTTOM_M - Z_TOP_M)

def get_kappa_for_depth(bh_name: str, z_m: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if bh_name not in SIMPLE_LAYERS:
        raise ValueError(f'Public-release status message.{bh_name}Public-release status message.')
    layers = SIMPLE_LAYERS[bh_name]
    kappa_z = np.zeros_like(z_m, dtype=float)
    for (i, z) in enumerate(z_m):
        mat_key = layers[0][2]
        for (z_top, z_bot, key) in layers:
            if z >= z_top and z <= z_bot:
                mat_key = key
                break
        if mat_key not in MATERIAL_PROPS:
            raise ValueError(f'Public-release status message.{mat_key}Public-release status message.')
        C = MATERIAL_PROPS[mat_key]['C']
        lam = MATERIAL_PROPS[mat_key]['lambda']
        kappa_z[i] = lam / C
    return kappa_z

def load_borehole_daily(file_path: Path, time_col: str='\u65f6\u95f4') -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print(f'Public-release status message.{file_path}')
    df = pd.read_csv(file_path, encoding='utf-8')
    if time_col not in df.columns:
        raise ValueError(f'Public-release status message.{file_path}Public-release status message.{time_col}Public-release status message.')
    df[time_col] = pd.to_datetime(df[time_col])
    depth_cols: List[str] = []
    for col in df.columns:
        if col == 'date':
            continue
        if col == time_col:
            continue
        try:
            _ = float(col)
            depth_cols.append(col)
        except Exception:
            continue
    for col in depth_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        bad_mask = (df[col] < TEMP_QC_MIN) | (df[col] > TEMP_QC_MAX)
        if bad_mask.any():
            print(f'[BH-QC] {file_path.name}Public-release status message.{col}Public-release status message.{int(bad_mask.sum())}Public-release status message.{TEMP_QC_MIN}, {TEMP_QC_MAX}Public-release status message.')
            df.loc[bad_mask, col] = np.nan
    df = df.set_index(time_col)
    df_daily = df[depth_cols].resample('D').mean()
    df_daily = df_daily.reset_index().rename(columns={time_col: 'date'})
    cols_keep = ['date'] + depth_cols
    df_daily = df_daily[cols_keep]
    for col in depth_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    return df_daily

class MLP_PINN(nn.Module):
    """Public-release documentation. Scientific logic and parameters are unchanged."""

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_units: int):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Public-release documentation. Scientific logic and parameters are unchanged."""
        return self.net(x)

def train_pinn_for_borehole(bh_name: str) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print('=' * 80)
    print(f'Public-release status message.{bh_name}')
    (dates_ts, Ts_daily) = load_surface_temperature(bh_name)
    print(f'Public-release status message.{dates_ts.min()} ~ {dates_ts.max()}Public-release status message.{len(dates_ts)}Public-release status message.')
    (depth_init, T_init_profile) = load_Tinit_profile(bh_name)
    print(f'Public-release status message.{depth_init.min():.2f} ~ {depth_init.max():.2f}Public-release status message.{len(depth_init)}')
    (t_scaled_full, t0_days, t1_days) = build_time_scaling(dates_ts)
    delta_t_days = t1_days - t0_days
    delta_t_seconds = delta_t_days * 24.0 * 3600.0
    print(f'[TIME] t0={t0_days:.0f}, t1={t1_days:.0f}Public-release status message.{delta_t_days:.0f}Public-release status message.')
    z_sample = np.linspace(Z_TOP_M, Z_BOTTOM_M, 101)
    kappa_sample = get_kappa_for_depth(bh_name, z_sample)
    print(f'Public-release status message.{kappa_sample.min():.3e} ~ {kappa_sample.max():.3e}')
    df_bh_daily: Optional[pd.DataFrame] = None
    depth_cols_bh: List[str] = []
    if USE_BH_TEMP_DATA:
        bh_file = BOREHOLE_FILES.get(bh_name, None)
        if bh_file is None:
            raise ValueError(f'Public-release status message.{bh_name}Public-release status message.')
        file_path = DATA_DIR_BH / bh_file
        if not file_path.exists():
            raise FileNotFoundError(f'Public-release status message.{file_path}')
        df_bh_daily = load_borehole_daily(file_path)
        df_bh_daily['date'] = pd.to_datetime(df_bh_daily['date'])
        t_min = dates_ts.min().astype('datetime64[D]')
        t_max = dates_ts.max().astype('datetime64[D]')
        df_bh_daily = df_bh_daily[(df_bh_daily['date'] >= t_min) & (df_bh_daily['date'] <= t_max)].reset_index(drop=True)
        df_ts = pd.DataFrame({'date': pd.to_datetime(dates_ts), 'Ts_C_daily': Ts_daily})
        df_merge = pd.merge(df_ts, df_bh_daily, on='date', how='inner')
        dates_ts = df_merge['date'].values.astype('datetime64[D]')
        Ts_daily = df_merge['Ts_C_daily'].values.astype(float)
        (t_scaled_full, t0_days, t1_days) = build_time_scaling(dates_ts)
        delta_t_days = t1_days - t0_days
        delta_t_seconds = delta_t_days * 24.0 * 3600.0
        depth_cols_bh = [col for col in df_bh_daily.columns if col != 'date']
        df_bh_daily = df_merge[['date'] + depth_cols_bh].copy()
        print(f"Public-release status message.{df_bh_daily['date'].min()} ~ {df_bh_daily['date'].max()}Public-release status message.{len(df_bh_daily)}Public-release status message.{len(depth_cols_bh)}")
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Public-release status message.{device}')
    Lz = Z_BOTTOM_M - Z_TOP_M
    t_c = torch.rand((N_COLLOC_POINTS, 1), dtype=torch.float32, device=device, requires_grad=True)
    z_c = torch.rand((N_COLLOC_POINTS, 1), dtype=torch.float32, device=device, requires_grad=True)
    z_c_m = scaled_to_depth(z_c.detach().cpu().numpy().reshape(-1))
    kappa_c = get_kappa_for_depth(bh_name, z_c_m)
    alpha_c = kappa_c * delta_t_seconds / Lz ** 2
    alpha_c_t = torch.tensor(alpha_c, dtype=torch.float32, device=device).view(-1, 1)
    z_ic_phys = np.linspace(Z_TOP_M, Z_BOTTOM_M, N_IC_POINTS)
    T_ic_vals = np.interp(z_ic_phys, depth_init, T_init_profile)
    z_ic_scaled = depth_to_scaled(z_ic_phys)
    t_ic_scaled = np.zeros_like(z_ic_scaled, dtype=np.float32)
    t_ic_t = torch.tensor(t_ic_scaled, dtype=torch.float32, device=device).view(-1, 1)
    z_ic_t = torch.tensor(z_ic_scaled, dtype=torch.float32, device=device).view(-1, 1)
    T_ic_t = torch.tensor(T_ic_vals, dtype=torch.float32, device=device).view(-1, 1)
    n_top = min(N_BC_TOP_POINTS, len(t_scaled_full))
    idx_top = np.linspace(0, len(t_scaled_full) - 1, n_top).astype(int)
    t_top_scaled = t_scaled_full[idx_top]
    Ts_top_vals = Ts_daily[idx_top]
    t_top_t = torch.tensor(t_top_scaled, dtype=torch.float32, device=device).view(-1, 1)
    z_top_t = torch.zeros_like(t_top_t, device=device)
    Ts_top_t = torch.tensor(Ts_top_vals, dtype=torch.float32, device=device).view(-1, 1)
    T_bottom_init = float(np.interp(Z_BOTTOM_M, depth_init, T_init_profile))
    n_bottom = min(N_BC_BOTTOM_POINTS, len(t_scaled_full))
    idx_bottom = np.linspace(0, len(t_scaled_full) - 1, n_bottom).astype(int)
    t_bottom_scaled = t_scaled_full[idx_bottom]
    t_bottom_t = torch.tensor(t_bottom_scaled, dtype=torch.float32, device=device).view(-1, 1)
    z_bottom_t = torch.ones_like(t_bottom_t, device=device)
    T_bottom_t = torch.full_like(t_bottom_t, T_bottom_init, dtype=torch.float32, device=device)
    t_data_t = None
    z_data_t = None
    T_data_t = None
    if USE_BH_TEMP_DATA and df_bh_daily is not None and (len(depth_cols_bh) > 0):
        dates_bh = df_bh_daily['date'].values.astype('datetime64[D]')
        days_bh = dates_bh.astype('int64').astype(float)
        t_scaled_bh = ((days_bh - t0_days) / (t1_days - t0_days)).astype(np.float32)
        nt_bh = len(df_bh_daily)
        nz_bh = len(depth_cols_bh)
        total_points = nt_bh * nz_bh
        n_sample = min(N_BH_DATA_POINTS, total_points)
        (all_t_idx, all_z_idx) = np.meshgrid(np.arange(nt_bh), np.arange(nz_bh), indexing='ij')
        all_t_idx = all_t_idx.reshape(-1)
        all_z_idx = all_z_idx.reshape(-1)
        perm = np.random.permutation(total_points)[:n_sample]
        sel_t_idx = all_t_idx[perm]
        sel_z_idx = all_z_idx[perm]
        t_data_scaled = t_scaled_bh[sel_t_idx]
        depth_vals = np.array([float(depth_cols_bh[j]) for j in sel_z_idx], dtype=float)
        T_vals = df_bh_daily[depth_cols_bh].values[sel_t_idx, sel_z_idx]
        mask_valid = (depth_vals >= Z_TOP_M) & (depth_vals <= Z_BOTTOM_M)
        if np.any(mask_valid):
            t_data_scaled = t_data_scaled[mask_valid]
            depth_vals = depth_vals[mask_valid]
            T_vals = T_vals[mask_valid]
            z_data_scaled = depth_to_scaled(depth_vals)
            t_data_t = torch.tensor(t_data_scaled, dtype=torch.float32, device=device).view(-1, 1)
            z_data_t = torch.tensor(z_data_scaled, dtype=torch.float32, device=device).view(-1, 1)
            T_data_t = torch.tensor(T_vals, dtype=torch.float32, device=device).view(-1, 1)
            print(f'Public-release status message.{t_data_t.shape[0]}')
        else:
            print('Public-release status message.')
            t_data_t = None
            z_data_t = None
            T_data_t = None
    model = MLP_PINN(in_dim=PINN_INPUT_DIM, out_dim=PINN_OUTPUT_DIM, hidden_layers=PINN_HIDDEN_LAYERS, hidden_units=PINN_HIDDEN_UNITS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    print('Public-release status message.')
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        input_c = torch.cat([t_c, z_c], dim=1)
        T_c = model(input_c)
        T_t = torch.autograd.grad(outputs=T_c, inputs=t_c, grad_outputs=torch.ones_like(T_c), retain_graph=True, create_graph=True)[0]
        T_z = torch.autograd.grad(outputs=T_c, inputs=z_c, grad_outputs=torch.ones_like(T_c), retain_graph=True, create_graph=True)[0]
        T_zz = torch.autograd.grad(outputs=T_z, inputs=z_c, grad_outputs=torch.ones_like(T_z), retain_graph=True, create_graph=True)[0]
        pde_res = T_t - alpha_c_t * T_zz
        loss_pde = torch.mean(pde_res ** 2)
        input_ic = torch.cat([t_ic_t, z_ic_t], dim=1)
        T_ic_pred = model(input_ic)
        loss_ic = mse_loss(T_ic_pred, T_ic_t)
        input_top = torch.cat([t_top_t, z_top_t], dim=1)
        T_top_pred = model(input_top)
        loss_bc_top = mse_loss(T_top_pred, Ts_top_t)
        input_bottom = torch.cat([t_bottom_t, z_bottom_t], dim=1)
        T_bottom_pred = model(input_bottom)
        loss_bc_bottom = mse_loss(T_bottom_pred, T_bottom_t)
        if USE_BH_TEMP_DATA and t_data_t is not None and (z_data_t is not None) and (T_data_t is not None):
            input_data = torch.cat([t_data_t, z_data_t], dim=1)
            T_data_pred = model(input_data)
            loss_data = mse_loss(T_data_pred, T_data_t)
        else:
            loss_data = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss_total = WEIGHT_PDE * loss_pde + WEIGHT_IC * loss_ic + WEIGHT_BC_TOP * loss_bc_top + WEIGHT_BC_BOTTOM * loss_bc_bottom + WEIGHT_DATA * loss_data
        loss_total.backward()
        optimizer.step()
        if epoch == 1 or epoch % LOG_INTERVAL == 0 or epoch == NUM_EPOCHS:
            print(f'[Adam] Epoch {epoch:5d} | Total={loss_total.item():.4e} | PDE={loss_pde.item():.4e} | IC={loss_ic.item():.4e} | BC_top={loss_bc_top.item():.4e} | BC_bottom={loss_bc_bottom.item():.4e} | Data={loss_data.item():.4e}')
    print('Public-release status message.')
    if USE_LBFGS:
        print('Public-release status message.')

        def closure():
            optimizer_lbfgs.zero_grad()
            input_c_loc = torch.cat([t_c, z_c], dim=1)
            T_c_loc = model(input_c_loc)
            T_t_loc = torch.autograd.grad(outputs=T_c_loc, inputs=t_c, grad_outputs=torch.ones_like(T_c_loc), retain_graph=True, create_graph=True)[0]
            T_z_loc = torch.autograd.grad(outputs=T_c_loc, inputs=z_c, grad_outputs=torch.ones_like(T_c_loc), retain_graph=True, create_graph=True)[0]
            T_zz_loc = torch.autograd.grad(outputs=T_z_loc, inputs=z_c, grad_outputs=torch.ones_like(T_z_loc), retain_graph=True, create_graph=True)[0]
            pde_res_loc = T_t_loc - alpha_c_t * T_zz_loc
            loss_pde_loc = torch.mean(pde_res_loc ** 2)
            input_ic_loc = torch.cat([t_ic_t, z_ic_t], dim=1)
            T_ic_pred_loc = model(input_ic_loc)
            loss_ic_loc = mse_loss(T_ic_pred_loc, T_ic_t)
            input_top_loc = torch.cat([t_top_t, z_top_t], dim=1)
            T_top_pred_loc = model(input_top_loc)
            loss_bc_top_loc = mse_loss(T_top_pred_loc, Ts_top_t)
            input_bottom_loc = torch.cat([t_bottom_t, z_bottom_t], dim=1)
            T_bottom_pred_loc = model(input_bottom_loc)
            loss_bc_bottom_loc = mse_loss(T_bottom_pred_loc, T_bottom_t)
            if USE_BH_TEMP_DATA and t_data_t is not None and (z_data_t is not None) and (T_data_t is not None):
                input_data_loc = torch.cat([t_data_t, z_data_t], dim=1)
                T_data_pred_loc = model(input_data_loc)
                loss_data_loc = mse_loss(T_data_pred_loc, T_data_t)
            else:
                loss_data_loc = torch.tensor(0.0, dtype=torch.float32, device=device)
            loss_total_loc = WEIGHT_PDE * loss_pde_loc + WEIGHT_IC * loss_ic_loc + WEIGHT_BC_TOP * loss_bc_top_loc + WEIGHT_BC_BOTTOM * loss_bc_bottom_loc + WEIGHT_DATA * loss_data_loc
            loss_total_loc.backward()
            return loss_total_loc
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_MAX_ITER, tolerance_grad=1e-08, tolerance_change=1e-12, line_search_fn='strong_wolfe')
        optimizer_lbfgs.step(closure)
        print('Public-release status message.')
    model.eval()
    with torch.no_grad():
        nt = len(dates_ts)
        t_scaled_full_torch = torch.tensor(t_scaled_full, dtype=torch.float32, device=device).view(-1, 1)
        z_scaled_grid = np.linspace(0.0, 1.0, PLOT_NUM_DEPTH).astype(np.float32)
        z_scaled_grid_torch = torch.tensor(z_scaled_grid, dtype=torch.float32, device=device).view(-1, 1)
        t_flat = t_scaled_full_torch.repeat_interleave(PLOT_NUM_DEPTH, dim=0)
        z_flat = z_scaled_grid_torch.repeat(nt, 1)
        input_flat = torch.cat([t_flat, z_flat], dim=1)
        T_flat = model(input_flat).cpu().numpy().reshape(nt, PLOT_NUM_DEPTH)
        z_vec_m = scaled_to_depth(z_scaled_grid)
    out_npz_path = PINN_OUT_DIR / f'PINN_Tfield_10A_{bh_name}.npz'
    np.savez(out_npz_path, z_vec_m=z_vec_m, t_vec=dates_ts, T_pred_C=T_flat)
    print(f'Public-release status message.{out_npz_path}')
    valid_global = np.isfinite(T_flat)
    if valid_global.any():
        vmin_global = float(np.nanpercentile(T_flat[valid_global], 5)) - COLORBAR_MARGIN
        vmax_global = float(np.nanpercentile(T_flat[valid_global], 95)) + COLORBAR_MARGIN
    else:
        (vmin_global, vmax_global) = (-5.0, 10.0)
    if ENABLE_PLOTTING:
        print('Public-release status message.')
        dates_pd = pd.to_datetime(dates_ts.astype('datetime64[D]'))
        t_num = dates_pd.map(pd.Timestamp.toordinal).values.astype(float)
        T_pinn = T_flat
        (vmin, vmax) = (vmin_global, vmax_global)
        (T_mesh, Z_mesh) = np.meshgrid(t_num, z_vec_m, indexing='ij')
        levels = np.linspace(vmin, vmax, PLOT_NUM_LEVELS)
        (fig, ax) = plt.subplots(figsize=(7, 4))
        cf = ax.contourf(T_mesh, Z_mesh, T_pinn, levels=levels, cmap='viridis', extend='both')
        cs = ax.contour(T_mesh, Z_mesh, T_pinn, levels=levels, colors='k', linewidths=0.3, alpha=0.6)
        try:
            ax.clabel(cs, inline=True, fmt='%.1f', fontsize=CONTOUR_LABEL_FONTSIZE)
        except Exception:
            pass
        if HIGHLIGHT_ZERO_ISOTHERM and vmin < 0.0 < vmax:
            try:
                cs0 = ax.contour(T_mesh, Z_mesh, T_pinn, levels=[0.0], colors='white', linewidths=1.3)
                ax.clabel(cs0, fmt='0 deg C', fontsize=CONTOUR_LABEL_FONTSIZE + 1)
            except Exception:
                pass
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Temperature (deg C)')
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')
        ax.set_title(f'PINN T-field 10A - {bh_name}')
        if len(dates_pd) >= 6:
            tick_idx = np.linspace(0, len(dates_pd) - 1, 6).astype(int)
        else:
            tick_idx = np.arange(len(dates_pd))
        tick_vals = t_num[tick_idx]
        tick_labels = [str(dates_pd[i].date()) for i in tick_idx]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels, rotation=30, ha='right')
        fig.tight_layout()
        out_png_path = PINN_OUT_DIR / f'PINN_Tfield_10A_{bh_name}.png'
        fig.savefig(out_png_path, dpi=300)
        plt.close(fig)
        print(f'Public-release status message.{out_png_path}')
    print(f'Public-release status message.{bh_name}Public-release status message.')
    if ENABLE_PLOTTING_OBS:
        print('Public-release status message.')
        (vmin, vmax) = (vmin_global, vmax_global)
        df_obs = df_bh_daily.copy()
        df_obs['date'] = pd.to_datetime(df_obs['date'])
        df_obs = df_obs.sort_values('date').reset_index(drop=True)
        dates_pd_obs = df_obs['date']
        t_num_obs = dates_pd_obs.map(pd.Timestamp.toordinal).values.astype(float)
        depth_cols = []
        depth_vals = []
        for col in df_obs.columns:
            if col == 'date':
                continue
            try:
                depth_vals.append(float(col))
                depth_cols.append(col)
            except Exception:
                continue
        depth_vals = np.array(depth_vals, dtype=float)
        sort_idx = np.argsort(depth_vals)
        depth_vals = depth_vals[sort_idx]
        depth_cols_sorted = [depth_cols[i] for i in sort_idx]
        T_obs = df_obs[depth_cols_sorted].values.astype(float)
        (nt_obs, Nz_obs) = T_obs.shape
        print(f'Public-release status message.{nt_obs}, Nz={Nz_obs}')
        levels_obs = np.linspace(vmin, vmax, PLOT_NUM_LEVELS)
        (T_mesh_obs, Z_mesh_obs) = np.meshgrid(t_num_obs, depth_vals, indexing='ij')
        (fig, ax) = plt.subplots(figsize=(7, 4))
        cf_obs = ax.contourf(T_mesh_obs, Z_mesh_obs, T_obs, levels=levels_obs, cmap='viridis', extend='both')
        cs_obs = ax.contour(T_mesh_obs, Z_mesh_obs, T_obs, levels=levels_obs, colors='k', linewidths=0.3, alpha=0.6)
        try:
            ax.clabel(cs_obs, inline=True, fmt='%.1f', fontsize=CONTOUR_LABEL_FONTSIZE)
        except Exception:
            pass
        if HIGHLIGHT_ZERO_ISOTHERM and vmin < 0.0 < vmax:
            try:
                cs0_obs = ax.contour(T_mesh_obs, Z_mesh_obs, T_obs, levels=[0.0], colors='white', linewidths=1.3)
                ax.clabel(cs0_obs, fmt='0 deg C', fontsize=CONTOUR_LABEL_FONTSIZE + 1)
            except Exception:
                pass
        cbar_obs = fig.colorbar(cf_obs, ax=ax)
        cbar_obs.set_label('Temperature (deg C)')
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')
        ax.set_title(f'Observed T-field (daily) - {bh_name}')
        if len(dates_pd_obs) >= 6:
            tick_idx_obs = np.linspace(0, len(dates_pd_obs) - 1, 6).astype(int)
        else:
            tick_idx_obs = np.arange(len(dates_pd_obs))
        tick_vals_obs = t_num_obs[tick_idx_obs]
        tick_labels_obs = [str(dates_pd_obs.iloc[i].date()) for i in tick_idx_obs]
        ax.set_xticks(tick_vals_obs)
        ax.set_xticklabels(tick_labels_obs, rotation=30, ha='right')
        fig.tight_layout()
        out_png_path_obs = PINN_OUT_DIR / f'OBS_Tfield_daily_{bh_name}.png'
        fig.savefig(out_png_path_obs, dpi=300)
        plt.close(fig)
        print(f'Public-release status message.{out_png_path_obs}')
        mask_03 = depth_vals <= 3.0
        depth_vals_03 = depth_vals[mask_03]
        T_obs_03 = T_obs[:, mask_03]
        if depth_vals_03.size > 0:
            (T_mesh_obs_03, Z_mesh_obs_03) = np.meshgrid(t_num_obs, depth_vals_03, indexing='ij')
            (fig2, ax2) = plt.subplots(figsize=(7, 4))
            cf_obs_03 = ax2.contourf(T_mesh_obs_03, Z_mesh_obs_03, T_obs_03, levels=levels_obs, cmap='viridis', extend='both')
            cs_obs_03 = ax2.contour(T_mesh_obs_03, Z_mesh_obs_03, T_obs_03, levels=levels_obs, colors='k', linewidths=0.3, alpha=0.6)
            try:
                ax2.clabel(cs_obs_03, inline=True, fmt='%.1f', fontsize=CONTOUR_LABEL_FONTSIZE)
            except Exception:
                pass
            if HIGHLIGHT_ZERO_ISOTHERM and vmin < 0.0 < vmax:
                try:
                    cs0_obs_03 = ax2.contour(T_mesh_obs_03, Z_mesh_obs_03, T_obs_03, levels=[0.0], colors='white', linewidths=1.3)
                    ax2.clabel(cs0_obs_03, fmt='0 deg C', fontsize=CONTOUR_LABEL_FONTSIZE + 1)
                except Exception:
                    pass
            cbar_obs_03 = fig2.colorbar(cf_obs_03, ax=ax2)
            cbar_obs_03.set_label('Temperature (deg C)')
            ax2.invert_yaxis()
            ax2.set_ylabel('Depth (m)')
            ax2.set_xlabel('Date')
            ax2.set_title(f'Public-release status message.{bh_name}')
            if len(dates_pd_obs) >= 6:
                tick_idx_obs = np.linspace(0, len(dates_pd_obs) - 1, 6).astype(int)
            else:
                tick_idx_obs = np.arange(len(dates_pd_obs))
            tick_vals_obs = t_num_obs[tick_idx_obs]
            tick_labels_obs = [str(dates_pd_obs.iloc[i].date()) for i in tick_idx_obs]
            ax2.set_xticks(tick_vals_obs)
            ax2.set_xticklabels(tick_labels_obs, rotation=30, ha='right')
            fig2.tight_layout()
            out_png_path_obs_03 = PINN_OUT_DIR / f'OBS_Tfield_daily_0to3m_{bh_name}.png'
            fig2.savefig(out_png_path_obs_03, dpi=300)
            plt.close(fig2)
            print(f'Public-release status message.{out_png_path_obs_03}')

def main() -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print('=' * 80)
    print('Public-release status message.')
    print(f'Public-release status message.{NFACTOR_DIR}')
    print(f'Public-release status message.{KAPPA_TINIT_DIR}')
    print(f'Public-release status message.{DATA_DIR_BH}')
    print(f'Public-release status message.{PINN_OUT_DIR}')
    print('=' * 80)
    for bh in BOREHOLES:
        try:
            train_pinn_for_borehole(bh)
        except Exception as e:
            print(f'Public-release status message.{bh}Public-release status message.{e}')
    print('=' * 80)
    print('Public-release status message.')
    print('=' * 80)
if __name__ == '__main__':
    main()
