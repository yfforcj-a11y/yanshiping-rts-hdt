"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_DIR_BH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'borehole_temperature'))
BOREHOLE_FILES = {'1C': 'yanshiping_rts_borehole_bh1_ground_temperature_20250719_20251017.csv', '2C': 'yanshiping_rts_borehole_bh2_ground_temperature_20250719_20251017.csv', '3C': 'yanshiping_rts_borehole_bh3_ground_temperature_20250719_20251017.csv', '4C': 'yanshiping_rts_borehole_bh4_ground_temperature_20250719_20251017.csv', '5C': 'yanshiping_rts_borehole_bh5_ground_temperature_20250719_20251016.csv'}
TEMP_QC_MIN = -30.0
TEMP_QC_MAX = 30.0
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'kappa_Tinit'))
SURFACE_DEPTH_M = 0.2
TINIT_T0_MODE = 'manual'
TINIT_T0_VALUE = '2025-07-21'
KAPPA_DEPTH_MIN = 0.2
KAPPA_DEPTH_MAX = 3.0
USE_ROLLING_SMOOTH = True
ROLLING_WINDOW_DAYS = 3
PLOT_TINIT_PROFILES = False
RANDOM_SEED = 42
MATERIAL_PROPS = {'clay': {'C': 2800000.0, 'lambda': 1.3}, 'mudstone': {'C': 2400000.0, 'lambda': 2.0}, 'mudstone_frags_ice': {'C': 3000000.0, 'lambda': 2.5}, 'ice_rich_clay': {'C': 3100000.0, 'lambda': 2.3}, 'pure_ice': {'C': 1900000.0, 'lambda': 2.2}}
SIMPLE_LAYERS = {'1C': [(0.0, 2.3, 'clay'), (2.3, 3.5, 'mudstone_frags_ice'), (3.5, 15.0, 'mudstone')], '2C': [(0.0, 2.5, 'clay'), (2.5, 3.4, 'mudstone'), (3.4, 4.7, 'ice_rich_clay'), (4.7, 15.0, 'clay')], '3C': [(0.0, 2.6, 'clay'), (2.6, 5.3, 'mudstone_frags_ice'), (5.3, 5.5, 'pure_ice'), (5.5, 15.0, 'mudstone_frags_ice')], '4C': [(0.0, 1.8, 'clay'), (1.8, 4.5, 'mudstone_frags_ice'), (4.5, 15.0, 'mudstone_frags_ice')], '5C': [(0.0, 2.0, 'clay'), (2.0, 2.5, 'pure_ice'), (2.5, 4.6, 'ice_rich_clay'), (4.6, 7.0, 'clay'), (7.0, 15.0, 'mudstone_frags_ice')]}

def compute_effective_C_kappa_from_simple_layers(borehole_name: str, depth_top: float=0.0, depth_bottom: float=3.0):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if borehole_name not in SIMPLE_LAYERS:
        raise ValueError(f'Public-release status message.{borehole_name}Public-release status message.')
    layers = SIMPLE_LAYERS[borehole_name]
    total_thickness = 0.0
    sum_C_dz = 0.0
    sum_lambda_dz = 0.0
    for (z1, z2, mat_key) in layers:
        z_top = max(z1, depth_top)
        z_bot = min(z2, depth_bottom)
        if z_bot <= z_top:
            continue
        dz = z_bot - z_top
        if dz <= 0:
            continue
        if mat_key not in MATERIAL_PROPS:
            raise ValueError(f'Public-release status message.{mat_key}Public-release status message.')
        C_i = MATERIAL_PROPS[mat_key]['C']
        lam_i = MATERIAL_PROPS[mat_key]['lambda']
        total_thickness += dz
        sum_C_dz += C_i * dz
        sum_lambda_dz += lam_i * dz
    if total_thickness <= 0:
        raise ValueError(f'Public-release status message.{borehole_name}Public-release status message.{depth_top}-{depth_bottom}Public-release status message.')
    C_eff = sum_C_dz / total_thickness
    lambda_eff = sum_lambda_dz / total_thickness
    kappa_eff = lambda_eff / C_eff
    return (C_eff, kappa_eff, lambda_eff)

def load_borehole_daily(file_path: str, time_col: str='\u65f6\u95f4') -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print(f'Public-release status message.{file_path}')
    df = pd.read_csv(file_path, encoding='utf-8')
    if time_col not in df.columns:
        raise ValueError(f'Public-release status message.{file_path}Public-release status message.{time_col}Public-release status message.')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    df_daily = df.resample('D').mean()
    df_daily = df_daily.reset_index().rename(columns={time_col: 'date'})
    depth_cols = []
    for col in df_daily.columns:
        if col == 'date':
            continue
        try:
            float(col)
            depth_cols.append(col)
        except ValueError:
            pass
    depth_cols_sorted = sorted(depth_cols, key=lambda x: float(x))
    if not depth_cols_sorted:
        raise ValueError(f'Public-release status message.{file_path}Public-release status message.')
    print(f'Public-release status message.{len(depth_cols_sorted)}Public-release status message.{depth_cols_sorted[0]} m ~ {depth_cols_sorted[-1]} m')
    keep_cols = ['date'] + depth_cols_sorted
    df_daily = df_daily[keep_cols]
    if USE_ROLLING_SMOOTH and ROLLING_WINDOW_DAYS > 1:
        df_daily = df_daily.set_index('date')
        df_daily[depth_cols_sorted] = df_daily[depth_cols_sorted].rolling(window=ROLLING_WINDOW_DAYS, center=True, min_periods=1).mean()
        df_daily = df_daily.reset_index()
        print(f'Public-release status message.{ROLLING_WINDOW_DAYS}Public-release status message.')
    return df_daily

def build_Tinit_from_raw_daily(df_daily: pd.DataFrame, t0=None, weight_raw: float=0.7, temp_min: float=-40.0, temp_max: float=40.0, smooth_depth: bool=True) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df = df_daily.copy()
    if 'date' not in df.columns:
        raise ValueError('Public-release status message.')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if TINIT_T0_MODE == 'first':
        t0 = df['date'].iloc[0]
    elif TINIT_T0_MODE == 'manual':
        t0 = pd.to_datetime(TINIT_T0_VALUE)
    elif TINIT_T0_MODE == 'auto_best':
        valid_counts = df.drop(columns=['date']).notna().sum(axis=1)
        idx_best = valid_counts.idxmax()
        t0 = df['date'].iloc[idx_best]
    else:
        raise ValueError('Public-release status message.')
    if t0 not in df['date'].values:
        idx_closest = (df['date'] - t0).abs().idxmin()
        t0 = df['date'].iloc[idx_closest]
    depth_cols = []
    depth_vals = []
    for col in df.columns:
        if col == 'date':
            continue
        try:
            z = float(col)
        except Exception:
            continue
        depth_cols.append(col)
        depth_vals.append(z)
    if not depth_cols:
        raise ValueError('Public-release status message.')
    depth_pairs = sorted(zip(depth_vals, depth_cols), key=lambda x: x[0])
    depth_vals = [p[0] for p in depth_pairs]
    depth_cols = [p[1] for p in depth_pairs]
    df_qc = df.copy()
    for col in depth_cols:
        s = df_qc[col]
        mask_bad = (s < TEMP_QC_MIN) | (s > TEMP_QC_MAX)
        if mask_bad.any():
            print(f'Public-release status message.{col}Public-release status message.{mask_bad.sum()}Public-release status message.{TEMP_QC_MIN}, {TEMP_QC_MAX}Public-release status message.')
        df_qc[col] = s.where(~mask_bad, np.nan)
    row_t0 = df_qc.loc[df_qc['date'] == t0]
    if row_t0.empty:
        raise ValueError(f'Public-release status message.{t0}Public-release status message.')
    row_t0 = row_t0.iloc[0]
    T_raw_t0 = []
    T_mean = []
    for col in depth_cols:
        v0 = row_t0[col]
        v_mean = df_qc[col].mean(skipna=True)
        if pd.isna(v_mean):
            v_mean = 0.0
        if pd.isna(v0):
            v0 = v_mean
        v0 = float(v0)
        v_mean = float(v_mean)
        v0 = max(temp_min, min(temp_max, v0))
        v_mean = max(temp_min, min(temp_max, v_mean))
        T_raw_t0.append(v0)
        T_mean.append(v_mean)
    T_raw_t0 = np.array(T_raw_t0, dtype=float)
    T_mean = np.array(T_mean, dtype=float)
    T_init = weight_raw * T_raw_t0 + (1.0 - weight_raw) * T_mean
    if smooth_depth and len(T_init) >= 3:
        T_smooth = T_init.copy()
        for i in range(1, len(T_init) - 1):
            T_smooth[i] = (T_init[i - 1] + T_init[i] + T_init[i + 1]) / 3.0
        T_init = T_smooth
    out = pd.DataFrame({'depth_m': depth_vals, 'T_raw_t0_C': T_raw_t0, 'T_mean_C': T_mean, 'T_init_C': T_init})
    return (out, t0)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for (bh_name, fname) in BOREHOLE_FILES.items():
        print('=' * 72)
        print(f'Public-release status message.{bh_name}')
        bh_path = os.path.join(DATA_DIR_BH, fname)
        print(f'Public-release status message.{bh_path}')
        df_daily = load_borehole_daily(bh_path, time_col='\u65f6\u95f4')
        n_days = len(df_daily)
        print(f"Public-release status message.{n_days}Public-release status message.{df_daily['date'].min()} ~ {df_daily['date'].max()}")
        (Tinit_df, t0) = build_Tinit_from_raw_daily(df_daily, t0=None, weight_raw=0.7, temp_min=-40.0, temp_max=40.0, smooth_depth=True)
        print(f'Public-release status message.{t0}')
        (C_eff, kappa_eff, lambda_eff) = compute_effective_C_kappa_from_simple_layers(borehole_name=bh_name, depth_top=0.0, depth_bottom=3.0)
        print(f'Public-release status message.{bh_name}: C_eff ~= {C_eff:.3e} J/m3/K, kappa_eff ~= {kappa_eff:.3e}Public-release status message.{lambda_eff:.3f} W/m/K')
        kappa_summary_path = os.path.join(OUT_DIR, f'kappa_summary_{bh_name}.csv')
        Tinit_profile_path = os.path.join(OUT_DIR, f'Tinit_profiles_{bh_name}.csv')
        df_kappa = pd.DataFrame({'borehole': [bh_name], 'C_eff_J_m3K': [C_eff], 'kappa_eff_m2_s': [kappa_eff], 'lambda_eff_W_mK': [lambda_eff], 'depth_top_m': [0.0], 'depth_bottom_m': [3.0], 'note': ['Public-release status message.']})
        df_kappa.to_csv(kappa_summary_path, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{kappa_summary_path}')
        Tinit_df.to_csv(Tinit_profile_path, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{Tinit_profile_path}')
        if PLOT_TINIT_PROFILES:
            plt.figure(figsize=(4, 6))
            plt.plot(Tinit_df['T_raw_t0_C'], Tinit_df['depth_m'], label='T_raw_t0')
            plt.plot(Tinit_df['T_mean_C'], Tinit_df['depth_m'], label='T_mean')
            plt.plot(Tinit_df['T_init_C'], Tinit_df['depth_m'], label='T_init')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (deg C)')
            plt.ylabel('Depth (m)')
            plt.title(f'Borehole {bh_name} initial profiles (t0={t0.date()})')
            plt.legend()
            plt.tight_layout()
            plt.show()
    print('=' * 72)
    print('Public-release status message.')
if __name__ == '__main__':
    main()
