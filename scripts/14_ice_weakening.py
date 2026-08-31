"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import re
import glob
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
BASE_OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
PINN_OUT_DIR = BASE_OUT_DIR / 'PINN_THERMAL_10A'
EVENT_OUT_DIR = BASE_OUT_DIR / 'EVENT_OUT'
GEOMORPH_CSV = BASE_OUT_DIR / 'PATCH' / 'BH_GNSS_geomorph_features.csv'
EVENT_CSV = EVENT_OUT_DIR / 'event_times_v5.csv'
OUT_DIR = PINN_OUT_DIR / 'ICE_WEAKENING'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENABLE_PLOTTING = False
DT_MUSHY_C = 1.0
L_VOL_ICE = 306000000.0
DEFAULT_ICE_VOL_FRAC = {'pure_ice': 0.98, 'ice_rich_clay': 0.6, 'mudstone_frags_ice': 0.35, 'clay': 0.18, 'mudstone': 0.05}
DEFAULT_ICE_VOL_FRAC_FALLBACK = 0.12

def try_import_layer_config_from_10A() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    candidate = Path(__file__).with_name('10_pinn_thermal.py')
    if not candidate.exists():
        return (None, None)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('pinn10A', str(candidate))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        layers = getattr(mod, 'SIMPLE_LAYERS', None)
        props = getattr(mod, 'MATERIAL_PROPS', None)
        if isinstance(layers, dict) and isinstance(props, dict):
            return (layers, props)
        return (None, None)
    except Exception:
        return (None, None)

def fallback_material_props_and_layers() -> Tuple[Dict, Dict]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    material_props = {'clay': {'C': 2500000.0, 'lambda': 1.5}, 'mudstone': {'C': 2200000.0, 'lambda': 2.0}, 'mudstone_frags_ice': {'C': 2000000.0, 'lambda': 1.8}, 'ice_rich_clay': {'C': 2300000.0, 'lambda': 1.7}, 'pure_ice': {'C': 1900000.0, 'lambda': 2.2}}
    simple_layers = {'1C': [(0.0, 3.0, 'clay')], '2C': [(0.0, 3.0, 'clay')], '3C': [(0.0, 3.0, 'clay')], '4C': [(0.0, 3.0, 'clay')], '5C': [(0.0, 3.0, 'clay')]}
    return (simple_layers, material_props)
(SIMPLE_LAYERS, MATERIAL_PROPS) = try_import_layer_config_from_10A()
if SIMPLE_LAYERS is None or MATERIAL_PROPS is None:
    (SIMPLE_LAYERS, MATERIAL_PROPS) = fallback_material_props_and_layers()

def smoothstep(x: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def melt_fraction_from_T(Tc: np.ndarray, dt_mushy: float=DT_MUSHY_C) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    x = (Tc + dt_mushy) / max(dt_mushy, 1e-06)
    return smoothstep(x)

def depth_of_isotherm(Tz: np.ndarray, z: np.ndarray, iso: float=0.0) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if len(Tz) != len(z):
        raise ValueError('Public-release status message.')
    if np.all(Tz >= iso):
        return float(np.nanmax(z))
    if np.all(Tz <= iso):
        return 0.0
    for i in range(len(z) - 1):
        a = Tz[i] - iso
        b = Tz[i + 1] - iso
        if a == 0:
            return float(z[i])
        if a * b < 0:
            (za, zb) = (z[i], z[i + 1])
            return float(za + (0 - a) * (zb - za) / (b - a))
    return float(np.nan)

def build_profile_arrays_for_bh(bh: str, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    C_z = np.full_like(z, np.nan, dtype=float)
    lam_z = np.full_like(z, np.nan, dtype=float)
    ice0_z = np.full_like(z, np.nan, dtype=float)
    layers = SIMPLE_LAYERS.get(bh, None)
    if layers is None or len(layers) == 0:
        layers = [(float(np.nanmin(z)), float(np.nanmax(z)), 'clay')]
    for (ztop, zbot, mat) in layers:
        mask = (z >= ztop) & (z <= zbot + 1e-12)
        prop = MATERIAL_PROPS.get(mat, None)
        if prop is None:
            C_val = 2300000.0
            lam_val = 1.6
        else:
            C_val = float(prop.get('C', 2300000.0))
            lam_val = float(prop.get('lambda', 1.6))
        ice_val = float(DEFAULT_ICE_VOL_FRAC.get(mat, DEFAULT_ICE_VOL_FRAC_FALLBACK))
        C_z[mask] = C_val
        lam_z[mask] = lam_val
        ice0_z[mask] = ice_val
    C_z = np.where(np.isfinite(C_z), C_z, np.nanmedian(C_z[np.isfinite(C_z)]) if np.any(np.isfinite(C_z)) else 2300000.0)
    lam_z = np.where(np.isfinite(lam_z), lam_z, np.nanmedian(lam_z[np.isfinite(lam_z)]) if np.any(np.isfinite(lam_z)) else 1.6)
    ice0_z = np.where(np.isfinite(ice0_z), ice0_z, np.nanmedian(ice0_z[np.isfinite(ice0_z)]) if np.any(np.isfinite(ice0_z)) else DEFAULT_ICE_VOL_FRAC_FALLBACK)
    return (C_z, lam_z, ice0_z)

def integrate_over_depth(y: np.ndarray, z: np.ndarray, z_max: Optional[float]=None) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if z_max is None:
        zz = z
        yy = y
    else:
        mask = z <= z_max + 1e-12
        zz = z[mask]
        yy = y[mask]
        if len(zz) < 2:
            return float(np.nan)
    return float(np.trapz(yy, zz))

def compute_indices_timeseries(bh: str, npz_path: Path, z_max_for_integrals: float=3.0) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    data = np.load(npz_path, allow_pickle=True)
    z = np.array(data['z_vec_m'], dtype=float).reshape(-1)
    t_vec = data['t_vec']
    T = np.array(data['T_pred_C'], dtype=float)
    order = np.argsort(z)
    z = z[order]
    T = T[:, order]
    if z_max_for_integrals is not None:
        mask = z <= z_max_for_integrals + 1e-12
        if mask.sum() >= 2:
            z_use = z[mask]
            T_use = T[:, mask]
        else:
            z_use = z
            T_use = T
    else:
        z_use = z
        T_use = T
    (C_z, lam_z, ice0_z) = build_profile_arrays_for_bh(bh, z_use)
    dates = pd.to_datetime(np.array(t_vec).astype('datetime64[D]'))
    posT_int = []
    z0C_list = []
    ALT_list = []
    melt_col = []
    for k in range(T_use.shape[0]):
        Tk = T_use[k, :]
        posT = np.maximum(Tk, 0.0)
        posT_int.append(integrate_over_depth(posT, z_use, z_max=None))
        z0c = depth_of_isotherm(Tk, z_use, iso=0.0)
        z0C_list.append(z0c)
        ALT_list.append(z0c)
        fm = melt_fraction_from_T(Tk, dt_mushy=DT_MUSHY_C)
        melt_col.append(integrate_over_depth(ice0_z * fm, z_use, z_max=None))
    posT_int = np.array(posT_int, dtype=float)
    z0C_arr = np.array(z0C_list, dtype=float)
    ALT_arr = np.array(ALT_list, dtype=float)
    melt_col = np.array(melt_col, dtype=float)
    date_series = pd.to_datetime(np.array(t_vec).astype('datetime64[D]'))
    dt_days = np.diff(date_series.values).astype('timedelta64[D]').astype(float)
    dt_days = np.where(dt_days <= 0, np.nan, dt_days)
    d_posT_dt = np.full_like(posT_int, np.nan, dtype=float)
    d_melt_dt = np.full_like(melt_col, np.nan, dtype=float)
    d_posT_dt[1:] = np.diff(posT_int) / dt_days
    d_melt_dt[1:] = np.diff(melt_col) / dt_days
    dT_dt = np.full_like(T_use, np.nan, dtype=float)
    dT = np.diff(T_use, axis=0)
    dT_dt[1:, :] = dT / dt_days[:, None]
    sensible_rate = []
    for k in range(T_use.shape[0]):
        if k == 0:
            sensible_rate.append(np.nan)
        else:
            sensible_rate.append(integrate_over_depth(C_z * dT_dt[k, :], z_use, z_max=None))
    sensible_rate = np.array(sensible_rate, dtype=float)
    latent_rate = L_VOL_ICE * d_melt_dt
    enthalpy_rate_proxy = sensible_rate + latent_rate
    df = pd.DataFrame({'date': dates, 'bh': bh, 'zmax_m': float(np.nanmax(z_use)), 'posT_thickness_int_Cm': posT_int, 'z0C_m': z0C_arr, 'ALT_m': ALT_arr, 'd_posT_dt_Cm_per_day': d_posT_dt, 'melt_frac_col_m': melt_col, 'd_melt_col_dt_m_per_day': d_melt_dt, 'sensible_rate_Jm2_per_day': sensible_rate, 'latent_rate_Jm2_per_day': latent_rate, 'enthalpy_rate_proxy_Jm2_per_day': enthalpy_rate_proxy, 'DT_MUSHY_C': DT_MUSHY_C, 'L_VOL_ICE_Jm3': L_VOL_ICE})
    return df

def load_event_times(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    for c in ['t_trigger', 't_failure']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def summarize_over_window(df_ts: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, prefix: str) -> Dict[str, float]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    sub = df_ts[(df_ts['date'] >= t0) & (df_ts['date'] <= t1)].copy()
    out = {}
    if sub.empty:
        for col in ['posT_thickness_int_Cm', 'ALT_m', 'z0C_m', 'melt_frac_col_m', 'd_melt_col_dt_m_per_day', 'enthalpy_rate_proxy_Jm2_per_day']:
            out[f'{prefix}_{col}_mean'] = np.nan
            out[f'{prefix}_{col}_max'] = np.nan
        out[f'{prefix}_n'] = 0
        return out
    for col in ['posT_thickness_int_Cm', 'ALT_m', 'z0C_m', 'melt_frac_col_m', 'd_melt_col_dt_m_per_day', 'enthalpy_rate_proxy_Jm2_per_day']:
        out[f'{prefix}_{col}_mean'] = float(np.nanmean(sub[col].values))
        out[f'{prefix}_{col}_max'] = float(np.nanmax(sub[col].values))
    out[f'{prefix}_n'] = int(len(sub))
    return out

def event_aligned_summary(df_ts_all: pd.DataFrame, df_events: pd.DataFrame, pre_days: int=7, post_days: int=7, during_fallback_days: int=3) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df_events = df_events.copy()
    df_events['event_id'] = np.arange(len(df_events)) + 1
    out_rows = []
    bhs = sorted(df_ts_all['bh'].unique().tolist())
    for (_, ev) in df_events.iterrows():
        t_tr = ev.get('t_trigger', pd.NaT)
        t_fa = ev.get('t_failure', pd.NaT)
        if pd.isna(t_tr):
            continue
        if pd.isna(t_fa):
            t_fa = t_tr + pd.Timedelta(days=during_fallback_days)
        pre0 = t_tr - pd.Timedelta(days=pre_days)
        pre1 = t_tr - pd.Timedelta(days=1)
        dur0 = t_tr
        dur1 = t_fa
        post0 = t_fa + pd.Timedelta(days=1)
        post1 = t_fa + pd.Timedelta(days=post_days)
        for bh in bhs:
            dfts = df_ts_all[df_ts_all['bh'] == bh].copy()
            row = {'event_id': int(ev['event_id']), 'sensor': ev.get('sensor', ''), 'type': ev.get('type', ''), 't_trigger': t_tr, 't_failure': t_fa, 'bh': bh, 'pre_days': pre_days, 'post_days': post_days, 'during_fallback_days': during_fallback_days}
            row.update(summarize_over_window(dfts, pre0, pre1, 'pre'))
            row.update(summarize_over_window(dfts, dur0, dur1, 'during'))
            row.update(summarize_over_window(dfts, post0, post1, 'post'))
            row['delta_during_pre_posT_mean'] = row['during_posT_thickness_int_Cm_mean'] - row['pre_posT_thickness_int_Cm_mean']
            row['delta_during_pre_meltcol_mean'] = row['during_melt_frac_col_m_mean'] - row['pre_melt_frac_col_m_mean']
            row['delta_during_pre_enthalpy_mean'] = row['during_enthalpy_rate_proxy_Jm2_per_day_mean'] - row['pre_enthalpy_rate_proxy_Jm2_per_day_mean']
            out_rows.append(row)
    return pd.DataFrame(out_rows)

def parse_pair_to_dates(pair: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    m = re.match('^\\s*(\\d{6})-(\\d{6})\\s*$', str(pair))
    if not m:
        return None
    (a, b) = (m.group(1), m.group(2))
    y1 = int('20' + a[:2])
    m1 = int(a[2:4])
    d1 = int(a[4:6])
    y2 = int('20' + b[:2])
    m2 = int(b[2:4])
    d2 = int(b[4:6])
    try:
        tA = pd.Timestamp(y1, m1, d1)
        tB = pd.Timestamp(y2, m2, d2)
        t0 = min(tA, tB)
        t1 = max(tA, tB)
        return (t0, t1)
    except Exception:
        return None

def pair_aligned_summary(df_ts_all: pd.DataFrame, geomorph_csv: Path) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not geomorph_csv.exists():
        return pd.DataFrame()
    df_geo = pd.read_csv(geomorph_csv, encoding='utf-8-sig')
    if 'pair' not in df_geo.columns or 'site_id' not in df_geo.columns:
        return pd.DataFrame()
    bhs = sorted(df_ts_all['bh'].unique().tolist())
    df_geo_bh = df_geo[df_geo['site_id'].astype(str).isin(bhs)].copy()
    if df_geo_bh.empty:
        return pd.DataFrame()
    df_key = df_geo_bh[['site_id', 'pair']].drop_duplicates().reset_index(drop=True)
    rows = []
    for (_, r) in df_key.iterrows():
        bh = str(r['site_id'])
        pair = str(r['pair'])
        pr = parse_pair_to_dates(pair)
        if pr is None:
            continue
        (t0, t1) = pr
        dfts = df_ts_all[df_ts_all['bh'] == bh]
        sub = dfts[(dfts['date'] >= t0) & (dfts['date'] <= t1)]
        if sub.empty:
            rows.append({'site_id': bh, 'pair': pair, 'pair_t0': t0, 'pair_t1': t1, 'n': 0, 'posT_mean': np.nan, 'ALT_mean': np.nan, 'meltcol_mean': np.nan, 'enthalpy_mean': np.nan})
            continue
        rows.append({'site_id': bh, 'pair': pair, 'pair_t0': t0, 'pair_t1': t1, 'n': int(len(sub)), 'posT_mean': float(np.nanmean(sub['posT_thickness_int_Cm'])), 'posT_max': float(np.nanmax(sub['posT_thickness_int_Cm'])), 'ALT_mean': float(np.nanmean(sub['ALT_m'])), 'ALT_max': float(np.nanmax(sub['ALT_m'])), 'meltcol_mean': float(np.nanmean(sub['melt_frac_col_m'])), 'meltcol_max': float(np.nanmax(sub['melt_frac_col_m'])), 'enthalpy_mean': float(np.nanmean(sub['enthalpy_rate_proxy_Jm2_per_day'])), 'enthalpy_max': float(np.nanmax(sub['enthalpy_rate_proxy_Jm2_per_day']))})
    df_pair = pd.DataFrame(rows)
    df_pair_join = df_geo.merge(df_pair, on=['site_id', 'pair'], how='left')
    df_pair_join_out = OUT_DIR / 'ICE_WEAKENING_join_geomorph_by_pair.csv'
    df_pair_join.to_csv(df_pair_join_out, index=False, encoding='utf-8-sig')
    return df_pair

def discover_bh_npz(pinn_dir: Path) -> Dict[str, Path]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    out = {}
    patt = str(pinn_dir / 'PINN_Tfield_10A_*.npz')
    for fp in glob.glob(patt):
        p = Path(fp)
        name = p.stem.replace('PINN_Tfield_10A_', '')
        out[name] = p
    return out

def main():
    print('Public-release status message.')
    print(f'[INFO] PINN_OUT_DIR = {PINN_OUT_DIR}')
    print(f'[INFO] EVENT_CSV    = {EVENT_CSV}')
    print(f'[INFO] GEOMORPH_CSV = {GEOMORPH_CSV}')
    bh_npz = discover_bh_npz(PINN_OUT_DIR)
    if not bh_npz:
        raise FileNotFoundError(f'Public-release status message.{PINN_OUT_DIR}\\PINN_Tfield_10A_*.npz')
    print(f'Public-release status message.{len(bh_npz)}Public-release status message.{sorted(list(bh_npz.keys()))}')
    ts_all = []
    for (bh, npz_path) in bh_npz.items():
        print(f'[BH] {bh} -> {npz_path}')
        df_ts = compute_indices_timeseries(bh=bh, npz_path=npz_path, z_max_for_integrals=3.0)
        out_ts = OUT_DIR / f'ICE_WEAKENING_timeseries_{bh}.csv'
        df_ts.to_csv(out_ts, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{out_ts}')
        ts_all.append(df_ts)
    df_ts_all = pd.concat(ts_all, axis=0, ignore_index=True)
    out_all = OUT_DIR / 'ICE_WEAKENING_timeseries_ALL.csv'
    df_ts_all.to_csv(out_all, index=False, encoding='utf-8-sig')
    print(f'Public-release status message.{out_all}')
    df_prior = pd.DataFrame([{'material_key': k, 'ice_vol_frac': v} for (k, v) in DEFAULT_ICE_VOL_FRAC.items()])
    df_prior['DT_MUSHY_C'] = DT_MUSHY_C
    df_prior['L_VOL_ICE_Jm3'] = L_VOL_ICE
    out_prior = OUT_DIR / 'ICE_WEAKENING_prior_config_used.csv'
    df_prior.to_csv(out_prior, index=False, encoding='utf-8-sig')
    print(f'Public-release status message.{out_prior}')
    if EVENT_CSV.exists():
        df_events = load_event_times(EVENT_CSV)
        df_ev = event_aligned_summary(df_ts_all, df_events, pre_days=7, post_days=7, during_fallback_days=3)
        out_ev = OUT_DIR / 'ICE_WEAKENING_event_aligned_v5.csv'
        df_ev.to_csv(out_ev, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{out_ev}')
    else:
        print(f'Public-release status message.{EVENT_CSV}Public-release status message.')
    df_pair = pair_aligned_summary(df_ts_all, GEOMORPH_CSV)
    if not df_pair.empty:
        out_pair = OUT_DIR / 'ICE_WEAKENING_pair_aligned.csv'
        df_pair.to_csv(out_pair, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{out_pair}')
        print(f"Public-release status message.{OUT_DIR / 'ICE_WEAKENING_join_geomorph_by_pair.csv'}")
    else:
        print('Public-release status message.')
    print('Public-release status message.')
if __name__ == '__main__':
    main()
