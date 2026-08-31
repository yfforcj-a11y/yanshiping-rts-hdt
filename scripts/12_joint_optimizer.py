"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import re
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT_DIR, 'outputs')
BH_GNSS_LFJ_CSV = os.path.join(ROOT_DIR, 'data', 'site_catalog.csv')
GNSS_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'gnss')
TFIELD_DIR = os.path.join(OUT_DIR, 'PINN_THERMAL_10A')
TFIELD_NPZ_PATTERN = 'PINN_Tfield_10A_{BH}.npz'
LFJ_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'crack_meter')
LFJ_FILES = {'3A9': os.path.join(LFJ_DATA_DIR, 'yanshiping_rts_crack_meter_c1_20250720_20251126.csv'), '3D3': os.path.join(LFJ_DATA_DIR, 'yanshiping_rts_crack_meter_c2_20250720_20251126.csv')}
LFJ_TO_GNSS = {'3A9': '9286', '3D3': '7704'}
LFJ_COL_TIME_DEV = '\u8bbe\u5907\u65f6\u95f4'
LFJ_COL_TIME_SYS = '\u7cfb\u7edf\u65f6\u95f4'
LFJ_COL_OPEN = '\u62c9\u7ebf\u53d8\u5316'
LFJ_COL_ACC = '\u52a0\u901f\u5ea6'
LFJ_COL_TILT_D = '\u503e\u89d2\u53d8\u5316'
LFJ_OPEN_ZERO_EPS = 1e-06
LFJ_ZERO_STREAK_N = 3
USE_LFJ_COUPLING = True
LFJ_GATES = {'7704': {'F': 0.05, 'G': 0.05}, '9286': {'F': 1.0, 'G': 1.0}, '7627': {'F': 0.0, 'G': 0.0}}
GEOMORPH_CSV = os.path.join(OUT_DIR, 'PATCH', 'BH_GNSS_geomorph_features.csv')
OUT_10B_DIR = os.path.join(OUT_DIR, '10B_THERMO_SLUMP_FINAL')
os.makedirs(OUT_10B_DIR, exist_ok=True)
USE_GNSS_IDS = ['7704', '9286', '7627']
CP_ENABLE_9286_RAMP = True
CP_ENABLE_9286 = True
CP_9286_T0 = '2025-09-12'
CP_9286_T1 = '2025-10-15'
CP_K_ON = 0.8
CP_K_OFF = 0.8
GNSS_FILENAME_REGEX = {'7704': re.compile('.*gn2_.*\\.csv$', re.IGNORECASE), '9286': re.compile('.*gn1_.*\\.csv$', re.IGNORECASE), '7627': re.compile('.*gn3_.*\\.csv$', re.IGNORECASE)}
COL_TIME = '\u91c7\u96c6\u65f6\u95f4'
COL_H_CUM = 'H\u65b9\u5411\u7d2f\u8ba1(mm)'
RESAMPLE_RULE = 'D'
SUBSIDENCE_POSITIVE = True
THERMO_INDEX_MODE = 'signed'
WEIGHT_EACH_STATION_EQUAL = True
SUBSIDENCE_POSITIVE_BY_SITE = {'9286': True, '7627': False, '7704': False}
FREEZE_POINT_C = 0.0
SENSE_LAYER_MIN_M = 0.0
SENSE_LAYER_MAX_M = 4.0
INTEGRATION_DT_UNIT = 'days'
PAIR_DATE_FORMAT = '%Y-%m-%d'
CAND_COL_SITE_ID = ['site_id', 'ID', 'id', 'site', 'SiteID']
CAND_COL_PAIR = ['pair', 'PAIR', 'time_pair', 'pair_id']
CAND_COL_LOD_SIG = ['LOD_sig', 'lod_sig', 'LOD95_sig', 'sig_lod95']
CAND_COL_DZ_MEAN_LOCAL = ['dz_mean_local(m)', 'dz_mean_local', 'dz_mean_local_m']
CAND_COL_DZ_POINT = ['dz_point(m)', 'dz_point', 'dz_point_m']
CAND_COL_DIST_PATCH = ['dist_to_patch_m', 'dist_to_patch', 'dist_m', 'nearest_patch_dist_m']
CAND_COL_SCM_PROB = ['scm_prob_local', 'SCM_prob_local', 'scm_prob', 'prob_local']
CAND_COL_SCM_BIN = ['scm_bin_local', 'SCM_bin_local', 'scm_bin', 'bin_local']
USE_SCM_GATE_FOR_DEBRIS = True
SCM_PROB_MIN_FOR_DEBRIS = 0.3
ALLOW_DEBRIS_WITHOUT_SCM = True
USE_LOD_GATE_FOR_EROSION = True
USE_LOD_GATE_FOR_DEBRIS = True
SHEAR_RETRO_COUPLING = 1.0
SITE_GATES = {'7704': {'I': 1.0, 'E': 0.05, 'L': 0.05, 'S': 0.05}, '9286': {'I': 0.4, 'E': 1.0, 'L': 0.2, 'S': 0.35}, '7627': {'I': 1.0, 'E': 0.03, 'L': 0.03, 'S': 0.03}}
RIDGE_LAMBDA = 0.01
USE_PINN = True
PINN_EPOCHS = 1500
PINN_LR = 0.005
PINN_L2 = 0.001
PINN_SMOOTHNESS = 0.0
USE_DETREND_TARGET = True
DETREND_WIN = 25
FORCE_MONOTONIC_BY_ROLE = True
SAVE_PER_STATION_CSV = True
SAVE_PER_STATION_PNG = True
SAVE_GLOBAL_CSV = True
PLOT_DPI = 300

def safe_read_csv(csv_path: str) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    for enc in ['utf-8-sig', 'gbk', 'cp936', 'utf-8']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f'Public-release status message.{enc}Public-release status message.{csv_path}')
            return df
        except Exception:
            continue
    raise UnicodeDecodeError('unknown', b'', 0, 1, f'Public-release status message.{csv_path}')

def pick_first_existing_col(df: pd.DataFrame, candidates):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_slash3_to_norm(x):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not isinstance(x, str):
        return np.nan
    parts = str(x).strip().split('/')
    if len(parts) != 3:
        return np.nan
    try:
        (a, b, c) = [float(p) for p in parts]
        return float(np.sqrt(a * a + b * b + c * c))
    except Exception:
        return np.nan

def detect_lfj_t_init(df, open_col):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    arr = pd.to_numeric(df[open_col], errors='coerce').values
    ok = np.isfinite(arr) & (np.abs(arr) <= LFJ_OPEN_ZERO_EPS)
    if ok.sum() == 0:
        return None
    streak = 0
    for i in range(len(ok)):
        if ok[i]:
            streak += 1
            if streak >= LFJ_ZERO_STREAK_N:
                return i - LFJ_ZERO_STREAK_N + 1
        else:
            streak = 0
    return int(np.where(ok)[0][0])

def load_lfj_series(lfj_id: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    fp = LFJ_FILES.get(lfj_id, None)
    if fp is None or not os.path.exists(fp):
        print(f'Public-release status message.{lfj_id}Public-release status message.{fp}')
        out = pd.DataFrame(index=target_index)
        out['open_mm'] = 0.0
        out['open_rate'] = 0.0
        out['tilt_norm'] = 0.0
        out['tilt_cum'] = 0.0
        return out
    df = safe_read_csv(fp)
    tcol = LFJ_COL_TIME_DEV if LFJ_COL_TIME_DEV in df.columns else LFJ_COL_TIME_SYS if LFJ_COL_TIME_SYS in df.columns else None
    if tcol is None:
        raise KeyError(f'[LFJ] {lfj_id}Public-release status message.')
    df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
    df = df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    if LFJ_COL_OPEN not in df.columns:
        raise KeyError(f'[LFJ] {lfj_id}Public-release status message.{LFJ_COL_OPEN}')
    i0 = detect_lfj_t_init(df.reset_index(), LFJ_COL_OPEN)
    if i0 is not None:
        t0 = df.index[i0]
        df = df.loc[df.index >= t0].copy()
        print(f'[LFJ] {lfj_id} t_init = {t0}')
    else:
        print(f'[LFJ] {lfj_id}Public-release status message.')
    open_mm = pd.to_numeric(df[LFJ_COL_OPEN], errors='coerce').fillna(0.0)
    open_mm = open_mm.clip(lower=0.0)
    if LFJ_COL_TILT_D in df.columns:
        tilt_norm = df[LFJ_COL_TILT_D].apply(parse_slash3_to_norm)
        tilt_norm = pd.to_numeric(tilt_norm, errors='coerce').fillna(0.0)
    else:
        tilt_norm = pd.Series(0.0, index=df.index)
    open_mm_D = open_mm.resample(RESAMPLE_RULE).median().interpolate()
    tilt_D = tilt_norm.resample(RESAMPLE_RULE).median().interpolate()
    open_mm_A = open_mm_D.reindex(target_index).interpolate(method='time').fillna(0.0)
    tilt_A = tilt_D.reindex(target_index).interpolate(method='time').fillna(0.0)
    dt_days = target_index.to_series().diff().dt.total_seconds().fillna(0.0) / 86400.0
    dopen = open_mm_A.diff().fillna(0.0)
    open_rate = (dopen / dt_days.replace(0.0, np.nan)).fillna(0.0)
    tilt_rate_ser = pd.Series(np.abs(tilt_A.values), index=target_index, name=f'{lfj_id}_tilt_rate')
    tilt_cum = integrate_rate_to_cumulative(tilt_rate_ser)
    out = pd.DataFrame(index=target_index)
    out['open_mm'] = open_mm_A.values
    out['open_rate'] = open_rate.values
    out['tilt_norm'] = tilt_A.values
    out['tilt_cum'] = tilt_cum.values
    return out

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def time_gate_window(t: pd.DatetimeIndex, t0: str, t1: str, k_on: float, k_off: float):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    tt = pd.to_datetime(t).astype('int64') / 1000000000.0
    t0s = pd.Timestamp(t0).value / 1000000000.0
    t1s = pd.Timestamp(t1).value / 1000000000.0
    day = 86400.0
    g_on = _sigmoid((tt - t0s) / day * float(k_on))
    g_off = _sigmoid((tt - t1s) / day * float(k_off))
    g_win = g_on * (1.0 - g_off)
    return (g_on, g_off, g_win)

def haversine_m(lon1, lat1, lon2, lat2) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    R = 6371000.0
    rad = math.pi / 180.0
    (lon1, lat1, lon2, lat2) = (lon1 * rad, lat1 * rad, lon2 * rad, lat2 * rad)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_gnss_to_bh_mapping() -> dict:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df = safe_read_csv(BH_GNSS_LFJ_CSV)
    col_id = pick_first_existing_col(df, ['ID', 'id', 'site_id'])
    col_num = pick_first_existing_col(df, ['Num', 'num', 'TYPE', 'type'])
    col_lon = pick_first_existing_col(df, ['Longitude', 'longitude', 'lon', '\u7ecf\u5ea6'])
    col_lat = pick_first_existing_col(df, ['Latitude', 'latitude', 'lat', '\u7eac\u5ea6'])
    if any((x is None for x in [col_id, col_num, col_lon, col_lat])):
        raise ValueError('Public-release status message.')
    df['_id'] = df[col_id].astype(str)
    df['_num'] = df[col_num].astype(str)
    df['_lon'] = pd.to_numeric(df[col_lon], errors='coerce')
    df['_lat'] = pd.to_numeric(df[col_lat], errors='coerce')
    df_gnss = df[df['_num'].str.upper().str.startswith('GNSS')].copy()
    df_bh = df[df['_num'].str.upper().str.startswith('BH')].copy()
    if df_bh[['_lon', '_lat']].isna().any().any():
        raise ValueError('Public-release status message.')
    if df_gnss[['_lon', '_lat']].isna().any().any():
        raise ValueError('Public-release status message.')
    mapping = {}
    mapping_dist = {}
    print('Public-release status message.')
    for gid in USE_GNSS_IDS:
        gsel = df_gnss[df_gnss['_id'] == str(gid)]
        if len(gsel) == 0:
            raise ValueError(f'Public-release status message.{gid}Public-release status message.')
        g = gsel.iloc[0]
        best_bh = None
        best_d = None
        for (_, b) in df_bh.iterrows():
            d = haversine_m(g['_lon'], g['_lat'], b['_lon'], b['_lat'])
            if best_d is None or d < best_d:
                best_d = d
                best_bh = b['_id']
        mapping[gid] = best_bh
        mapping_dist[gid] = float(best_d)
        print(f'  GNSS {gid}Public-release status message.{best_bh}Public-release status message.{best_d:.1f} m)')
    out_map = os.path.join(OUT_10B_DIR, 'GNSS_to_BH_nearest_mapping.json')
    with open(out_map, 'w', encoding='utf-8') as f:
        json.dump({'mapping': mapping, 'distance_m': mapping_dist}, f, ensure_ascii=False, indent=2)
    print(f'Public-release status message.{out_map}')
    return mapping

def find_gnss_file(gnss_id: str) -> str:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    pat = GNSS_FILENAME_REGEX[str(gnss_id)]
    cands = [fn for fn in os.listdir(GNSS_DATA_DIR) if pat.search(fn)]
    if len(cands) == 0:
        raise FileNotFoundError(f'Public-release status message.{gnss_id}Public-release status message.{GNSS_DATA_DIR}Public-release status message.')
    if len(cands) > 1:
        cands = sorted(cands, key=lambda x: len(x), reverse=True)
    return os.path.join(GNSS_DATA_DIR, cands[0])

def load_gnss_vertical_cum_m(gnss_id: str) -> pd.Series:
    fp = find_gnss_file(gnss_id)
    print(f'Public-release status message.{gnss_id}Public-release status message.{fp}')
    df = safe_read_csv(fp)
    if COL_TIME not in df.columns:
        raise KeyError(f'Public-release status message.{COL_TIME}Public-release status message.')
    if COL_H_CUM not in df.columns:
        raise KeyError(f'Public-release status message.{COL_H_CUM}Public-release status message.')
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors='coerce')
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME).set_index(COL_TIME)
    w_mm = pd.to_numeric(df[COL_H_CUM], errors='coerce').dropna()
    w_m = w_mm.resample(RESAMPLE_RULE).mean().interpolate() / 1000.0
    subs_pos = SUBSIDENCE_POSITIVE
    if isinstance(SUBSIDENCE_POSITIVE_BY_SITE, dict) and str(gnss_id) in SUBSIDENCE_POSITIVE_BY_SITE:
        v = SUBSIDENCE_POSITIVE_BY_SITE[str(gnss_id)]
        if v is not None:
            subs_pos = bool(v)
    if subs_pos:
        w_m = -w_m
    w_m = w_m - w_m.iloc[0]
    return w_m

def load_tfield_npz(bh_id: str) -> dict:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    fp = os.path.join(TFIELD_DIR, TFIELD_NPZ_PATTERN.format(BH=bh_id))
    if not os.path.exists(fp):
        raise FileNotFoundError(f'Public-release status message.{fp}Public-release status message.')
    print(f'Public-release status message.{bh_id}Public-release status message.{fp}')
    data = np.load(fp, allow_pickle=True)
    keys = list(data.keys())
    t_key = 't_vec' if 't_vec' in keys else 't' if 't' in keys else None
    z_key = 'z_vec_m' if 'z_vec_m' in keys else 'z_vec' if 'z_vec' in keys else None
    T_key = 'T_pred_C' if 'T_pred_C' in keys else 'T_pred' if 'T_pred' in keys else None
    if any((k is None for k in [t_key, z_key, T_key])):
        raise KeyError(f'Public-release status message.{keys}')
    t = pd.to_datetime(data[t_key])
    z = np.array(data[z_key], dtype=float)
    T = np.array(data[T_key], dtype=float)
    if T.ndim != 2:
        raise ValueError(f'Public-release status message.{T.shape}Public-release status message.')
    if T.shape[0] != len(t) or T.shape[1] != len(z):
        if T.shape[0] == len(z) and T.shape[1] == len(t):
            T = T.T
        else:
            raise ValueError(f'Public-release status message.{T.shape}, t={len(t)}, z={len(z)}')
    return {'t': t, 'z': z, 'T': T}

def compute_IT_star_and_dITdt(bh_id: str, target_index: pd.DatetimeIndex) -> (pd.Series, pd.Series):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    d = load_tfield_npz(bh_id)
    t = d['t']
    z = d['z']
    T = d['T']
    z_abs = np.abs(z)
    idx = np.where((z_abs >= SENSE_LAYER_MIN_M) & (z_abs <= SENSE_LAYER_MAX_M))[0]
    if len(idx) < 2:
        raise ValueError(f'Public-release status message.{len(idx)}Public-release status message.')
    z_sel = z_abs[idx]
    dz = np.diff(z_sel)
    dz = np.append(dz, dz[-1])
    T_sel = T[:, idx]
    if THERMO_INDEX_MODE.lower() == 'signed':
        thermo_field = T_sel - FREEZE_POINT_C
    else:
        thermo_field = np.maximum(T_sel - FREEZE_POINT_C, 0.0)
    IT = (thermo_field * dz.reshape(1, -1)).sum(axis=1)
    IT = pd.Series(IT, index=pd.DatetimeIndex(t)).sort_index()
    IT_aligned = IT.reindex(target_index).interpolate(method='time')
    IT_star = IT_aligned - IT_aligned.iloc[0]
    dt_days = IT_star.index.to_series().diff().dt.total_seconds().fillna(0.0) / 86400.0
    dIT = IT_star.diff().fillna(0.0)
    dITdt = (dIT / dt_days.replace(0.0, np.nan)).fillna(0.0)
    return (IT_star, dITdt)

def weighted_ridge_fit(X: np.ndarray, y: np.ndarray, lam: float, w: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    sw = np.sqrt(w).reshape(-1, 1)
    Xw = X * sw
    yw = y.reshape(-1, 1) * sw
    p = X.shape[1]
    A = Xw.T @ Xw + lam * np.eye(p)
    b = Xw.T @ yw
    gamma = np.linalg.solve(A, b).reshape(-1)
    return gamma

def _parse_yyMMdd_to_ts(s: str) -> pd.Timestamp:
    s = (s or '').strip()
    if len(s) != 6:
        return pd.NaT
    y = int('20' + s[0:2])
    m = int(s[2:4])
    d = int(s[4:6])
    try:
        return pd.Timestamp(year=y, month=m, day=d)
    except Exception:
        return pd.NaT

def parse_pair_late_date(pair_str: str) -> pd.Timestamp:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not isinstance(pair_str, str) or '-' not in pair_str:
        return pd.NaT
    (a, b) = pair_str.split('-', 1)
    ta = _parse_yyMMdd_to_ts(a)
    tb = _parse_yyMMdd_to_ts(b)
    if pd.isna(ta) or pd.isna(tb):
        return pd.NaT
    return max(ta, tb)

def compute_pair_dt_days(pair_str: str) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not isinstance(pair_str, str) or '-' not in pair_str:
        return np.nan
    (a, b) = pair_str.split('-', 1)
    ta = _parse_yyMMdd_to_ts(a)
    tb = _parse_yyMMdd_to_ts(b)
    if pd.isna(ta) or pd.isna(tb):
        return np.nan
    dt = abs((ta - tb).total_seconds() / 86400.0)
    return float(dt) if dt > 0 else 1.0

def load_geomorph_table() -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df = safe_read_csv(GEOMORPH_CSV)
    col_site = pick_first_existing_col(df, CAND_COL_SITE_ID)
    col_pair = pick_first_existing_col(df, CAND_COL_PAIR)
    col_lod = pick_first_existing_col(df, CAND_COL_LOD_SIG)
    col_dzm = pick_first_existing_col(df, CAND_COL_DZ_MEAN_LOCAL)
    col_dzp = pick_first_existing_col(df, CAND_COL_DZ_POINT)
    col_dist = pick_first_existing_col(df, CAND_COL_DIST_PATCH)
    col_prob = pick_first_existing_col(df, CAND_COL_SCM_PROB)
    col_bin = pick_first_existing_col(df, CAND_COL_SCM_BIN)
    if col_site is None or col_pair is None:
        raise ValueError(f'Public-release status message.{GEOMORPH_CSV}')
    out = pd.DataFrame()
    out['site_id'] = df[col_site].astype(str)
    out['pair'] = df[col_pair].astype(str)
    if col_lod is None:
        out['LOD_sig'] = 1
    else:
        out['LOD_sig'] = pd.to_numeric(df[col_lod], errors='coerce').fillna(0).astype(int)
    out['dz_mean_local'] = pd.to_numeric(df[col_dzm], errors='coerce') if col_dzm else np.nan
    out['dz_point'] = pd.to_numeric(df[col_dzp], errors='coerce') if col_dzp else np.nan
    out['dist_to_patch_m'] = pd.to_numeric(df[col_dist], errors='coerce') if col_dist else np.nan
    if col_prob is None:
        out['scm_prob_local'] = np.nan
    else:
        out['scm_prob_local'] = pd.to_numeric(df[col_prob], errors='coerce')
    if col_bin is None:
        out['scm_bin_local'] = np.nan
    else:
        out['scm_bin_local'] = pd.to_numeric(df[col_bin], errors='coerce')
    out['pair_late_date'] = out['pair'].apply(parse_pair_late_date)
    out['pair_dt_days'] = out['pair'].apply(compute_pair_dt_days)
    return out

def build_piecewise_rates_for_site(df_geom: pd.DataFrame, site_id: str) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    g = df_geom[df_geom['site_id'].astype(str) == str(site_id)].copy()
    g = g.dropna(subset=['pair_late_date'])
    if len(g) == 0:
        return pd.DataFrame(columns=['pair', 'pair_late_date', 'pair_dt_days', 'E_retro_rate', 'L_debris_rate', 'S_shear_rate'])
    rows = []
    for (pair, sub) in g.groupby('pair'):
        late_date = sub['pair_late_date'].iloc[0]
        dt_days = float(sub['pair_dt_days'].iloc[0]) if np.isfinite(sub['pair_dt_days'].iloc[0]) else 1.0
        sub_e = sub.copy()
        if USE_LOD_GATE_FOR_EROSION and 'LOD_sig' in sub_e.columns:
            sub_e = sub_e[sub_e['LOD_sig'] == 1]
        if sub_e['dz_mean_local'].notna().any():
            e_amount = float(np.nansum(np.maximum(-sub_e['dz_mean_local'].values, 0.0)))
        else:
            e_amount = 0.0
        E_rate = e_amount / dt_days
        sub_l = sub.copy()
        if USE_LOD_GATE_FOR_DEBRIS and 'LOD_sig' in sub_l.columns:
            sub_l = sub_l[sub_l['LOD_sig'] == 1]
        debris_gate = np.ones(len(sub_l), dtype=float)
        if USE_SCM_GATE_FOR_DEBRIS:
            if sub_l['scm_bin_local'].notna().any():
                debris_gate = np.where(sub_l['scm_bin_local'].values >= 0.5, 1.0, 0.0)
            elif sub_l['scm_prob_local'].notna().any():
                debris_gate = np.where(sub_l['scm_prob_local'].values >= SCM_PROB_MIN_FOR_DEBRIS, 1.0, 0.0)
            elif not ALLOW_DEBRIS_WITHOUT_SCM:
                debris_gate = np.zeros(len(sub_l), dtype=float)
        if sub_l['dz_mean_local'].notna().any():
            l_amount_raw = np.maximum(sub_l['dz_mean_local'].values, 0.0)
            l_amount = float(np.nansum(l_amount_raw * debris_gate))
        else:
            l_amount = 0.0
        L_rate = l_amount / dt_days
        sub_s = sub.copy()
        dzm = sub_s['dz_mean_local'].values if sub_s['dz_mean_local'].notna().any() else np.zeros(len(sub_s))
        dzp = sub_s['dz_point'].values if sub_s['dz_point'].notna().any() else np.zeros(len(sub_s))
        dist = sub_s['dist_to_patch_m'].values if sub_s['dist_to_patch_m'].notna().any() else np.ones(len(sub_s))
        dist = np.where(np.isfinite(dist) & (dist > 0), dist, 1.0)
        base_shear_amount = float(np.nansum(np.abs(dzp - dzm) / (dist + 1.0)))
        S_rate = base_shear_amount / dt_days
        rows.append([pair, late_date, dt_days, E_rate, L_rate, S_rate])
    out = pd.DataFrame(rows, columns=['pair', 'pair_late_date', 'pair_dt_days', 'E_retro_rate', 'L_debris_rate', 'S_shear_rate'])
    out = out.sort_values('pair_late_date').reset_index(drop=True)
    return out

def rates_to_piecewise_series(rates_df: pd.DataFrame, time_index: pd.DatetimeIndex, name_prefix: str='') -> (pd.Series, pd.Series, pd.Series):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    E_rate = pd.Series(0.0, index=time_index, name=f'{name_prefix}E_rate')
    L_rate = pd.Series(0.0, index=time_index, name=f'{name_prefix}L_rate')
    S_rate = pd.Series(0.0, index=time_index, name=f'{name_prefix}S_rate')
    if rates_df is None or len(rates_df) == 0:
        return (E_rate, L_rate, S_rate)
    nodes = list(rates_df['pair_late_date'].values)
    for i in range(len(rates_df)):
        node = rates_df.loc[i, 'pair_late_date']
        e = float(rates_df.loc[i, 'E_retro_rate'])
        l = float(rates_df.loc[i, 'L_debris_rate'])
        s = float(rates_df.loc[i, 'S_shear_rate'])
        if i < len(rates_df) - 1:
            next_node = rates_df.loc[i + 1, 'pair_late_date']
            mask = (time_index >= node) & (time_index < next_node)
        else:
            mask = time_index >= node
        E_rate.loc[mask] = e
        L_rate.loc[mask] = l
        S_rate.loc[mask] = s
    first_node = rates_df.loc[0, 'pair_late_date']
    pre_mask = time_index < first_node
    E_rate.loc[pre_mask] = 0.0
    L_rate.loc[pre_mask] = 0.0
    S_rate.loc[pre_mask] = 0.0
    return (E_rate, L_rate, S_rate)

def integrate_rate_to_cumulative(rate_series: pd.Series) -> pd.Series:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    idx = rate_series.index
    dt_days = idx.to_series().diff().dt.total_seconds().fillna(0.0) / 86400.0
    inc = rate_series.values * dt_days.values
    cum = np.cumsum(inc)
    base_name = rate_series.name if isinstance(rate_series.name, str) and len(rate_series.name) > 0 else 'rate'
    out_name = base_name.replace('_rate', '_cum')
    out = pd.Series(cum, index=idx, name=out_name)
    out = out - out.iloc[0]
    return out

def build_features_for_gnss(df_geom: pd.DataFrame, gnss_id: str, bh_id: str) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    w = load_gnss_vertical_cum_m(gnss_id)
    time_index = w.index
    (IT_star, dITdt) = compute_IT_star_and_dITdt(bh_id, time_index)
    rates_df = build_piecewise_rates_for_site(df_geom, gnss_id)
    out_rates = os.path.join(OUT_10B_DIR, f'GNSS_{gnss_id}_pair_rates.csv')
    rates_df.to_csv(out_rates, index=False, encoding='utf-8-sig')
    print(f'[OUT] GNSS {gnss_id}Public-release status message.{out_rates}')
    (E_rate, L_rate, S_rate) = rates_to_piecewise_series(rates_df, time_index, name_prefix=f'{gnss_id}_')
    E_cum_tmp = integrate_rate_to_cumulative(E_rate)
    if E_cum_tmp.max() > 0:
        E_cum_norm = (E_cum_tmp - E_cum_tmp.min()) / (E_cum_tmp.max() - E_cum_tmp.min())
    else:
        E_cum_norm = E_cum_tmp * 0.0
    S_rate = S_rate * (1.0 + SHEAR_RETRO_COUPLING * E_cum_norm)
    E_cum = integrate_rate_to_cumulative(E_rate)
    L_cum = integrate_rate_to_cumulative(L_rate)
    S_cum = integrate_rate_to_cumulative(S_rate)
    df = pd.DataFrame({'gnss_id': str(gnss_id), 'time': time_index, 'w_obs_m': w.values, 'I_T_star': IT_star.values, 'dI_T_dt': dITdt.values, 'E_rate': E_rate.values, 'L_rate': L_rate.values, 'S_rate': S_rate.values, 'E_cum': E_cum.values, 'L_cum': L_cum.values, 'S_cum': S_cum.values})
    gates = SITE_GATES.get(str(gnss_id), {'I': 1.0, 'E': 1.0, 'L': 1.0, 'S': 1.0})
    cool_rate_arr = np.minimum(df['dI_T_dt'].values.astype(float), 0.0)
    cool_rate_ser = pd.Series(cool_rate_arr, index=time_index, name=f'{gnss_id}_cool_rate')
    H_cum_ser = integrate_rate_to_cumulative(cool_rate_ser)
    df['H_cum'] = H_cum_ser.values
    df['xI'] = df['I_T_star'] * float(gates['I'])
    df['xE'] = df['E_cum'] * float(gates['E'])
    df['xL'] = df['L_cum'] * float(gates['L'])
    df['xS'] = df['S_cum'] * float(gates['S'])
    df['xH'] = df['H_cum'] * float(gates['I'])
    df['xC'] = 0.0
    df['xR'] = 0.0
    df['xP'] = 0.0
    if str(gnss_id) == '9286':
        (g_on, g_off, g_win) = time_gate_window(df['time'], CP_9286_T0, CP_9286_T1, CP_K_ON, CP_K_OFF)
        E_rate_g = pd.Series(df['E_rate'].values * g_win, index=time_index, name=f'{gnss_id}_E_rate_g')
        L_rate_g = pd.Series(df['L_rate'].values * g_win, index=time_index, name=f'{gnss_id}_L_rate_g')
        S_rate_g = pd.Series(df['S_rate'].values * g_win, index=time_index, name=f'{gnss_id}_S_rate_g')
        E_cum_g = integrate_rate_to_cumulative(E_rate_g)
        L_cum_g = integrate_rate_to_cumulative(L_rate_g)
        S_cum_g = integrate_rate_to_cumulative(S_rate_g)
        dI_g = pd.Series(df['dI_T_dt'].values * g_win, index=time_index, name=f'{gnss_id}_dI_g')
        I_cum_g = integrate_rate_to_cumulative(dI_g)
        df['xI'] = I_cum_g.values * float(gates['I'])
        df['xE'] = E_cum_g.values * float(gates['E'])
        df['xL'] = L_cum_g.values * float(gates['L'])
        df['xS'] = S_cum_g.values * float(gates['S'])
        df['xC'] = g_on
        if CP_ENABLE_9286_RAMP:
            t = pd.to_datetime(df['time'])
            t0 = pd.Timestamp(CP_9286_T0)
            t1 = pd.Timestamp(CP_9286_T1)
            ramp_days = ((t - t0).dt.total_seconds() / 86400.0).clip(lower=0.0)
            ramp_days = np.minimum(ramp_days, (t1 - t0).total_seconds() / 86400.0)
            df['xR'] = ramp_days.values
            post_days = ((t - t1).dt.total_seconds() / 86400.0).clip(lower=0.0)
            df['xP'] = post_days.values * g_off
    df['xF'] = 0.0
    df['xG'] = 0.0
    if USE_LFJ_COUPLING:
        lfjs = [k for (k, v) in LFJ_TO_GNSS.items() if str(v) == str(gnss_id)]
        if len(lfjs) > 0:
            gates_m = LFJ_GATES.get(str(gnss_id), {'F': 1.0, 'G': 1.0})
            xF_sum = np.zeros(len(time_index), float)
            xG_sum = np.zeros(len(time_index), float)
            for lfj_id in lfjs:
                lfj = load_lfj_series(lfj_id, time_index)
                open_m = lfj['open_mm'].values / 1000.0
                open_m = open_m - open_m[0]
                tilt_c = lfj['tilt_cum'].values
                tilt_c = tilt_c - tilt_c[0]
                xF_sum += open_m
                xG_sum += tilt_c
            df['xF'] = xF_sum * float(gates_m['F'])
            df['xG'] = xG_sum * float(gates_m['G'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['w_obs_m', 'xI', 'xE', 'xL', 'xS', 'xH', 'xC'])
    return df

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float=1.0, sample_weight: np.ndarray=None, fit_intercept: bool=True) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    assert X.ndim == 2 and y.ndim == 1 and (X.shape[0] == y.shape[0])
    (N, D) = X.shape
    if sample_weight is None:
        w = np.ones(N, float)
    else:
        w = np.asarray(sample_weight, float).reshape(-1)
        w = np.clip(w, 1e-12, None)
    if fit_intercept:
        X_aug = np.concatenate([np.ones((N, 1), float), X], axis=1)
        reg = np.eye(D + 1, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
    else:
        X_aug = X
        reg = np.eye(D, dtype=float) * float(alpha)
    Xw = X_aug * w[:, None]
    A = X_aug.T @ Xw + reg
    b = X_aug.T @ (w * y)
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
    return coef

def zscore_fit_transform(df: pd.DataFrame, cols: list):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df2 = df.copy()
    stats = {}
    for c in cols:
        arr = df2[c].astype(float).values
        mu = np.nanmean(arr)
        sd = np.nanstd(arr)
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        df2[c] = (arr - mu) / sd
        stats[c] = {'mean': float(mu), 'std': float(sd)}
    return (df2, stats)

def zscore_apply(df: pd.DataFrame, stats: dict):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df2 = df.copy()
    for (c, st) in stats.items():
        if c not in df2.columns:
            continue
        mu = float(st['mean'])
        sd = float(st['std'])
        if sd < 1e-12:
            sd = 1.0
        df2[c] = (df2[c].astype(float).values - mu) / sd
    return df2

def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    if y_true.size < 2:
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-20:
        return float('nan')
    return 1.0 - ss_res / ss_tot

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan

def pin_optimize_gamma(X: np.ndarray, y: np.ndarray, gamma0: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not TORCH_OK or not USE_PINN:
        return gamma0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    gamma = torch.nn.Parameter(torch.tensor(gamma0.reshape(-1, 1), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([gamma], lr=PINN_LR)
    for ep in range(1, PINN_EPOCHS + 1):
        opt.zero_grad()
        yhat = X_t @ gamma
        mse = torch.mean((yhat - y_t) ** 2)
        l2 = torch.sum(gamma ** 2)
        loss = mse + PINN_L2 * l2
        loss.backward()
        opt.step()
        if ep % 200 == 0 or ep == 1 or ep == PINN_EPOCHS:
            print(f'[PINN] ep={ep:4d} loss={loss.item():.6e} mse={mse.item():.6e} l2={l2.item():.6e}')
    gamma_opt = gamma.detach().cpu().numpy().reshape(-1)
    return gamma_opt

def save_station_outputs(df_station: pd.DataFrame, gnss_id: str):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if SAVE_PER_STATION_CSV:
        out_csv = os.path.join(OUT_10B_DIR, f'GNSS_{gnss_id}_thermo_slump_final_timeseries.csv')
        df_station.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{gnss_id}Public-release status message.{out_csv}')
    if SAVE_PER_STATION_PNG:
        out_png = os.path.join(OUT_10B_DIR, f'GNSS_{gnss_id}_thermo_slump_final_fit.png')
        plt.figure(figsize=(11, 4.2), dpi=PLOT_DPI)
        plt.plot(pd.to_datetime(df_station['time']), df_station['w_obs_m'], label='obs')
        plt.plot(pd.to_datetime(df_station['time']), df_station['w_pred_m'], label='pred')
        plt.title(f'GNSS {gnss_id} thermo-slump fit (FINAL)')
        plt.xlabel('time')
        plt.ylabel('vertical displacement (m) (subsidence positive)' if SUBSIDENCE_POSITIVE else 'vertical displacement (m)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f'Public-release status message.{gnss_id}Public-release status message.{out_png}')

def main():
    print('Public-release status message.')
    gnss_to_bh = build_gnss_to_bh_mapping()
    df_geom = load_geomorph_table()
    all_rows = []
    station_dfs = {}
    for gid in USE_GNSS_IDS:
        bh = gnss_to_bh[str(gid)]
        df_feat = build_features_for_gnss(df_geom, str(gid), str(bh))
        station_dfs[str(gid)] = df_feat
        all_rows.append(df_feat)
        print(f'[FEATURE] GNSS {gid} / BH {bh}Public-release status message.{len(df_feat)}')
    df_all = pd.concat(all_rows, ignore_index=True)
    X_cols = ['xI', 'xE', 'xL', 'xS', 'xH', 'xC', 'xR', 'xP', 'xF', 'xG']
    X = df_all[X_cols].values.astype(float)
    y = df_all['w_obs_m'].values.astype(float)
    X_std = np.nanstd(X, axis=0).astype(float)
    X_std[X_std == 0] = 1.0
    Xs = X / X_std
    print('Public-release status message.')
    if WEIGHT_EACH_STATION_EQUAL:
        sample_w = np.zeros(len(df_all), dtype=float)
        for gid in USE_GNSS_IDS:
            mask = df_all['gnss_id'].astype(str).values == str(gid)
            n = int(mask.sum())
            if n > 0:
                sample_w[mask] = 1.0 / n
        gamma0_scaled = weighted_ridge_fit(Xs, y, RIDGE_LAMBDA, sample_w)
    else:
        gamma0_scaled = ridge_fit(Xs, y, RIDGE_LAMBDA, sample_weight=None, fit_intercept=False)
    gamma0 = gamma0_scaled / X_std
    yhat0 = X @ gamma0
    r2_all_0 = r2_score(y, yhat0)
    print(f'Public-release status message.{gamma0}')
    print(f'Public-release status message.{r2_all_0:.4f}')
    if USE_PINN:
        if not TORCH_OK:
            print('Public-release status message.')
            gamma = gamma0
        else:
            print('Public-release status message.')
            gamma = pin_optimize_gamma(X, y, gamma0)
            yhat = X @ gamma
            r2_all = r2_score(y, yhat)
            print(f'Public-release status message.{gamma}')
            print(f'Public-release status message.{r2_all:.4f}')
    else:
        gamma = gamma0
    out_gamma = os.path.join(OUT_10B_DIR, f'gamma_results_10B_{X.shape[1]}drivers_FINAL.csv')
    row = {'ridge_lambda': float(RIDGE_LAMBDA), 'use_pinn': bool(USE_PINN), 'pinn_epochs': int(PINN_EPOCHS) if USE_PINN else 0, 'pinn_lr': float(PINN_LR) if USE_PINN else 0.0, 'subsid_pos': bool(SUBSIDENCE_POSITIVE), 'resample': str(RESAMPLE_RULE)}
    name_map = {'xI': 'gamma_I_thermo', 'xE': 'gamma_E_retro', 'xL': 'gamma_L_debris', 'xS': 'gamma_S_shear', 'xH': 'gamma_H_cool', 'xC': 'gamma_C_collapse', 'xR': 'gamma_R_ramp', 'xP': 'gamma_P_post', 'xF': 'gamma_F_open', 'xG': 'gamma_G_tilt'}
    for (j, col) in enumerate(X_cols):
        key = name_map.get(col, f'gamma_{col}')
        row[key] = float(gamma[j]) if len(gamma) > j else np.nan
    row['cp_enable_9286'] = bool(CP_ENABLE_9286)
    row['cp_enable_9286_ramp'] = bool(CP_ENABLE_9286_RAMP)
    row['cp_9286_t0'] = str(CP_9286_T0)
    row['cp_9286_t1'] = str(CP_9286_T1)
    row['cp_k_on'] = float(CP_K_ON)
    row['cp_k_off'] = float(CP_K_OFF)
    df_gamma = pd.DataFrame([row])
    df_gamma.to_csv(out_gamma, index=False, encoding='utf-8-sig')
    print(f'Public-release status message.{out_gamma}')
    df_all['w_pred_m'] = X @ gamma
    if SAVE_GLOBAL_CSV:
        out_all = os.path.join(OUT_10B_DIR, 'ALL_GNSS_thermo_slump_FINAL_timeseries.csv')
        df_all.to_csv(out_all, index=False, encoding='utf-8-sig')
        print(f'Public-release status message.{out_all}')
    for gid in USE_GNSS_IDS:
        dfg = df_all[df_all['gnss_id'].astype(str) == str(gid)].copy()
        r2g = r2_score(dfg['w_obs_m'].values, dfg['w_pred_m'].values)
        print(f'[R2] GNSS {gid} R2={r2g:.4f}  (N={len(dfg)})')
        save_station_outputs(dfg, str(gid))
    print('Public-release status message.')
if __name__ == '__main__':
    main()
