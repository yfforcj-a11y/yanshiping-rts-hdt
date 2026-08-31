"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import warnings
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except Exception:
    rpt = None
    _HAS_RUPTURES = True
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'EVENT_OUT'))
os.makedirs(OUT_DIR, exist_ok=True)
RESAMPLE_RULE_GNSS = '1h'
RESAMPLE_RULE_LFJ = '2h'
MIN_VALID_RATIO = 0.6
GNSS_STABLE_IDS = {'7704', '7627'}
GNSS_ACTIVE_IDS = {'9286'}
LFJ_STABLE_IDS = {'3D3'}
LFJ_ACTIVE_IDS = {'3A9'}
HAMPEL = {'GNSS': dict(ENABLE=True, K=3.5, WINDOW=25), 'LFJ': dict(ENABLE=True, K=3.5, WINDOW=25)}
BASELINE_HOURS = {'GNSS': 72, 'LFJ': 72}
THR_NORMALIZE = {'GNSS': {'ENABLE': True, 'MIN_BASELINE_POINTS': 120, 'MAX_BASELINE_HOURS': 168, 'SIGMA_CLIP_FACTOR': (0.5, 2.0)}}
GNSS_SIGMA_GROUP = {}
SMOOTH = {'GNSS': dict(WINDOW=9, MIN_PERIODS=5), 'LFJ': dict(WINDOW=9, MIN_PERIODS=5)}
EVENT_PARAMS: Dict[str, Dict[str, Any]] = {'GNSS': {'FAILURE': {'K_MAD': {'stable': 3.0, 'active': 2.0}, 'MIN_CONSECUTIVE': {'stable': 25, 'active': 20}}, 'TRIGGER': {'ENABLE_FOR_STABLE': False, 'K_MAD': {'stable': 6, 'active': 4}, 'MIN_CONSECUTIVE': {'stable': 24, 'active': 20}, 'CHANGEPOINT': {'METHOD': 'auto', 'MODEL': 'rbf', 'PENALTY': None, 'MAX_EARLY_DAYS': 2, 'MIN_INDEX_AFTER_BASELINE': 24}}}, 'LFJ': {'FAILURE': {'K_MAD': {'stable': 25.0, 'active': 20.0}, 'MIN_CONSECUTIVE': {'stable': 20, 'active': 15}, 'REQUIRE_MAX_RUN': True}, 'TRIGGER': {'ENABLE_FOR_STABLE': False, 'K_MAD': {'stable': 15.0, 'active': 10.0}, 'MIN_CONSECUTIVE': {'stable': 15, 'active': 10}, 'CHANGEPOINT': {'METHOD': 'auto', 'MODEL': 'rbf', 'PENALTY': None, 'MAX_EARLY_DAYS': 2, 'MIN_INDEX_AFTER_BASELINE': 24}}}}
PLOT_DPI = 300
SHOW_PLOTS = False
SAVE_PLOTS = False
FIGSIZE = (12, 4)
ZOOM_DAYS_BEFORE = 5
ZOOM_DAYS_AFTER = 30
GNSS_TIME_COL = '\u91c7\u96c6\u65f6\u95f4'
GNSS_DH_COL_CANDIDATES = ['H\u65b9\u5411\u7d2f\u8ba1(mm)', 'H\u7d2f\u8ba1\u53d8\u5f62', 'H(mm)', 'H']
LFJ_TIME_COL_CANDIDATES = ['\u8bbe\u5907\u65f6\u95f4', '\u7cfb\u7edf\u65f6\u95f4', 'device_time', 'system_time', 'Time', 'time']
LFJ_OPEN_COL_CANDIDATES = ['\u62c9\u7ebf\u53d8\u5316', 'open_mm', 'delta_len', 'D']
LFJ_LEN_COL_CANDIDATES = ['\u62c9\u7ebf', 'length_mm', 'C']
LOCAL_TEST = False
LOCAL_TEST_GNSS_9286 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gnss', 'yanshiping_rts_gnss_gn1_20250722_20251126.csv'))
LOCAL_TEST_LFJ_3A9 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'crack_meter', 'yanshiping_rts_crack_meter_c1_20250720_20251126.csv'))

def safe_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'cp936']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def read_gnss_csv(filepath: str) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df_raw = safe_read_csv(filepath)
    tcol = guess_time_col(df_raw, GNSS_TIME_COL, ['\u91c7\u96c6\u65f6\u95f4', 'time', 'Time', 'datetime', '\u65e5\u671f'])
    df_raw[tcol] = to_datetime_safe(df_raw[tcol])
    df_raw = df_raw.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    dh_col = pick_first_existing(df_raw, GNSS_DH_COL_CANDIDATES)
    if dh_col is None:
        raise ValueError(f'Public-release status message.{GNSS_DH_COL_CANDIDATES}')
    df_raw[dh_col] = pd.to_numeric(df_raw[dh_col], errors='coerce')
    return df_raw[[dh_col]].rename(columns={dh_col: 'vert_mm'})

def guess_time_col(df: pd.DataFrame, preferred: str, candidates: list) -> str:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c).lower()
        if '\u65f6\u95f4' in str(c) or 'time' in s or 'date' in s:
            return c
    raise ValueError(f'Public-release status message.{list(df.columns)[:40]}')

def pick_first_existing(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')

def ensure_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def hampel_filter(series: pd.Series, window: int=25, k: float=3.5) -> pd.Series:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    x = series.copy()
    if x.dropna().shape[0] < max(10, window):
        return x
    rolling_median = x.rolling(window, center=True, min_periods=max(5, window // 3)).median()
    abs_dev = (x - rolling_median).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=max(5, window // 3)).median()
    sigma = 1.4826 * mad
    sigma = sigma.replace(0, np.nan)
    outlier = abs_dev > k * sigma
    x[outlier] = np.nan
    return x

def time_diff_hours(index: pd.DatetimeIndex) -> pd.Series:
    return index.to_series().diff().dt.total_seconds() / 3600.0

def calc_rate_mm_per_h(cum_mm: pd.Series, dt_hours: pd.Series) -> pd.Series:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    return cum_mm.diff() / dt_hours

def robust_threshold_from_segment(series_abs: pd.Series, k_mad: float, baseline_hours: int) -> float:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    s = series_abs.dropna()
    if s.empty:
        return np.nan
    min_pts = 30
    max_hours = 24 * 7
    try:
        g = THR_NORMALIZE.get('GNSS', {}) if isinstance(THR_NORMALIZE, dict) else {}
        min_pts = int(g.get('MIN_BASELINE_POINTS', min_pts))
        max_hours = int(g.get('MAX_BASELINE_HOURS', max_hours))
    except Exception:
        pass
    t0 = s.index.min()
    h = int(baseline_hours)
    t1 = t0 + pd.Timedelta(hours=h)
    sb = s.loc[(s.index >= t0) & (s.index <= t1)].dropna()
    while sb.shape[0] < min_pts and h < max_hours:
        h += 24
        t1 = t0 + pd.Timedelta(hours=h)
        sb = s.loc[(s.index >= t0) & (s.index <= t1)].dropna()
    if sb.empty:
        sb = s
    med = float(np.median(sb.values))
    mad = float(np.median(np.abs(sb.values - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(sb.values))
    if not np.isfinite(sigma) or sigma <= 0:
        std = float(np.std(sb.values))
        sigma = std if std > 0 else 1e-06
    return float(med + k_mad * sigma)

def baseline_med_sigma(series_abs: pd.Series, baseline_hours: int) -> Tuple[float, float]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    s = series_abs.dropna()
    if s.empty:
        return (np.nan, np.nan)
    min_pts = 30
    max_hours = 24 * 7
    try:
        g = THR_NORMALIZE.get('GNSS', {}) if isinstance(THR_NORMALIZE, dict) else {}
        min_pts = int(g.get('MIN_BASELINE_POINTS', min_pts))
        max_hours = int(g.get('MAX_BASELINE_HOURS', max_hours))
    except Exception:
        pass
    t0 = s.index.min()
    h = int(baseline_hours)
    t1 = t0 + pd.Timedelta(hours=h)
    sb = s.loc[(s.index >= t0) & (s.index <= t1)].dropna()
    while sb.shape[0] < min_pts and h < max_hours:
        h += 24
        t1 = t0 + pd.Timedelta(hours=h)
        sb = s.loc[(s.index >= t0) & (s.index <= t1)].dropna()
    if sb.empty:
        sb = s
    med = float(np.median(sb.values))
    mad = float(np.median(np.abs(sb.values - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(sb.values))
    if not np.isfinite(sigma) or sigma <= 0:
        std = float(np.std(sb.values))
        sigma = std if std > 0 else 1e-06
    return (med, sigma)

def detect_first_run_true(bool_series: pd.Series, min_consecutive: int) -> Optional[pd.Timestamp]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if bool_series is None or len(bool_series) == 0:
        return None
    b = bool_series.fillna(False).astype(bool).values
    count = 0
    for (i, flag) in enumerate(b):
        if flag:
            count += 1
            if count >= min_consecutive:
                return bool_series.index[i - min_consecutive + 1]
        else:
            count = 0
    return None

def max_consecutive_true(bool_series: pd.Series) -> int:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if bool_series is None or len(bool_series) == 0:
        return 0
    b = bool_series.fillna(False).astype(bool).values
    best = cur = 0
    for flag in b:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)

def baseline_end_time(index: pd.DatetimeIndex, baseline_hours: int) -> Optional[pd.Timestamp]:
    if index is None or len(index) == 0:
        return None
    t0 = index.min()
    return pd.Timestamp(t0) + pd.Timedelta(hours=baseline_hours)

def detect_trigger_ruptures(series: pd.Series, model: str='rbf', penalty=None, min_index_after_baseline: int=24) -> Optional[pd.Timestamp]:
    if not _HAS_RUPTURES:
        return None
    s = series.dropna()
    if s.shape[0] < 80:
        return None
    x = s.values.astype(float).reshape(-1, 1)
    if penalty is None:
        n = len(x)
        v = float(np.var(x))
        penalty = 3.0 * np.log(max(n, 2)) * max(v, 1e-06)
    try:
        algo = rpt.Pelt(model=model).fit(x)
        bkps = algo.predict(pen=penalty)
        for b in bkps:
            if b < len(s) and b > min_index_after_baseline:
                return s.index[b - 1]
    except Exception:
        return None
    return None

def choose_trigger(rate_smooth: pd.Series, thr_trigger: float, baseline_t1: Optional[pd.Timestamp], t_failure: Optional[pd.Timestamp], cp_cfg: Dict[str, Any], min_consecutive_trigger: int) -> Optional[pd.Timestamp]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    s = rate_smooth.copy()
    if s.isna().mean() < 0.9:
        s = s.interpolate('time', limit_direction='both')
    if baseline_t1 is not None:
        s = s.loc[s.index >= baseline_t1]
    if s.dropna().shape[0] < 30 or not np.isfinite(thr_trigger):
        return None
    cond_tr = s.abs() > thr_trigger
    t_thr = detect_first_run_true(cond_tr, min_consecutive=min_consecutive_trigger)
    t_cp = None
    method = str(cp_cfg.get('METHOD', 'off')).lower()
    if method in ['auto', 'ruptures']:
        t_cp = detect_trigger_ruptures(s, model=str(cp_cfg.get('MODEL', 'rbf')), penalty=cp_cfg.get('PENALTY', None), min_index_after_baseline=int(cp_cfg.get('MIN_INDEX_AFTER_BASELINE', 24)))
    t_trigger = t_thr
    max_early_days = float(cp_cfg.get('MAX_EARLY_DAYS', 2))
    if t_thr is not None and t_cp is not None:
        if t_cp <= t_thr and t_thr - t_cp <= pd.Timedelta(days=max_early_days):
            t_trigger = t_cp
    elif t_thr is None:
        t_trigger = t_cp
    if t_failure is not None and t_trigger is not None and (t_trigger >= t_failure):
        return None
    return t_trigger

def set_adaptive_time_ticks(ax, start: pd.Timestamp, end: pd.Timestamp):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    import matplotlib.dates as mdates
    span_days = max(1.0, (end - start).total_seconds() / 86400.0)
    if span_days <= 10:
        locator = mdates.DayLocator(interval=1)
        fmt = mdates.DateFormatter('%m-%d')
    elif span_days <= 30:
        locator = mdates.DayLocator(interval=3)
        fmt = mdates.DateFormatter('%m-%d')
    elif span_days <= 90:
        locator = mdates.DayLocator(interval=7)
        fmt = mdates.DateFormatter('%Y-%m-%d')
    elif span_days <= 180:
        locator = mdates.DayLocator(interval=14)
        fmt = mdates.DateFormatter('%Y-%m-%d')
    else:
        locator = mdates.MonthLocator(interval=1)
        fmt = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

def plot_series_full_and_zoom(df: pd.DataFrame, y_col: str, thr: Optional[float], t_trigger: Optional[pd.Timestamp], t_failure: Optional[pd.Timestamp], title_prefix: str, out_prefix: str, zoom_by_interval: Optional[Tuple[pd.Timestamp, pd.Timestamp]]=None):
    if not SAVE_PLOTS:
        return
    'Public-release status message.'
    if y_col not in df.columns or df[y_col].notna().sum() == 0:
        return
    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(df.index, df[y_col], label=y_col)
    if thr is not None and np.isfinite(thr):
        plt.axhline(thr, linestyle='--', label='thr(+)')
        plt.axhline(-thr, linestyle='--', label='thr(-)')
    if t_trigger is not None:
        plt.axvline(t_trigger, color='red', linestyle='--', label='trigger')
    if t_failure is not None:
        plt.axvline(t_failure, color='purple', linestyle='--', label='failure')
    plt.title(f'{title_prefix} (FULL)')
    plt.legend()
    ax = plt.gca()
    set_adaptive_time_ticks(ax, df.index.min(), df.index.max())
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{out_prefix}_FULL.png'), dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    if zoom_by_interval is not None and zoom_by_interval[0] is not None and (zoom_by_interval[1] is not None):
        (t0, t1) = zoom_by_interval
        t0b = pd.Timestamp(t0) - pd.Timedelta(days=2)
        t1b = pd.Timestamp(t1) + pd.Timedelta(days=2)
    else:
        anchors = [t for t in [t_trigger, t_failure] if t is not None]
        if not anchors:
            return
        (t0_ref, t1_ref) = (min(anchors), max(anchors))
        t0b = pd.Timestamp(t0_ref) - pd.Timedelta(days=max(ZOOM_DAYS_BEFORE, 3))
        t1b = pd.Timestamp(t1_ref) + pd.Timedelta(days=max(ZOOM_DAYS_AFTER, 7))
    dfx = df.loc[(df.index >= t0b) & (df.index <= t1b)].copy()
    if dfx.empty or dfx[y_col].notna().sum() == 0:
        return
    fig = plt.figure(figsize=FIGSIZE)
    plt.plot(dfx.index, dfx[y_col], label=y_col)
    if thr is not None and np.isfinite(thr):
        plt.axhline(thr, linestyle='--', label='thr(+)')
        plt.axhline(-thr, linestyle='--', label='thr(-)')
    if t_trigger is not None:
        plt.axvline(t_trigger, color='red', linestyle='--', label='trigger')
    if t_failure is not None:
        plt.axvline(t_failure, color='purple', linestyle='--', label='failure')
    plt.title(f'{title_prefix} (ZOOM)')
    plt.legend()
    ax = plt.gca()
    set_adaptive_time_ticks(ax, dfx.index.min(), dfx.index.max())
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{out_prefix}_ZOOM.png'), dpi=PLOT_DPI)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def process_gnss(gnss_id: str, filepath: str):
    p_all = EVENT_PARAMS['GNSS']
    p_fail = p_all['FAILURE']
    p_trig = p_all['TRIGGER']
    baseline_hours = int(BASELINE_HOURS['GNSS'])
    smooth_cfg = SMOOTH['GNSS']
    df_raw = safe_read_csv(filepath)
    tcol = guess_time_col(df_raw, GNSS_TIME_COL, ['\u91c7\u96c6\u65f6\u95f4', 'time', 'Time', 'datetime', '\u65e5\u671f'])
    df_raw[tcol] = to_datetime_safe(df_raw[tcol])
    df_raw = df_raw.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    dh_col = pick_first_existing(df_raw, GNSS_DH_COL_CANDIDATES)
    if dh_col is None:
        raise ValueError(f'GNSS{gnss_id}Public-release status message.{GNSS_DH_COL_CANDIDATES}')
    df_raw = ensure_numeric(df_raw, [dh_col])
    df = df_raw[[dh_col]].resample(RESAMPLE_RULE_GNSS).mean()
    valid_ratio = float(df[dh_col].notna().mean())
    if valid_ratio < MIN_VALID_RATIO:
        print(f'[WARN] GNSS{gnss_id}Public-release status message.{valid_ratio:.2f}')
    is_stable = gnss_id in GNSS_STABLE_IDS
    is_active = gnss_id in GNSS_ACTIVE_IDS
    hamp = HAMPEL['GNSS']
    if bool(hamp.get('ENABLE', False)):
        k = float(hamp.get('K', 3.5))
        w = int(hamp.get('WINDOW', 25))
        k_use = k if is_stable else k * 2.0
        df[dh_col] = hampel_filter(df[dh_col], window=w, k=k_use)
    dt_h = time_diff_hours(df.index)
    rate = calc_rate_mm_per_h(df[dh_col], dt_h)
    out = pd.DataFrame(index=df.index)
    out['vert_mm'] = df[dh_col]
    out['vert_rate'] = rate
    out['vert_rate_smooth'] = out['vert_rate'].rolling(int(smooth_cfg['WINDOW']), center=True, min_periods=int(smooth_cfg['MIN_PERIODS'])).median()
    role = 'stable' if is_stable else 'active'
    k_fail = float(p_fail['K_MAD'][role])
    (med0, sigma0) = baseline_med_sigma(out['vert_rate_smooth'].abs(), baseline_hours=baseline_hours)
    sigma_eff = sigma0
    if THR_NORMALIZE.get('GNSS', {}).get('ENABLE', False):
        (lo, hi) = THR_NORMALIZE['GNSS'].get('SIGMA_CLIP_FACTOR', (0.5, 2.0))
        sg = GNSS_SIGMA_GROUP.get(role, np.nan)
        if np.isfinite(sg) and sg > 0 and np.isfinite(sigma_eff) and (sigma_eff > 0):
            sigma_eff = float(np.clip(sigma_eff, lo * sg, hi * sg))
    thr_fail = float(med0 + k_fail * sigma_eff) if np.isfinite(med0) and np.isfinite(sigma_eff) else np.nan
    cond_fail = out['vert_rate_smooth'].abs() > thr_fail if np.isfinite(thr_fail) else pd.Series(False, index=out.index)
    min_consec_fail = int(p_fail['MIN_CONSECUTIVE'][role])
    t_failure = detect_first_run_true(cond_fail, min_consecutive=min_consec_fail)
    t_trigger = None
    thr_tr = None
    allow_trigger = is_active or (is_stable and bool(p_trig.get('ENABLE_FOR_STABLE', False)))
    if allow_trigger:
        k_tr = float(p_trig['K_MAD'][role])
        thr_tr = float(med0 + k_tr * sigma_eff) if np.isfinite(med0) and np.isfinite(sigma_eff) else np.nan
        b_end = baseline_end_time(out.index, baseline_hours)
        min_consec_tr = int(p_trig['MIN_CONSECUTIVE'][role])
        cp_cfg = dict(p_trig.get('CHANGEPOINT', {}))
        t_trigger = choose_trigger(rate_smooth=out['vert_rate_smooth'], thr_trigger=thr_tr, baseline_t1=b_end, t_failure=t_failure, cp_cfg=cp_cfg, min_consecutive_trigger=min_consec_tr)
    stability = {'sensor': gnss_id, 'type': 'GNSS', 'role': role, 'valid_ratio': valid_ratio, 'baseline_hours': baseline_hours, 'smooth_window': int(smooth_cfg['WINDOW']), 'thr_failure': float(thr_fail) if np.isfinite(thr_fail) else np.nan, 'thr_trigger': float(thr_tr) if thr_tr is not None and np.isfinite(thr_tr) else np.nan, 'fail_min_consecutive': min_consec_fail, 'fail_max_run': max_consecutive_true(cond_fail), 'fail_exceed_ratio': float(cond_fail.mean()) if len(cond_fail) else np.nan, 'trigger_min_consecutive': int(p_trig['MIN_CONSECUTIVE'][role]) if allow_trigger else np.nan, 'trigger_method': 'ruptures+thr' if _HAS_RUPTURES and str(p_trig.get('CHANGEPOINT', {}).get('METHOD', 'off')).lower() in ['auto', 'ruptures'] else 'thr_only'}
    return (out, t_trigger, t_failure, thr_fail, stability)

def process_lfj(lfj_id: str, filepath: str):
    p_all = EVENT_PARAMS['LFJ']
    p_fail = p_all['FAILURE']
    p_trig = p_all['TRIGGER']
    baseline_hours = int(BASELINE_HOURS['LFJ'])
    smooth_cfg = SMOOTH['LFJ']
    df_raw = safe_read_csv(filepath)
    tcol = guess_time_col(df_raw, preferred='\u8bbe\u5907\u65f6\u95f4', candidates=LFJ_TIME_COL_CANDIDATES)
    df_raw[tcol] = to_datetime_safe(df_raw[tcol])
    df_raw = df_raw.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    open_col = pick_first_existing(df_raw, LFJ_OPEN_COL_CANDIDATES)
    len_col = pick_first_existing(df_raw, LFJ_LEN_COL_CANDIDATES)
    if open_col is None:
        raise ValueError(f'LFJ{lfj_id}Public-release status message.{LFJ_OPEN_COL_CANDIDATES}')
    df_raw = ensure_numeric(df_raw, [open_col] + ([len_col] if len_col is not None else []))
    df = pd.DataFrame(index=df_raw.index)
    df['open_mm'] = df_raw[open_col]
    df['len_mm'] = df_raw[len_col] if len_col is not None else np.nan
    df = df.resample(RESAMPLE_RULE_LFJ).mean()
    valid_ratio = float(df['open_mm'].notna().mean())
    if valid_ratio < MIN_VALID_RATIO:
        print(f'[WARN] LFJ{lfj_id}Public-release status message.{valid_ratio:.2f}')
    is_stable = lfj_id in LFJ_STABLE_IDS
    is_active = lfj_id in LFJ_ACTIVE_IDS
    hamp = HAMPEL['LFJ']
    if bool(hamp.get('ENABLE', False)):
        k = float(hamp.get('K', 3.5))
        w = int(hamp.get('WINDOW', 25))
        k_use = k if is_stable else k * 2.0
        df['open_mm'] = hampel_filter(df['open_mm'], window=w, k=k_use)
    dt_h = time_diff_hours(df.index)
    rate_diff = calc_rate_mm_per_h(df['open_mm'], dt_h)
    rate_div = df['open_mm'] / dt_h

    def _mad_baseline(x: pd.Series) -> float:
        x = x.dropna()
        if x.shape[0] < 30:
            return np.inf
        med = float(np.median(x.values))
        return float(np.median(np.abs(x.values - med)))
    t0 = df.index.min()
    t1 = t0 + pd.Timedelta(hours=baseline_hours)
    dt_ok = (dt_h >= 1.0) & (dt_h <= 6.0)
    m1 = _mad_baseline(rate_diff.loc[(rate_diff.index >= t0) & (rate_diff.index <= t1)])
    m2 = _mad_baseline(rate_div.loc[dt_ok & (rate_div.index >= t0) & (rate_div.index <= t1)])
    df['open_rate'] = rate_diff if m1 <= m2 else rate_div
    df['open_rate_smooth'] = df['open_rate'].rolling(int(smooth_cfg['WINDOW']), center=True, min_periods=int(smooth_cfg['MIN_PERIODS'])).median()
    role = 'stable' if is_stable else 'active'
    k_fail = float(p_fail['K_MAD'][role])
    thr_fail = robust_threshold_from_segment(df['open_rate_smooth'].abs(), k_mad=k_fail, baseline_hours=baseline_hours)
    cond_fail = df['open_rate_smooth'].abs() > thr_fail if np.isfinite(thr_fail) else pd.Series(False, index=df.index)
    min_consec_fail = int(p_fail['MIN_CONSECUTIVE'][role])
    t_failure = detect_first_run_true(cond_fail, min_consecutive=min_consec_fail)
    if bool(p_fail.get('REQUIRE_MAX_RUN', True)):
        if max_consecutive_true(cond_fail) < min_consec_fail:
            t_failure = None
    t_trigger = None
    thr_tr = None
    allow_trigger = is_active or (is_stable and bool(p_trig.get('ENABLE_FOR_STABLE', False)))
    if allow_trigger:
        k_tr = float(p_trig['K_MAD'][role])
        thr_tr = robust_threshold_from_segment(df['open_rate_smooth'].abs(), k_mad=k_tr, baseline_hours=baseline_hours)
        b_end = baseline_end_time(df.index, baseline_hours)
        min_consec_tr = int(p_trig['MIN_CONSECUTIVE'][role])
        cp_cfg = dict(p_trig.get('CHANGEPOINT', {}))
        t_trigger = choose_trigger(rate_smooth=df['open_rate_smooth'], thr_trigger=thr_tr, baseline_t1=b_end, t_failure=t_failure, cp_cfg=cp_cfg, min_consecutive_trigger=min_consec_tr)
    stability = {'sensor': lfj_id, 'type': 'LFJ', 'role': role, 'valid_ratio': valid_ratio, 'baseline_hours': baseline_hours, 'smooth_window': int(smooth_cfg['WINDOW']), 'thr_failure': float(thr_fail) if np.isfinite(thr_fail) else np.nan, 'thr_trigger': float(thr_tr) if thr_tr is not None and np.isfinite(thr_tr) else np.nan, 'fail_min_consecutive': min_consec_fail, 'fail_max_run': max_consecutive_true(cond_fail), 'fail_exceed_ratio': float(cond_fail.mean()) if len(cond_fail) else np.nan, 'trigger_min_consecutive': int(p_trig['MIN_CONSECUTIVE'][role]) if allow_trigger else np.nan, 'trigger_method': 'ruptures+thr' if _HAS_RUPTURES and str(p_trig.get('CHANGEPOINT', {}).get('METHOD', 'off')).lower() in ['auto', 'ruptures'] else 'thr_only'}
    return (df, t_trigger, t_failure, thr_fail, stability)

def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    GNSS_FILES = {'7704': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gnss', 'yanshiping_rts_gnss_gn2_20250722_20251126.csv')), '7627': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gnss', 'yanshiping_rts_gnss_gn3_20250722_20251126.csv')), '9286': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gnss', 'yanshiping_rts_gnss_gn1_20250722_20251126.csv'))}
    global GNSS_SIGMA_GROUP
    GNSS_SIGMA_GROUP = {}
    if THR_NORMALIZE.get('GNSS', {}).get('ENABLE', False):
        sigma_by_role = {'stable': [], 'active': []}
        for (_sid, _fp) in GNSS_FILES.items():
            if not os.path.exists(_fp):
                continue
            try:
                _df = read_gnss_csv(_fp)
                _is_stable = _sid in GNSS_STABLE_IDS
                _role = 'stable' if _is_stable else 'active'
                _smooth_cfg = SMOOTH['GNSS']
                _rate = _df['vert_mm'].diff() / _df.index.to_series().diff().dt.total_seconds() * 3600.0
                _vr = _rate.rolling(int(_smooth_cfg['WINDOW']), center=True, min_periods=int(_smooth_cfg['MIN_PERIODS'])).median()
                (_med, _sigma) = baseline_med_sigma(_vr.abs(), baseline_hours=int(BASELINE_HOURS['GNSS']))
                if np.isfinite(_sigma) and _sigma > 0:
                    sigma_by_role[_role].append(float(_sigma))
            except Exception as e:
                print(f'[WARN] GNSS sigma pre-scan failed: {_sid} -> {e}')
                continue
        for (_r, _arr) in sigma_by_role.items():
            if len(_arr) > 0:
                GNSS_SIGMA_GROUP[_r] = float(np.median(_arr))
        print(f'[INFO] GNSS_SIGMA_GROUP (median sigma by role) = {GNSS_SIGMA_GROUP}')
    LFJ_FILES = {'3A9': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'crack_meter', 'yanshiping_rts_crack_meter_c1_20250720_20251126.csv')), '3D3': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'crack_meter', 'yanshiping_rts_crack_meter_c2_20250720_20251126.csv'))}
    if LOCAL_TEST:
        GNSS_FILES = {'9286': LOCAL_TEST_GNSS_9286}
        LFJ_FILES = {'3A9': LOCAL_TEST_LFJ_3A9}
    records = []
    stability_rows = []
    for (sid, fp) in GNSS_FILES.items():
        if not os.path.exists(fp):
            print(f'[WARN] missing GNSS file: {sid} -> {fp}')
            continue
        (df, t_trigger, t_failure, thr_fail, stability) = process_gnss(sid, fp)
        stability_rows.append(stability)
        plot_series_full_and_zoom(df=df, y_col='vert_rate_smooth', thr=thr_fail, t_trigger=t_trigger, t_failure=t_failure, title_prefix=f'GNSS {sid} vert_rate_smooth (mm/h)', out_prefix=f'GNSS_{sid}_rate')
        records.append({'sensor': sid, 'type': 'GNSS', 't_trigger': t_trigger, 't_failure': t_failure})
        df.to_csv(os.path.join(OUT_DIR, f'GNSS_{sid}_timeseries_clean_v5.csv'), encoding='utf-8-sig')
    for (sid, fp) in LFJ_FILES.items():
        if not os.path.exists(fp):
            print(f'[WARN] missing LFJ file: {sid} -> {fp}')
            continue
        (df, t_trigger, t_failure, thr_fail, stability) = process_lfj(sid, fp)
        stability_rows.append(stability)
        plot_series_full_and_zoom(df=df, y_col='open_rate_smooth', thr=thr_fail, t_trigger=t_trigger, t_failure=t_failure, title_prefix=f'LFJ {sid} open_rate_smooth (mm/h)', out_prefix=f'LFJ_{sid}_rate')
        plot_series_full_and_zoom(df=df, y_col='open_mm', thr=None, t_trigger=t_trigger, t_failure=t_failure, title_prefix=f'LFJ {sid} open_mm (cable delta, mm)', out_prefix=f'LFJ_{sid}_open')
        records.append({'sensor': sid, 'type': 'LFJ', 't_trigger': t_trigger, 't_failure': t_failure})
        df.to_csv(os.path.join(OUT_DIR, f'LFJ_{sid}_timeseries_clean_v5.csv'), encoding='utf-8-sig')
    df_events = pd.DataFrame(records)
    df_events.to_csv(os.path.join(OUT_DIR, 'event_times_v5.csv'), index=False, encoding='utf-8-sig')
    df_stab = pd.DataFrame(stability_rows)
    df_stab.to_csv(os.path.join(OUT_DIR, 'stability_metrics_v5.csv'), index=False, encoding='utf-8-sig')
    print('Public-release status message.')
    print(df_events)
    print('Public-release status message.')
    print(df_stab)
if __name__ == '__main__':
    main()
