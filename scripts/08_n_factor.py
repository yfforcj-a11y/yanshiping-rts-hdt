"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if SCRIPTS_DIR.exists() and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from utils_project_paths import get_data_source, load_manifest, write_input_audit
except Exception:
    get_data_source = None
    load_manifest = None
    write_input_audit = None

def _manifest_path(key, fallback):
    if get_data_source is None:
        return str(fallback)
    try:
        return str(get_data_source(key))
    except Exception as exc:
        print(f'[WARN] DATA_MANIFEST.yaml source {key!r} was not used: {exc}')
        return str(fallback)
ERA5_RAW_ROOT = _manifest_path('era5_root', PROJECT_ROOT.parent / 'data' / 'era5')
ERA5_TEMPERATURE_CSV = _manifest_path('era5_temperature_csv', PROJECT_ROOT.parent / 'data' / 'era5' / 'era5_temperature.csv')
ERA5_PRECIPITATION_CSV = _manifest_path('era5_precipitation_csv', PROJECT_ROOT.parent / 'data' / 'era5' / 'era5_precipitation.csv')
ERA5_DAILY_CSV = ERA5_TEMPERATURE_CSV
BOREHOLE_CONFIG = {'1C': {'file_path': PROJECT_ROOT.parent / 'data' / 'borehole_temperature' / 'yanshiping_rts_borehole_bh1_ground_temperature_20250719_20251017.csv', 'time_col': '\u65f6\u95f4', 'temp_col': '0'}, '2C': {'file_path': PROJECT_ROOT.parent / 'data' / 'borehole_temperature' / 'yanshiping_rts_borehole_bh2_ground_temperature_20250719_20251017.csv', 'time_col': '\u65f6\u95f4', 'temp_col': '0'}, '3C': {'file_path': PROJECT_ROOT.parent / 'data' / 'borehole_temperature' / 'yanshiping_rts_borehole_bh3_ground_temperature_20250719_20251017.csv', 'time_col': '\u65f6\u95f4', 'temp_col': '0'}, '4C': {'file_path': PROJECT_ROOT.parent / 'data' / 'borehole_temperature' / 'yanshiping_rts_borehole_bh4_ground_temperature_20250719_20251017.csv', 'time_col': '\u65f6\u95f4', 'temp_col': '0'}, '5C': {'file_path': PROJECT_ROOT.parent / 'data' / 'borehole_temperature' / 'yanshiping_rts_borehole_bh5_ground_temperature_20250719_20251016.csv', 'time_col': '\u65f6\u95f4', 'temp_col': '0'}}
ERA5_TIME_OFFSET_HOURS = 0
FREEZE_THRESHOLD = 0.0
TEMP_QC_MIN = -30.0
TEMP_QC_MAX = 30.0
OUT_DIR = _manifest_path('meteorology_nfactor_root', PROJECT_ROOT.parent / 'outputs' / 'n_factor_outputs' / '0')
ENABLE_PLOTTING = False
PLOT_DPI = 150
PLOT_FIGSIZE = (8, 5)

def load_era5_daily(csv_path, time_offset_hours=0):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path)
    if 'date' in df.columns and 't2m_C_mean' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if time_offset_hours != 0:
            df['date'] = df['date'] + pd.to_timedelta(time_offset_hours, unit='h')
        df['date'] = df['date'].dt.floor('D')
        df_era = pd.DataFrame({'date': df['date'], 'Ta_C_daily': pd.to_numeric(df['t2m_C_mean'], errors='coerce')})
    elif 'valid_time' in df.columns and any((col == 't2m' or col.startswith('t2m.') for col in df.columns)):
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        if time_offset_hours != 0:
            df['valid_time'] = df['valid_time'] + pd.to_timedelta(time_offset_hours, unit='h')
        temp_cols = [col for col in df.columns if col == 't2m' or col.startswith('t2m.')]
        selected_col = None
        for col in temp_cols:
            vals = pd.to_numeric(df[col], errors='coerce')
            median_val = vals.dropna().median()
            if pd.notna(median_val) and -80.0 <= median_val <= 60.0:
                selected_col = col
                ta_c = vals
                break
        if selected_col is None:
            selected_col = temp_cols[0]
            ta_c = pd.to_numeric(df[selected_col], errors='coerce') - 273.15
        df_raw = pd.DataFrame({'date': df['valid_time'].dt.floor('D'), 'Ta_C': ta_c})
        df_era = df_raw.groupby('date', as_index=False)['Ta_C'].mean()
        df_era = df_era.rename(columns={'Ta_C': 'Ta_C_daily'})
        print(f'Public-release status message.{selected_col}')
    else:
        raise KeyError("ERA5 CSV must contain either daily columns ['date', 't2m_C_mean'] or raw ERA-5 columns ['valid_time', 't2m...'].")
    df_era = df_era.dropna(subset=['Ta_C_daily']).reset_index(drop=True)
    print(f"Public-release status message.{len(df_era)}Public-release status message.{df_era['date'].min()} ~ {df_era['date'].max()}")
    return df_era

def load_borehole_temp_daily(config, freeze_threshold=FREEZE_THRESHOLD):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    file_path = config['file_path']
    time_col = config['time_col']
    temp_col = config['temp_col']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Public-release status message.{file_path}')
    df = pd.read_csv(file_path)
    if time_col not in df.columns:
        raise KeyError(f'Public-release status message.{time_col}Public-release status message.{list(df.columns)}')
    if temp_col not in df.columns:
        raise KeyError(f'Public-release status message.{temp_col}Public-release status message.{list(df.columns)}')
    df[time_col] = pd.to_datetime(df[time_col])
    df = df[[time_col, temp_col]].copy()
    df = df.rename(columns={time_col: 'time', temp_col: 'Ts_C'})
    df['Ts_C'] = pd.to_numeric(df['Ts_C'], errors='coerce')
    bad_mask = (df['Ts_C'] < TEMP_QC_MIN) | (df['Ts_C'] > TEMP_QC_MAX)
    if bad_mask.any():
        print(f'Public-release status message.{os.path.basename(file_path)}Public-release status message.{temp_col}Public-release status message.{int(bad_mask.sum())}Public-release status message.{TEMP_QC_MIN}, {TEMP_QC_MAX}Public-release status message.')
        df.loc[bad_mask, 'Ts_C'] = np.nan
    df = df.dropna(subset=['Ts_C']).reset_index(drop=True)
    df['date'] = df['time'].dt.floor('D')
    group = df.groupby('date', as_index=False)['Ts_C'].mean()
    group = group.rename(columns={'Ts_C': 'Ts_C_daily'})
    print(f"Public-release status message.{os.path.basename(file_path)}Public-release status message.{len(group)}Public-release status message.{group['date'].min()} ~ {group['date'].max()}")
    return group

def merge_air_ground(df_era, df_bh):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df_m = pd.merge(df_era, df_bh, on='date', how='inner')
    df_m = df_m.sort_values('date').reset_index(drop=True)
    print(f"Public-release status message.{len(df_m)}Public-release status message.{df_m['date'].min()} ~ {df_m['date'].max()}")
    return df_m

def add_degree_days(df, air_col='Ta_C_daily', ground_col='Ts_C_daily', freeze_threshold=FREEZE_THRESHOLD):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df_out = df.copy()
    Ta = df_out[air_col].values
    FDD_air_day = np.where(Ta < freeze_threshold, freeze_threshold - Ta, 0.0)
    TDD_air_day = np.where(Ta > freeze_threshold, Ta - freeze_threshold, 0.0)
    Ts = df_out[ground_col].values
    FDD_gnd_day = np.where(Ts < freeze_threshold, freeze_threshold - Ts, 0.0)
    TDD_gnd_day = np.where(Ts > freeze_threshold, Ts - freeze_threshold, 0.0)
    df_out['FDD_air_day'] = FDD_air_day
    df_out['TDD_air_day'] = TDD_air_day
    df_out['FDD_gnd_day'] = FDD_gnd_day
    df_out['TDD_gnd_day'] = TDD_gnd_day
    df_out['FDD_air_cum'] = df_out['FDD_air_day'].cumsum()
    df_out['TDD_air_cum'] = df_out['TDD_air_day'].cumsum()
    df_out['FDD_gnd_cum'] = df_out['FDD_gnd_day'].cumsum()
    df_out['TDD_gnd_cum'] = df_out['TDD_gnd_day'].cumsum()
    return df_out

def summarize_n_factor(df, freeze_threshold=FREEZE_THRESHOLD):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    df2 = df.copy()
    df2['year'] = df2['date'].dt.year

    def compute_n_for_df(sub):
        """Public-release documentation. Scientific logic and parameters are unchanged."""
        freeze_mask = sub['FDD_air_day'] > 0
        FDD_air_sum = float(sub.loc[freeze_mask, 'FDD_air_day'].sum())
        FDD_gnd_sum = float(sub.loc[freeze_mask, 'FDD_gnd_day'].sum())
        thaw_mask = sub['TDD_air_day'] > 0
        TDD_air_sum = float(sub.loc[thaw_mask, 'TDD_air_day'].sum())
        TDD_gnd_sum = float(sub.loc[thaw_mask, 'TDD_gnd_day'].sum())
        nf_freeze = np.nan
        nf_thaw = np.nan
        if FDD_air_sum > 0:
            nf_freeze = FDD_gnd_sum / FDD_air_sum
        if TDD_air_sum > 0:
            nf_thaw = TDD_gnd_sum / TDD_air_sum
        return {'FDD_air_sum': FDD_air_sum, 'FDD_gnd_sum': FDD_gnd_sum, 'TDD_air_sum': TDD_air_sum, 'TDD_gnd_sum': TDD_gnd_sum, 'nf_freeze': nf_freeze, 'nf_thaw': nf_thaw}
    summary_all = compute_n_for_df(df2)
    summary_year = {}
    for (y, sub) in df2.groupby('year'):
        summary_year[y] = compute_n_for_df(sub)
    return (summary_all, summary_year)

def quick_plot_air_ground(df, borehole_name, out_dir, dpi=150, figsize=(8, 5)):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    os.makedirs(out_dir, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.plot(df['date'], df['Ta_C_daily'], label='Ta (ERA5, daily mean)', linewidth=0.8)
    ax.plot(df['date'], df['Ts_C_daily'], label='Ts (borehole, daily mean)', linewidth=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (deg C)')
    ax.set_title(f'Daily Ta and Ts - {borehole_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    out_png1 = os.path.join(out_dir, f'{borehole_name}_Ta_Ts_timeseries.png')
    fig.savefig(out_png1, dpi=dpi)
    plt.close(fig)
    (fig, ax) = plt.subplots(figsize=figsize)
    freeze_mask = df['FDD_air_day'] > 0
    thaw_mask = df['TDD_air_day'] > 0
    ax.scatter(df.loc[freeze_mask, 'Ta_C_daily'], df.loc[freeze_mask, 'Ts_C_daily'], s=10, alpha=0.6, label='Freeze days')
    ax.scatter(df.loc[thaw_mask, 'Ta_C_daily'], df.loc[thaw_mask, 'Ts_C_daily'], s=10, alpha=0.6, label='Thaw days')
    ax.axhline(0.0, color='grey', linewidth=0.8)
    ax.axvline(0.0, color='grey', linewidth=0.8)
    ax.set_xlabel('Ta (deg C)')
    ax.set_ylabel('Ts (deg C)')
    ax.set_title(f'Ta vs Ts - {borehole_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    out_png2 = os.path.join(out_dir, f'{borehole_name}_Ta_Ts_scatter.png')
    fig.savefig(out_png2, dpi=dpi)
    plt.close(fig)

def main():
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print('========================================')
    print('Public-release status message.')
    print('========================================')
    os.makedirs(OUT_DIR, exist_ok=True)
    if write_input_audit is not None:
        try:
            manifest = load_manifest() if load_manifest is not None else None
            write_input_audit(task_name='n_factor_era5_borehole', sources={'era5_root': ['era5_temperature', 'era5_precipitation'], 'temperature_monitoring_root': 'borehole_temperature', 'meteorology_nfactor_root': 'nfactor_degree_days'}, output_dir=OUT_DIR, manifest=manifest)
        except Exception as exc:
            raise RuntimeError(f'Input audit failed; stop before n-factor computation: {exc}') from exc
    df_era = load_era5_daily(ERA5_DAILY_CSV, time_offset_hours=ERA5_TIME_OFFSET_HOURS)
    for (bh_name, bh_cfg) in BOREHOLE_CONFIG.items():
        print('----------------------------------------')
        print(f'Public-release status message.{bh_name}')
        df_bh_day = load_borehole_temp_daily(bh_cfg, freeze_threshold=FREEZE_THRESHOLD)
        df_m = merge_air_ground(df_era, df_bh_day)
        if df_m.empty:
            print(f'Public-release status message.{bh_name}Public-release status message.')
            continue
        df_dd = add_degree_days(df_m, air_col='Ta_C_daily', ground_col='Ts_C_daily', freeze_threshold=FREEZE_THRESHOLD)
        (summary_all, summary_year) = summarize_n_factor(df_dd, freeze_threshold=FREEZE_THRESHOLD)
        print(f'Public-release status message.{bh_name}:')
        print(f"Public-release status message.{summary_all['FDD_air_sum']:.2f}Public-release status message.{summary_all['FDD_gnd_sum']:.2f}, nf_freeze = {summary_all['nf_freeze']:.3f}")
        print(f"Public-release status message.{summary_all['TDD_air_sum']:.2f}Public-release status message.{summary_all['TDD_gnd_sum']:.2f}, nf_thaw   = {summary_all['nf_thaw']:.3f}")
        print(f'Public-release status message.{bh_name}:')
        for y in sorted(summary_year.keys()):
            s = summary_year[y]
            print(f"Public-release status message.{y}: FDD_air = {s['FDD_air_sum']:.2f}, FDD_gnd = {s['FDD_gnd_sum']:.2f}, nf_freeze = {s['nf_freeze']:.3f}; TDD_air = {s['TDD_air_sum']:.2f}, TDD_gnd = {s['TDD_gnd_sum']:.2f}, nf_thaw = {s['nf_thaw']:.3f}")
        out_csv = os.path.join(OUT_DIR, f'Ta_Ts_degree_days_{bh_name}.csv')
        df_dd.to_csv(out_csv, index=False)
        print(f'Public-release status message.{out_csv}')
        if ENABLE_PLOTTING:
            quick_plot_air_ground(df_dd, borehole_name=bh_name, out_dir=OUT_DIR, dpi=PLOT_DPI, figsize=PLOT_FIGSIZE)
            print(f'Public-release status message.{bh_name}Public-release status message.')
    print('========================================')
    print('Public-release status message.')
    print('========================================')
if __name__ == '__main__':
    main()
