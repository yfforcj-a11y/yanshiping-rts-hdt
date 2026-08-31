from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS_ROOT))
from utils_project_paths import get_data_source, load_manifest, write_input_audit
TASK_NAME = 'Thread 6 Table 8 GNSS/LFJ event catalog'
TABLE_STEM = 'TABLE08_gnss_lfj_event_catalog'
EVENT_DIR = PROJECT_ROOT / 'out' / 'EVENT_OUT'
GNSS_ORDER = ['7704', '7627', '9286']
LFJ_ORDER = ['3A9', '3D3']
SITE_ORDER = GNSS_ORDER + LFJ_ORDER
ACTIVE_SITES = {'9286', '3A9'}
STABLE_SITES = {'7704', '7627', '3D3'}

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def output_layout(output_root: Path) -> dict[str, Path]:
    return {'tables_main': output_root / 'TABLES_MAIN', 'tables_supp': output_root / 'TABLES_SUPPLEMENTARY', 'manifests': output_root / 'MANIFESTS', 'qc': output_root / 'QC_REPORTS'}

def ensure_layout(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'cp936'):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def load_series(site: str, sensor_type: str) -> pd.DataFrame:
    prefix = 'GNSS' if sensor_type == 'GNSS' else 'LFJ'
    path = EVENT_DIR / f'{prefix}_{site}_timeseries_clean_v5.csv'
    df = read_csv_flexible(path)
    df['time'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    if sensor_type == 'GNSS':
        for col in ('vert_mm', 'vert_rate', 'vert_rate_smooth'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        for col in ('open_mm', 'open_rate', 'open_rate_smooth'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['time']).sort_values('time')

def summarize_site(site: str, sensor_type: str, df: pd.DataFrame, event_time: pd.Timestamp | None) -> dict[str, float | str | None]:
    if sensor_type == 'GNSS':
        (value_col, rate_col) = ('vert_mm', 'vert_rate_smooth')
        metric_unit = 'vertical displacement mm'
    else:
        (value_col, rate_col) = ('open_mm', 'open_rate_smooth')
        metric_unit = 'aperture change mm'
    value = pd.to_numeric(df[value_col], errors='coerce')
    rate = pd.to_numeric(df[rate_col], errors='coerce')
    event_value = np.nan
    event_rate = np.nan
    if event_time is not None and pd.notna(event_time) and (not df.empty):
        idx = (df['time'] - event_time).abs().idxmin()
        event_value = float(value.loc[idx]) if pd.notna(value.loc[idx]) else np.nan
        event_rate = float(rate.loc[idx]) if pd.notna(rate.loc[idx]) else np.nan
    return {'metric_unit': metric_unit, 'n_valid_value': int(value.notna().sum()), 'start_time': df['time'].min(), 'end_time': df['time'].max(), 'net_change': float(value.dropna().iloc[-1] - value.dropna().iloc[0]) if value.notna().sum() >= 2 else np.nan, 'min_value': float(value.min()) if value.notna().any() else np.nan, 'max_value': float(value.max()) if value.notna().any() else np.nan, 'max_abs_rate_mm_per_h': float(rate.abs().max()) if rate.notna().any() else np.nan, 'event_value': event_value, 'event_rate_mm_per_h': event_rate}

def interpretation(site: str) -> str:
    if site == '9286':
        return 'active evidence; GNSS failure event retained from event_times_v5'
    if site == '3A9':
        return 'active evidence; LFJ aperture failure event retained from event_times_v5'
    if site == '7627':
        return 'stable evidence; 7627/5C is local weakening/minor response, not confirmed failure propagation'
    if site in {'7704', '3D3'}:
        return 'stable evidence; no confirmed trigger/failure event'
    return 'not assigned'

def build_table() -> pd.DataFrame:
    events = read_csv_flexible(EVENT_DIR / 'event_times_v5.csv')
    stability = read_csv_flexible(EVENT_DIR / 'stability_metrics_v5.csv')
    events['sensor'] = events['sensor'].astype(str)
    stability['sensor'] = stability['sensor'].astype(str)
    rows: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        sensor_type = 'GNSS' if site in GNSS_ORDER else 'LFJ'
        event_row = events[events['sensor'] == site].iloc[0]
        stab_row = stability[stability['sensor'] == site].iloc[0]
        t_trigger = pd.to_datetime(event_row.get('t_trigger'), errors='coerce')
        t_failure = pd.to_datetime(event_row.get('t_failure'), errors='coerce')
        event_time = t_failure if pd.notna(t_failure) else t_trigger if pd.notna(t_trigger) else None
        df = load_series(site, sensor_type)
        summary = summarize_site(site, sensor_type, df, event_time)
        role = str(stab_row.get('role', 'active' if site in ACTIVE_SITES else 'stable'))
        rows.append({'sensor': site, 'sensor_type': sensor_type, 'status': role, 't_trigger': t_trigger if pd.notna(t_trigger) else '', 't_failure': t_failure if pd.notna(t_failure) else '', 'event_time_used': event_time if event_time is not None and pd.notna(event_time) else '', 'metric_unit': summary['metric_unit'], 'event_value': summary['event_value'], 'event_rate_mm_per_h': summary['event_rate_mm_per_h'], 'net_change': summary['net_change'], 'min_value': summary['min_value'], 'max_value': summary['max_value'], 'max_abs_rate_mm_per_h': summary['max_abs_rate_mm_per_h'], 'valid_ratio': pd.to_numeric(stab_row.get('valid_ratio'), errors='coerce'), 'baseline_hours': pd.to_numeric(stab_row.get('baseline_hours'), errors='coerce'), 'smooth_window': pd.to_numeric(stab_row.get('smooth_window'), errors='coerce'), 'thr_failure_mm_per_h': pd.to_numeric(stab_row.get('thr_failure'), errors='coerce'), 'thr_trigger_mm_per_h': pd.to_numeric(stab_row.get('thr_trigger'), errors='coerce'), 'fail_max_run': pd.to_numeric(stab_row.get('fail_max_run'), errors='coerce'), 'fail_exceed_ratio': pd.to_numeric(stab_row.get('fail_exceed_ratio'), errors='coerce'), 'trigger_method': stab_row.get('trigger_method', ''), 'series_start': summary['start_time'], 'series_end': summary['end_time'], 'n_valid_value': summary['n_valid_value'], 'interpretation_note': interpretation(site)})
    table = pd.DataFrame(rows)
    return table

def write_markdown(table: pd.DataFrame, path: Path) -> Path:
    cols = ['sensor', 'sensor_type', 'status', 't_trigger', 't_failure', 'event_value', 'event_rate_mm_per_h', 'interpretation_note']
    md_rows = table[cols].copy()
    for col in md_rows.columns:
        md_rows[col] = md_rows[col].map(lambda v: '' if pd.isna(v) else str(v))
    header = '| ' + ' | '.join(cols) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body = ['| ' + ' | '.join(row) + ' |' for row in md_rows.to_numpy(dtype=str)]
    lines = ['# Table 8 GNSS/LFJ Event Catalog', '', 'Event times, thresholds, and site roles are taken from existing `EVENT_OUT` v5 products. No event detection, threshold tuning, or GNSS/LFJ cleaning was rerun.', '', header, separator, *body, '', 'Interpretation guard: `9286` and `3A9` are active evidence; `7704`, `7627`, and `3D3` are stable evidence. `7627/5C` is retained as local thermal weakening/minor geomorphic response and is not interpreted as confirmed instability propagation.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

def file_record(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat() if path.exists() else None
    return {'role': role, 'path': str(path), 'exists': path.exists(), 'size_bytes': int(stat.st_size) if stat else None, 'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds') if stat else None}

def update_thread_report(layout: dict[str, Path], audit: Path, outputs: list[Path]) -> Path:
    path = layout['qc'] / 'THREAD6_run_report.md'
    prior = path.read_text(encoding='utf-8') if path.exists() else '# Thread 6 Run Report\n'
    lines = [prior.rstrip(), '', '## Table 8', '', f'- Timestamp: `{now_iso()}`', f'- Script: `{Path(__file__).resolve()}`', f'- Input audit: `{audit}`', '- Event detection was not rerun; thresholds are copied from `stability_metrics_v5.csv`.', '']
    for out in outputs:
        lines.append(f'- `{out}`')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def build_all(output_root: Path) -> list[Path]:
    manifest = load_manifest()
    layout = output_layout(output_root)
    ensure_layout(layout)
    audit = write_input_audit(TASK_NAME, {'gnss_lfj_root': ['gnss_monitoring', 'lfj_monitoring'], 'site_catalog_csv': 'site_catalog_csv'}, layout['manifests'], manifest=manifest)
    table = build_table()
    csv_path = layout['tables_main'] / f'{TABLE_STEM}.csv'
    md_path = layout['tables_main'] / f'{TABLE_STEM}.md'
    table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    write_markdown(table, md_path)
    manifest_path = layout['manifests'] / f'{TABLE_STEM}_manifest.json'
    payload = {'task': TASK_NAME, 'timestamp': now_iso(), 'script': str(Path(__file__).resolve()), 'inputs': [file_record(EVENT_DIR / 'event_times_v5.csv', 'event times'), file_record(EVENT_DIR / 'stability_metrics_v5.csv', 'stability metrics'), *[file_record(EVENT_DIR / f'GNSS_{site}_timeseries_clean_v5.csv', f'GNSS {site} clean_v5') for site in GNSS_ORDER], *[file_record(EVENT_DIR / f'LFJ_{site}_timeseries_clean_v5.csv', f'LFJ {site} clean_v5') for site in LFJ_ORDER], file_record(get_data_source('gnss_lfj_root', manifest=manifest), 'GNSS/LFJ source root'), file_record(get_data_source('site_catalog_csv', manifest=manifest), 'site catalog'), file_record(audit, 'input audit')], 'outputs': [str(csv_path.resolve()), str(md_path.resolve())], 'restrictions': ['No 13-event_detection_monitoring_final.py rerun.', 'No event-threshold redefinition.', 'No GNSS/LFJ recleaning.', 'No locked DoD/DEM/LAS processing.', '7627/5C not interpreted as confirmed failure propagation.']}
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs = [csv_path, md_path, manifest_path]
    outputs.append(update_thread_report(layout, audit, outputs))
    return outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'out' / 'PUBLICATION_FINAL')
    args = parser.parse_args()
    for path in build_all(args.output_root.resolve()):
        print(path)
if __name__ == '__main__':
    main()
