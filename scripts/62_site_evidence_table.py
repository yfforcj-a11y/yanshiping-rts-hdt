from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from thread7_response_common import ACTIVE_SITES, DP_ENV, MAIN_SITES, MORANDI, PAIR_ORDER, PROJECT_ROOT, STABLE_SITES, evidence_score, ensure_layout, input_records, load_thread7_inputs, now_iso, output_layout, pair_sort_key, read_csv_flexible, role_for_site, write_json, write_thread7_audit
TASK_NAME = 'Thread 7 Table 9 multisource site-pair evidence'
TABLE_STEM = 'TABLE09_multisource_site_pair_evidence'

def _active_event_map(events: pd.DataFrame) -> dict[str, str]:
    if events.empty:
        return {}
    out: dict[str, str] = {}
    for (_, row) in events.iterrows():
        sensor = str(row.get('sensor', ''))
        failure = pd.to_datetime(row.get('t_failure'), errors='coerce')
        trigger = pd.to_datetime(row.get('t_trigger'), errors='coerce')
        if pd.notna(failure):
            out[sensor] = str(failure)
        elif pd.notna(trigger):
            out[sensor] = str(trigger)
    return out

def build_table() -> pd.DataFrame:
    data = load_thread7_inputs()
    sp = data['site_pair'].copy()
    if sp.empty:
        raise FileNotFoundError('site_pair_multisource_fusion.csv is required for Table 9.')
    sp['site_id'] = sp['site_id'].astype(str)
    sp['pair'] = sp['pair'].astype(str)
    sp = sp[sp['site_id'].isin(MAIN_SITES + ['2C', '5C'])].copy()
    sp['evidence_score_0_1'] = evidence_score(sp)
    event_map = _active_event_map(data['events'])
    rows: list[dict[str, Any]] = []
    for (_, row) in sp.sort_values(['site_id', 'pair'], key=lambda s: s.map(pair_sort_key) if s.name == 'pair' else s).iterrows():
        site = str(row['site_id'])
        role = str(row.get('field_role', '')) if pd.notna(row.get('field_role', '')) else role_for_site(site)
        nearest_bh = str(row.get('nearest_borehole', ''))
        confirmed_event = site in ACTIVE_SITES and site in event_map
        local_weakening = bool(pd.notna(row.get('meltcol_mean')) or nearest_bh in {'2C', '5C'})
        if site == '7627':
            interpretation = 'stable sector; 7627/5C indicates local weakening/minor geomorphic response, not confirmed failure propagation'
        elif confirmed_event:
            interpretation = 'confirmed active monitoring event with geomorphic/SCM context'
        elif site in STABLE_SITES:
            interpretation = 'stable monitoring evidence; no confirmed failure event'
        elif site in {'2C', '5C'}:
            interpretation = 'borehole thermal-geomorphic context only'
        else:
            interpretation = 'supporting multisource evidence'
        rows.append({'site_id': site, 'site_name': row.get('site_name', ''), 'site_type': row.get('site_type', ''), 'field_role': role, 'pair': row['pair'], 'nearest_borehole': nearest_bh, 'nearest_bh_distance_m': row.get('nearest_bh_distance_m', np.nan), 'dz_point_m': row.get('dz_point', np.nan), 'abs_dz_point_m': row.get('abs_dz_point', np.nan), 'lod95_significant': row.get('LOD_sig', np.nan), 'scm_probability': row.get('SCM_prob', np.nan), 'scm_binary': row.get('SCM_bin', np.nan), 'dist_to_patch_m': row.get('dist_to_patch_m', np.nan), 'posT_mean_Cm': row.get('posT_mean', np.nan), 'ALT_mean_m': row.get('ALT_mean', np.nan), 'meltcol_mean_m': row.get('meltcol_mean', np.nan), 'enthalpy_mean_proxy': row.get('enthalpy_mean', np.nan), 'evidence_score_0_1': row['evidence_score_0_1'], 'confirmed_event_time': event_map.get(site, ''), 'confirmed_event': confirmed_event, 'local_thermal_weakening_context': local_weakening, 'interpretation_note': interpretation})
    return pd.DataFrame(rows)

def write_markdown(table: pd.DataFrame, path: Path) -> Path:
    cols = ['site_id', 'field_role', 'pair', 'dz_point_m', 'lod95_significant', 'scm_probability', 'meltcol_mean_m', 'confirmed_event', 'interpretation_note']
    md = table[cols].copy()
    for col in ['dz_point_m', 'scm_probability', 'meltcol_mean_m']:
        md[col] = pd.to_numeric(md[col], errors='coerce').map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
    for col in md.columns:
        md[col] = md[col].map(lambda v: '' if pd.isna(v) else str(v))
    lines = ['# Table 9 Multisource Site-Pair Evidence', '', 'Evidence values are compiled from existing site-pair fusion, event, and ice-weakening CSV products. Locked DEM/DoD/LOD95/LAS/volume assets are cited only through existing table fields; no locked asset was redrawn or recalculated.', '', '| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    lines.extend(('| ' + ' | '.join(row) + ' |' for row in md.to_numpy(dtype=str)))
    lines.extend(['', 'Interpretation guard: `9286` and `3A9` are treated as confirmed active-event evidence. `7627/5C` is retained as local thermal weakening/minor geomorphic response and is not interpreted as whole-slope or sector-scale failure propagation.', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

def update_report(layout: dict[str, Path], audit: Path, outputs: list[Path]) -> Path:
    path = layout['qc'] / 'THREAD7_run_report.md'
    prior = path.read_text(encoding='utf-8') if path.exists() else '# Thread 7 Run Report\n'
    lines = [prior.rstrip(), '', '## Table 9', '', f'- Timestamp: `{now_iso()}`', f'- Script: `{Path(__file__).resolve()}`', f'- Environment intended: `DP` plotting/table stack (`{DP_ENV}`); executed with current Python.', f'- Input audit: `{audit}`', '- Source mode: existing CSV summaries only.', '- Locked assets: no DEM/DoD/LOD95/LAS/volume redraw, raster read, LAS read, or statistic recomputation.', '- Interpretation guard: 7627/5C is local weakening/minor response only, not confirmed failure propagation.', '', '### Outputs', '']
    lines.extend((f'- `{out}`' for out in outputs))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def build_all(output_root: Path) -> list[Path]:
    layout = output_layout(output_root)
    ensure_layout(layout)
    audit = write_thread7_audit(layout)
    table = build_table()
    csv_path = layout['tables_main'] / f'{TABLE_STEM}.csv'
    md_path = layout['tables_main'] / f'{TABLE_STEM}.md'
    table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    write_markdown(table, md_path)
    manifest_path = write_json(layout['manifests'] / f'{TABLE_STEM}_manifest.json', {'task': TASK_NAME, 'timestamp': now_iso(), 'script': str(Path(__file__).resolve()), 'inputs': input_records(audit), 'outputs': [str(csv_path.resolve()), str(md_path.resolve())], 'restrictions': ['No locked asset redraw or recalculation.', 'No DoD/LAS statistics recomputation.', '7627/5C not interpreted as confirmed failure propagation.']})
    outputs = [csv_path, md_path, manifest_path]
    outputs.append(update_report(layout, audit, outputs))
    return outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'out' / 'PUBLICATION_FINAL')
    args = parser.parse_args()
    for path in build_all(args.output_root.resolve()):
        print(path)
if __name__ == '__main__':
    main()
