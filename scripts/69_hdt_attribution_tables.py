from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = 'Thread 8 Table 10 HDT driver attribution'
DP_ENV = sys.executable
PAIR_ORDER = ['240630-230924', '250816-230924', '250816-240630', '251017-230924', '251017-240630', '251017-250816']
DRIVERS = {'geomorphic_change': ('driver_geomorphic_0_1', 0.3, 'terrain-change magnitude and HDT geomorphic score'), 'scm_active_change': ('driver_scm_0_1', 0.2, 'SCM active-change probability'), 'thermal_weakening': ('driver_thermal_0_1', 0.2, 'PINN/ice-weakening thermal proxy'), 'monitoring_event': ('driver_event_0_1', 0.2, 'GNSS/LFJ trigger and failure evidence'), 'spatial_proximity': ('driver_proximity_0_1', 0.1, 'monitoring-site proximity to active patches')}

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def output_layout(root: Path) -> dict[str, Path]:
    return {'fig_main': root / 'FIGURES_MAIN', 'fig_supp': root / 'FIGURES_SUPPLEMENTARY', 'tables_main': root / 'TABLES_MAIN', 'tables_supp': root / 'TABLES_SUPPLEMENTARY', 'captions': root / 'CAPTIONS', 'manifests': root / 'MANIFESTS', 'notes': root / 'NOTES', 'qc': root / 'QC_REPORTS'}

def ensure_layout(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 5:
        return pd.DataFrame()
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'gbk'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def file_record(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat() if path.exists() else None
    return {'role': role, 'path': str(path), 'exists': path.exists(), 'size_bytes': int(stat.st_size) if stat else None, 'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds') if stat else None}

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path

def pair_sort_key(pair: str) -> int:
    try:
        return PAIR_ORDER.index(str(pair))
    except ValueError:
        return len(PAIR_ORDER)

def ensure_driver_inputs(output_root: Path) -> Path:
    path = output_root / 'TABLES_SUPPLEMENTARY' / 'HDT_pair_driver_input_scores.csv'
    if path.exists() and path.stat().st_size > 5:
        return path
    script = PROJECT_ROOT / 'code' / '55-correlation_analysis_multisource.py'
    subprocess.run([sys.executable, str(script), '--output-root', str(output_root)], check=True)
    return path

def classify_sign(row: pd.Series) -> str:
    geom = float(row.get('driver_geomorphic_0_1', 0) or 0)
    scm = float(row.get('driver_scm_0_1', 0) or 0)
    thm = float(row.get('driver_thermal_0_1', 0) or 0)
    evt = float(row.get('driver_event_0_1', 0) or 0)
    if geom >= 0.55 and (scm >= 0.45 or evt >= 0.45):
        return 'positive active-change attribution'
    if thm >= 0.55 and geom < 0.45:
        return 'thermal preconditioning with weak geomorphic expression'
    if geom < 0.35 and scm < 0.35 and (evt < 0.35):
        return 'weak or background evidence'
    return 'mixed positive evidence'

def confidence_class(score: float) -> str:
    if score >= 0.72:
        return 'high'
    if score >= 0.48:
        return 'moderate'
    return 'low'

def numeric_value(value: Any, default: float=0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)

def build_table(driver_inputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (_, row) in driver_inputs.sort_values('pair', key=lambda s: s.map(pair_sort_key)).iterrows():
        raw = {}
        for (name, (col, weight, _)) in DRIVERS.items():
            raw[name] = max(numeric_value(row.get(col, 0), 0.0), 0)
        weighted = {name: val * DRIVERS[name][1] for (name, val) in raw.items()}
        total = sum(weighted.values())
        uncertainty_penalty = numeric_value(row.get('uncertainty_penalty_0_1', np.nan), 0.35)
        attribution_score = max(total * (1 - 0.28 * uncertainty_penalty), 0)
        contribution_sum = sum(weighted.values()) or 1.0
        dominant_driver = max(weighted, key=weighted.get)
        evidence_count = float(row.get('driver_evidence_count', sum((v > 0 for v in raw.values()))) or 0)
        confidence_score = np.clip(0.62 * attribution_score + 0.28 * (evidence_count / len(DRIVERS)) + 0.1 * (1 - uncertainty_penalty), 0, 1)
        rows.append({'pair': row['pair'], 'geomorphic_change_score_0_1': raw['geomorphic_change'], 'scm_active_change_score_0_1': raw['scm_active_change'], 'thermal_weakening_score_0_1': raw['thermal_weakening'], 'monitoring_event_score_0_1': raw['monitoring_event'], 'spatial_proximity_score_0_1': raw['spatial_proximity'], 'geomorphic_change_contribution_percent': weighted['geomorphic_change'] / contribution_sum * 100, 'scm_active_change_contribution_percent': weighted['scm_active_change'] / contribution_sum * 100, 'thermal_weakening_contribution_percent': weighted['thermal_weakening'] / contribution_sum * 100, 'monitoring_event_contribution_percent': weighted['monitoring_event'] / contribution_sum * 100, 'spatial_proximity_contribution_percent': weighted['spatial_proximity'] / contribution_sum * 100, 'dominant_driver': dominant_driver, 'driver_sign': classify_sign(row), 'uncertainty_penalty_0_1': uncertainty_penalty, 'attribution_score_0_1': attribution_score, 'confidence_score_0_1': confidence_score, 'confidence_class': confidence_class(float(confidence_score)), 'method_caveat': 'Derived from existing HDT, site-pair, event, thermal, and locked-asset companion CSVs; does not rerun fusion or redraw locked assets.'})
    return pd.DataFrame(rows)

def write_markdown(table: pd.DataFrame, path: Path) -> Path:
    cols = ['pair', 'dominant_driver', 'driver_sign', 'attribution_score_0_1', 'confidence_class', 'uncertainty_penalty_0_1']
    md = table[cols].copy()
    for col in ['attribution_score_0_1', 'uncertainty_penalty_0_1']:
        md[col] = pd.to_numeric(md[col], errors='coerce').map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
    lines = ['# Table 10 HDT Driver Attribution', '', 'Driver attribution is a post-fusion synthesis from existing HDT, multisource site-pair, event-response, thermal, and uncertainty companion tables. The HDT core fusion script was not rerun.', '', '| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    lines.extend(('| ' + ' | '.join(map(str, row)) + ' |' for row in md.to_numpy()))
    lines.extend(['', 'Default driver weights: geomorphic change 0.30, SCM active-change 0.20, thermal weakening 0.20, monitoring/event response 0.20, spatial proximity 0.10. Confidence is reduced by the uncertainty penalty and by sparse evidence coverage.', '', 'Locked DEM/DoD/LOD95/volume/LAS products are used only through existing companion tables and are not redrawn or recalculated.'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path

def build_all(output_root: Path) -> list[Path]:
    layout = output_layout(output_root)
    ensure_layout(layout)
    driver_path = ensure_driver_inputs(output_root)
    driver_inputs = read_csv(driver_path)
    table = build_table(driver_inputs)
    csv_path = layout['tables_main'] / 'TABLE10_hdt_driver_attribution.csv'
    md_path = layout['tables_main'] / 'TABLE10_hdt_driver_attribution.md'
    table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    write_markdown(table, md_path)
    manifest_path = write_json(layout['manifests'] / 'TABLE10_hdt_driver_attribution_manifest.json', {'task': TASK_NAME, 'timestamp': now_iso(), 'script': str(Path(__file__).resolve()), 'environment_intended': f'DP ({DP_ENV})', 'inputs': [file_record(driver_path, 'HDT pair driver input scores')], 'outputs': [str(csv_path.resolve()), str(md_path.resolve())], 'restrictions': ['No HDT core fusion rerun.', 'No locked asset redraw.', 'No LAS/TIF read.'], 'driver_weights': {name: weight for (name, (_, weight, _)) in DRIVERS.items()}})
    return [csv_path, md_path, manifest_path]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'out' / 'PUBLICATION_FINAL')
    args = parser.parse_args()
    for path in build_all(args.output_root.resolve()):
        print(path)
if __name__ == '__main__':
    main()
