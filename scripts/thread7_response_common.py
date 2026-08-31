from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS_ROOT))
from utils_project_paths import get_data_source, load_manifest, write_input_audit
TASK_NAME = 'Thread 7 event-scale response space and multisource evidence matrix'
DP_ENV = sys.executable
PUBLICATION_ROOT = PROJECT_ROOT / 'out' / 'PUBLICATION_FINAL'
LEGACY_TABLE_DIR = PROJECT_ROOT / 'out' / 'PAPER_RESULTS_20260506' / 'TABLES'
EVENT_DIR = PROJECT_ROOT / 'out' / 'EVENT_OUT'
ICE_DIR = PROJECT_ROOT / 'out' / 'PINN_THERMAL_10A' / 'ICE_WEAKENING'
HDT_DIR = PROJECT_ROOT / 'out' / 'HDT_FUSION_14'
MAIN_SITES = ['9286', '3A9', '7627', '3D3', '7704']
ACTIVE_SITES = {'9286', '3A9'}
STABLE_SITES = {'7627', '3D3', '7704'}
BOREHOLE_SITES = ['2C', '5C']
PAIR_ORDER = ['240630-230924', '250816-230924', '250816-240630', '251017-230924', '251017-240630', '251017-250816']
MORANDI = {'ink': '#2F3437', 'muted_blue': '#5B7C99', 'slate_blue': '#7A91A7', 'teal': '#6E9A8D', 'sage': '#9AAA8B', 'ochre': '#C2A46D', 'terracotta': '#B98270', 'rose': '#B88A8A', 'violet': '#8E879F', 'warm_gray': '#B8B0A6', 'light_gray': '#E7E5E0', 'near_white': '#F7F6F2'}
SITE_COLORS = {'9286': MORANDI['terracotta'], '3A9': MORANDI['violet'], '7627': MORANDI['slate_blue'], '3D3': MORANDI['muted_blue'], '7704': MORANDI['warm_gray'], '2C': MORANDI['terracotta'], '5C': MORANDI['muted_blue']}

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def output_layout(output_root: Path=PUBLICATION_ROOT) -> dict[str, Path]:
    return {'fig_main': output_root / 'FIGURES_MAIN', 'fig_supp': output_root / 'FIGURES_SUPPLEMENTARY', 'tables_main': output_root / 'TABLES_MAIN', 'tables_supp': output_root / 'TABLES_SUPPLEMENTARY', 'captions': output_root / 'CAPTIONS', 'manifests': output_root / 'MANIFESTS', 'notes': output_root / 'NOTES', 'qc': output_root / 'QC_REPORTS'}

def ensure_layout(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

def setup_style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7.2, 'axes.linewidth': 0.55, 'axes.edgecolor': MORANDI['ink'], 'axes.labelcolor': MORANDI['ink'], 'xtick.color': MORANDI['ink'], 'ytick.color': MORANDI['ink'], 'xtick.major.width': 0.45, 'ytick.major.width': 0.45, 'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'cp936'):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int=600) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f'{stem}.png', out_dir / f'{stem}.svg', out_dir / f'{stem}.pdf']
    fig.savefig(paths[0], dpi=dpi, bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight')
    fig.savefig(paths[2], bbox_inches='tight')
    plt.close(fig)
    return paths

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha='left', va='top', fontsize=9.1, fontweight='bold', color=MORANDI['ink'], bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.2), zorder=20)

def file_record(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat() if path.exists() else None
    return {'role': role, 'path': str(path), 'exists': path.exists(), 'size_bytes': int(stat.st_size) if stat else None, 'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds') if stat else None}

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path

def load_thread7_inputs() -> dict[str, pd.DataFrame]:
    paths = {'site_pair': LEGACY_TABLE_DIR / 'site_pair_multisource_fusion.csv', 'site_summary': LEGACY_TABLE_DIR / 'site_multisource_summary.csv', 'pair_summary': LEGACY_TABLE_DIR / 'pair_multisource_summary.csv', 'events': EVENT_DIR / 'event_times_v5.csv', 'stability': EVENT_DIR / 'stability_metrics_v5.csv', 'ice_pair': ICE_DIR / 'ICE_WEAKENING_pair_aligned.csv', 'ice_join': ICE_DIR / 'ICE_WEAKENING_join_geomorph_by_pair.csv', 'hdt_pair': HDT_DIR / 'HDT_pair_summary.csv'}
    out: dict[str, pd.DataFrame] = {}
    for (key, path) in paths.items():
        if path.exists() and path.stat().st_size > 5:
            out[key] = read_csv_flexible(path)
        else:
            out[key] = pd.DataFrame()
    return out

def write_thread7_audit(layout: dict[str, Path]) -> Path:
    manifest = load_manifest()
    return write_input_audit(TASK_NAME, {'ice_weakening_root': 'ice_weakening_timeseries', 'hdt_fusion_root': 'hdt_tables'}, layout['manifests'], manifest=manifest)

def input_records(audit: Path | None=None) -> list[dict[str, Any]]:
    records = [file_record(LEGACY_TABLE_DIR / 'site_pair_multisource_fusion.csv', 'site-pair multisource fusion table'), file_record(LEGACY_TABLE_DIR / 'site_multisource_summary.csv', 'site multisource summary table'), file_record(LEGACY_TABLE_DIR / 'pair_multisource_summary.csv', 'pair multisource summary table'), file_record(EVENT_DIR / 'event_times_v5.csv', 'event times v5'), file_record(EVENT_DIR / 'stability_metrics_v5.csv', 'event stability metrics v5'), file_record(ICE_DIR / 'ICE_WEAKENING_pair_aligned.csv', 'pair-aligned ice-weakening indicators'), file_record(ICE_DIR / 'ICE_WEAKENING_join_geomorph_by_pair.csv', 'ice-weakening geomorph join table'), file_record(HDT_DIR / 'HDT_pair_summary.csv', 'HDT pair summary')]
    if audit is not None:
        records.append(file_record(audit, 'input audit'))
    return records

def pair_sort_key(pair: str) -> int:
    try:
        return PAIR_ORDER.index(str(pair))
    except ValueError:
        return len(PAIR_ORDER)

def add_pair_axis(ax: plt.Axes, pairs: list[str]) -> None:
    ax.set_xticks(np.arange(len(pairs)), pairs, rotation=35, ha='right')
    ax.tick_params(axis='x', labelsize=6.3)

def norm01(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors='coerce')
    if vals.notna().sum() == 0:
        return vals
    (mn, mx) = (vals.min(), vals.max())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return vals * 0 + 0.5
    return (vals - mn) / (mx - mn)

def evidence_score(df: pd.DataFrame) -> pd.Series:
    parts = []
    if 'LOD_sig' in df:
        parts.append(pd.to_numeric(df['LOD_sig'], errors='coerce').fillna(0).clip(0, 1))
    if 'SCM_prob' in df:
        parts.append(norm01(df['SCM_prob']).fillna(0))
    if 'abs_dz_point' in df:
        parts.append(norm01(df['abs_dz_point']).fillna(0))
    if 'meltcol_mean' in df:
        parts.append(norm01(df['meltcol_mean']).fillna(0))
    if not parts:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)

def role_for_site(site: str) -> str:
    if site in ACTIVE_SITES:
        return 'active'
    if site in STABLE_SITES:
        return 'stable'
    if site in BOREHOLE_SITES:
        return 'borehole'
    return 'supporting'
