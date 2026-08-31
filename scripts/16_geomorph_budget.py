"""Public-release documentation. Scientific logic and parameters are unchanged."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows as rio_windows
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
DIFF_DIR = Path(__file__).resolve().parents[2] / 'data' / 'dod_0p1m'
LOD_DIR = OUT_DIR / 'LOD95'
ZONE_DIR = OUT_DIR / 'ZONES'
SCM_PROB_DIR = OUT_DIR / 'SCM_PROB'
SCM_BIN_DIR = OUT_DIR / 'SCM_BIN'
MOD13_DIR = OUT_DIR / 'GEOMORPH_CHANGE_13'
RAS_DIR = MOD13_DIR / 'RASTER'
PNG_DIR = MOD13_DIR / 'PNG'
STAT_DIR = MOD13_DIR / 'STATS'
GIF_DIR = MOD13_DIR / 'GIF'
for d in [RAS_DIR, PNG_DIR, STAT_DIR, GIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TAGS = ['230924', '240630', '250816', '251017']
FIRST = TAGS[0]
DZ_STRONG = 0.5
P_TRANSPORT = 0.5
USE_SCM_BIN_WHEN_NO_PROB = True
ALLOW_TRANSPORT_WITHOUT_SIAMESE = False
ENABLE_TRANSPORT_HALO = True
HALO_R = 2
MAKE_PNG = False
MAKE_GIF = False
PNG_DPI = 300
GIF_FPS = 0.5
FIGSIZE = (8, 12)
CROP_TO_VALID = True
SHOW_GRID = False
DZ_BBOX_MIN_ABS = 0.05
USE_GLOBAL_DZ_SCALE = True
DZ_ABS_PCTL = 98
DZ_VMAX_MIN = 0.5

def _find_single(dirp: Path, patterns):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if dirp is None or not dirp.exists():
        return None
    for pat in patterns:
        hits = list(dirp.glob(pat))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            hits = sorted(hits, key=lambda x: len(x.name))
            return hits[0]
    return None

def _pixel_area_from_transform(transform) -> float:
    xres = float(abs(transform.a))
    yres = float(abs(transform.e))
    return xres * yres

def dilate_mask(mask: np.ndarray, r: int):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if r <= 0:
        return mask
    out = mask.copy()
    (h, w) = mask.shape
    (ys, xs) = np.where(mask)
    for (y, x) in zip(ys, xs):
        (y0, y1) = (max(0, y - r), min(h, y + r + 1))
        (x0, x1) = (max(0, x - r), min(w, x + r + 1))
        out[y0:y1, x0:x1] = True
    return out

def save_uint8_geotiff(out_path: Path, arr_u8: np.ndarray, base_profile: dict, nodata=0):
    prof = base_profile.copy()
    prof.update(dtype=rasterio.uint8, count=1, nodata=nodata, compress='lzw')
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(arr_u8.astype(np.uint8), 1)

def _valid_bbox(mask: np.ndarray):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    (ys, xs) = np.where(mask)
    if len(ys) == 0:
        (h, w) = mask.shape
        return (0, h, 0, w)
    (y0, y1) = (int(ys.min()), int(ys.max()) + 1)
    (x0, x1) = (int(xs.min()), int(xs.max()) + 1)
    return (y0, y1, x0, x1)

def _fmt_stats_box(row: dict) -> str:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    Ae = row.get('area_erosion_m2', np.nan)
    Ad = row.get('area_deposition_m2', np.nan)
    At = row.get('area_transport_m2', np.nan)
    Ve = row.get('vol_erosion_m3', np.nan)
    Vd = row.get('vol_deposition_m3', np.nan)
    Vn = row.get('vol_net_m3', np.nan)
    Vto = row.get('turnover_ED_m3', np.nan)
    Vtr = row.get('transport_abs_m3', np.nan)
    Vm = row.get('mobilized_proxy_m3', np.nan)

    def f(x, nd='NA', fmt='{:.0f}'):
        return nd if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else fmt.format(x)
    lines = [f"Ae={f(Ae, fmt='{:.0f}')}, Ad={f(Ad, fmt='{:.0f}')}, At={f(At, fmt='{:.0f}')}" + ' m2', f"Ve={f(Ve, fmt='{:.0f}')}, Vd={f(Vd, fmt='{:.0f}')}, Vnet={f(Vn, fmt='{:.0f}')}" + ' m3', f"Turnover={f(Vto, fmt='{:.0f}')}, TransAbs={f(Vtr, fmt='{:.0f}')}" + ' m3', f"MobilizedProxy={f(Vm, fmt='{:.0f}')}" + ' m3']
    return '\n'.join(lines)
CLASS_LABELS = {0: 'NoData', 1: 'Stable', 2: 'Erosion', 3: 'Deposition', 4: 'Transport'}
CLASS_COLORS = {0: '#2b83ba', 1: '#2ca02c', 2: '#d62728', 3: '#8c564b', 4: '#7f7f7f'}
CLASS_CMAP = mcolors.ListedColormap([CLASS_COLORS[i] for i in range(5)], name='geomorph_class')
CLASS_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CLASS_CMAP.N)

def plot_class_png(out_png: Path, cls: np.ndarray, title: str, stats_row: dict | None=None):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    view = cls

    def _crop_to_valid_bbox(arr2d, valid_mask, pad=10):
        """Public-release documentation. Scientific logic and parameters are unchanged."""
        (ys, xs) = np.where(valid_mask)
        if len(xs) == 0 or len(ys) == 0:
            return (arr2d, (slice(None), slice(None)))
        (y0, y1) = (ys.min(), ys.max())
        (x0, x1) = (xs.min(), xs.max())
        y0 = max(0, y0 - pad)
        y1 = min(arr2d.shape[0] - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(arr2d.shape[1] - 1, x1 + pad)
        sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        return (arr2d[sl], sl)
        if CROP_TO_VALID:
            (view, _) = _crop_to_valid_bbox(cls, valid_mask=cls != 0, pad=15)
    (fig, ax) = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(view, cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation='nearest')
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    if SHOW_GRID:
        ax.grid(True, linewidth=0.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([CLASS_LABELS[i] for i in [0, 1, 2, 3, 4]])
    cbar.ax.tick_params(labelsize=10)
    if stats_row is not None:
        txt = _fmt_stats_box(stats_row)
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.35'))
    fig.tight_layout()
    fig.savefig(out_png, dpi=PNG_DPI)
    plt.close(fig)

def plot_dz_png(out_png: Path, dz: np.ndarray, title: str, stats_row: dict | None=None, vmax_fixed: float | None=None):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    view = dz

    def _crop_to_valid_bbox(arr2d, valid_mask, pad=10):
        """Public-release documentation. Scientific logic and parameters are unchanged."""
        (ys, xs) = np.where(valid_mask)
        if len(xs) == 0 or len(ys) == 0:
            return (arr2d, (slice(None), slice(None)))
        (y0, y1) = (ys.min(), ys.max())
        (x0, x1) = (xs.min(), xs.max())
        y0 = max(0, y0 - pad)
        y1 = min(arr2d.shape[0] - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(arr2d.shape[1] - 1, x1 + pad)
        sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        return (arr2d[sl], sl)
        if CROP_TO_VALID:
            v_for_bbox = np.isfinite(dz) & (np.abs(dz) >= DZ_BBOX_MIN_ABS)
            (view, _) = _crop_to_valid_bbox(dz, valid_mask=v_for_bbox, pad=15)
    if vmax_fixed is None:
        vmax = np.nanpercentile(np.abs(view), DZ_ABS_PCTL)
        vmax = float(max(vmax, DZ_VMAX_MIN))
    else:
        vmax = float(max(vmax_fixed, DZ_VMAX_MIN))
    (fig, ax) = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(view, vmin=-vmax, vmax=vmax, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    if SHOW_GRID:
        ax.grid(True, linewidth=0.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Elevation change (m)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    if stats_row is not None:
        txt = _fmt_stats_box(stats_row)
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.35'))
    fig.tight_layout()
    fig.savefig(out_png, dpi=PNG_DPI)
    plt.close(fig)

def try_make_gif(png_list, out_gif: Path, fps=1):
    try:
        import imageio.v2 as imageio
    except Exception:
        print('Public-release status message.')
        return
    imgs = [imageio.imread(str(p)) for p in png_list]
    imageio.mimsave(str(out_gif), imgs, fps=fps)

def classify_from_zones_and_scm(dz: np.ndarray, zones: np.ndarray, scm_active: np.ndarray, dz_strong: float, enable_halo: bool, halo_r: int):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    valid = np.isfinite(dz)
    if zones is None:
        erosion = valid & (dz <= -dz_strong)
        depo = valid & (dz >= +dz_strong)
        stable = valid & ~erosion & ~depo
    else:
        erosion = valid & (zones == 1)
        stable = valid & (zones == 2)
        depo = valid & (zones == 3)
        uniq = set(np.unique(zones[valid]).tolist())
        if not (1 in uniq and 2 in uniq and (3 in uniq)):
            erosion = valid & (dz <= -dz_strong)
            depo = valid & (dz >= +dz_strong)
            stable = valid & ~erosion & ~depo
    weak = valid & (np.abs(dz) < dz_strong)
    if scm_active is None:
        transport = np.zeros_like(valid, dtype=bool)
    else:
        transport = scm_active & weak
    if enable_halo and transport.any():
        edge_seed = erosion | depo
        halo = dilate_mask(edge_seed, halo_r)
        transport = transport | (scm_active if scm_active is not None else False) & weak & halo
    stable = stable & ~erosion & ~depo & ~transport
    cls = np.zeros_like(dz, dtype=np.uint8)
    cls[stable] = 1
    cls[transport] = 4
    cls[erosion] = 2
    cls[depo] = 3
    return (cls, erosion, depo, transport, stable)

def _pairs_from_tags(tags):
    pairs = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            pairs.append(f'{tags[j]}-{tags[i]}')
    return pairs

def _estimate_global_vmax(pairs) -> float | None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not USE_GLOBAL_DZ_SCALE:
        return None
    vmax_list = []
    for pair in pairs:
        (later, earlier) = pair.split('-')
        diff_fp = _find_single(DIFF_DIR, [f'yanshiping_rts_dod_20{later}_minus_20{earlier}_0p1m.tif', f'*{pair}*.tif'])
        if diff_fp is None:
            continue
        with rasterio.open(diff_fp) as ds:
            arr = ds.read(1, masked=False).astype(np.float32)
            if ds.nodata is not None:
                arr = np.where(arr == ds.nodata, np.nan, arr)
        v = float(np.nanpercentile(np.abs(arr), DZ_ABS_PCTL))
        if np.isfinite(v):
            vmax_list.append(max(v, DZ_VMAX_MIN))
    if len(vmax_list) == 0:
        return None
    return float(max(vmax_list))

def main():
    pairs = _pairs_from_tags(TAGS)
    global_vmax = _estimate_global_vmax(pairs)
    if USE_GLOBAL_DZ_SCALE and global_vmax is not None:
        print(f'Public-release status message.{global_vmax:.3f}Public-release status message.')
    class_pngs = []
    dz_pngs = []
    all_rows = []
    for pair in pairs:
        (later, first) = pair.split('-')
        (later, earlier) = pair.split('-')
        diff_fp = _find_single(DIFF_DIR, [f'yanshiping_rts_dod_20{later}_minus_20{earlier}_0p1m.tif', f'*{pair}*.tif'])
        zone_fp = _find_single(ZONE_DIR, [f'ZONES_{pair}.tif', f'*{pair}*.tif'])
        lod_fp = _find_single(LOD_DIR, [f'LOD95_sig_{pair}.tif', f'*{pair}*.tif'])
        prob_fp = _find_single(SCM_PROB_DIR, [f'SCM_prob_{pair}.tif', f'SCM_prob_{pair}_*.tif'])
        bin_fp = _find_single(SCM_BIN_DIR, [f'SCM_bin_{pair}_th*.tif', f'SCM_bin_{pair}_*.tif'])
        if diff_fp is None:
            print(f'Public-release status message.{pair}Public-release status message.')
            continue
        scm_active_full = None
        scm_bounds = None
        if prob_fp is not None:
            with rasterio.open(prob_fp) as ds_p:
                p_full = ds_p.read(1, masked=False).astype(np.float32)
                if ds_p.nodata is not None:
                    p_full = np.where(p_full == ds_p.nodata, np.nan, p_full)
                scm_bounds = ds_p.bounds
            scm_active_full = np.isfinite(p_full) & (p_full >= P_TRANSPORT)
        elif bin_fp is not None and USE_SCM_BIN_WHEN_NO_PROB:
            with rasterio.open(bin_fp) as ds_b:
                b_full = ds_b.read(1, masked=False)
                if ds_b.nodata is not None:
                    b_full = np.where(b_full == ds_b.nodata, 0, b_full)
                b_full = (b_full > 0).astype(np.uint8)
                scm_bounds = ds_b.bounds
            scm_active_full = b_full == 1
        elif not ALLOW_TRANSPORT_WITHOUT_SIAMESE:
            scm_active_full = None
            scm_bounds = None
        else:
            scm_active_full = None
            scm_bounds = None
        with rasterio.open(diff_fp) as ds_diff:
            prof_full = ds_diff.profile
            if scm_bounds is not None:
                w = from_bounds(*scm_bounds, transform=ds_diff.transform)
                w = w.round_offsets().round_lengths()
                dz = ds_diff.read(1, window=w, masked=False).astype(np.float32)
                out_transform = rio_windows.transform(w, ds_diff.transform)
                out_prof = ds_diff.profile.copy()
                out_prof.update(height=dz.shape[0], width=dz.shape[1], transform=out_transform)
            else:
                dz = ds_diff.read(1, masked=False).astype(np.float32)
                out_prof = ds_diff.profile.copy()
            if ds_diff.nodata is not None:
                dz = np.where(dz == ds_diff.nodata, np.nan, dz)
        zones = None
        if zone_fp is not None:
            with rasterio.open(zone_fp) as ds_z:
                if scm_bounds is not None:
                    w = from_bounds(*scm_bounds, transform=ds_z.transform).round_offsets().round_lengths()
                    zones = ds_z.read(1, window=w, masked=False).astype(np.int32)
                else:
                    zones = ds_z.read(1, masked=False).astype(np.int32)
                if ds_z.nodata is not None:
                    zones = np.where(zones == ds_z.nodata, 0, zones)
        cell_area = _pixel_area_from_transform(out_prof['transform'])
        scm_active = None
        if scm_active_full is not None:
            if scm_bounds is not None:
                if prob_fp is not None:
                    with rasterio.open(prob_fp) as ds_p:
                        w = from_bounds(*scm_bounds, transform=ds_p.transform).round_offsets().round_lengths()
                        p = ds_p.read(1, window=w, masked=False).astype(np.float32)
                        if ds_p.nodata is not None:
                            p = np.where(p == ds_p.nodata, np.nan, p)
                    scm_active = np.isfinite(p) & (p >= P_TRANSPORT)
                elif bin_fp is not None:
                    with rasterio.open(bin_fp) as ds_b:
                        w = from_bounds(*scm_bounds, transform=ds_b.transform).round_offsets().round_lengths()
                        b = ds_b.read(1, window=w, masked=False)
                        if ds_b.nodata is not None:
                            b = np.where(b == ds_b.nodata, 0, b)
                    scm_active = b > 0
            else:
                scm_active = scm_active_full
        else:
            scm_active = None
        (cls, erosion, depo, transport, stable) = classify_from_zones_and_scm(dz=dz, zones=zones, scm_active=scm_active if scm_active is not None else None, dz_strong=DZ_STRONG, enable_halo=ENABLE_TRANSPORT_HALO, halo_r=HALO_R)
        stable = cls == 1
        erosion = cls == 2
        depo = cls == 3
        transport = cls == 4
        valid = np.isfinite(dz)
        A_all = float(valid.sum() * cell_area)

        def _area(m):
            return float(np.count_nonzero(m) * cell_area)

        def _vol(m):
            return float(np.nansum(np.where(m, dz, 0.0)) * cell_area)

        def _vol_abs(m):
            return float(np.nansum(np.where(m, np.abs(dz), 0.0)) * cell_area)
        A_s = _area(stable)
        A_e = _area(erosion)
        A_d = _area(depo)
        A_t = _area(transport)
        V_e = _vol(erosion)
        V_d = _vol(depo)
        V_t = _vol(transport)
        V_net = V_e + V_d
        V_turnover_ED = abs(V_e) + V_d
        V_transport_abs = _vol_abs(transport)
        V_mobilized = V_turnover_ED + V_transport_abs
        erosion_modulus_m = abs(V_e) / A_e if A_e > 0 else np.nan
        deposition_modulus_m = V_d / A_d if A_d > 0 else np.nan
        row = dict(pair=pair, later=later, first=first, dz_strong_m=DZ_STRONG, p_transport=P_TRANSPORT if prob_fp is not None else np.nan, cell_area_m2=cell_area, area_all_m2=A_all, area_stable_m2=A_s, area_erosion_m2=A_e, area_deposition_m2=A_d, area_transport_m2=A_t, vol_erosion_m3=V_e, vol_deposition_m3=V_d, vol_transport_m3=V_t, vol_net_m3=V_net, turnover_ED_m3=V_turnover_ED, transport_abs_m3=V_transport_abs, mobilized_proxy_m3=V_mobilized, erosion_modulus_m=erosion_modulus_m, deposition_modulus_m=deposition_modulus_m)
        all_rows.append(row)
        out_cls_tif = RAS_DIR / f'CLASS_{pair}.tif'
        save_uint8_geotiff(out_cls_tif, cls, out_prof, nodata=0)
        pd.DataFrame([row]).to_csv(STAT_DIR / f'geomorph_budget_{pair}.csv', index=False, encoding='utf-8-sig')
        if MAKE_PNG:
            t_class = f'Geomorph Classes {pair}'
            t_dz = f'DoD elevation change (m) {pair}'
            out_class_png = PNG_DIR / f'class_{pair}.png'
            out_dz_png = PNG_DIR / f'dz_{pair}.png'
            plot_class_png(out_class_png, cls, t_class, stats_row=row)
            plot_dz_png(out_dz_png, dz, t_dz, stats_row=row, vmax_fixed=global_vmax)
            class_pngs.append(out_class_png)
            dz_pngs.append(out_dz_png)
        print(f'[OK] {pair}: Ae={A_e:.0f} m2, Ad={A_d:.0f} m2, Vnet={V_net:.0f} m3, Mobilized={V_mobilized:.0f} m3')
    if len(all_rows) > 0:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(STAT_DIR / 'geomorph_budget_all.csv', index=False, encoding='utf-8-sig')
    if MAKE_GIF and len(class_pngs) >= 2:
        try_make_gif(class_pngs, GIF_DIR / 'class_timeseries.gif', fps=GIF_FPS)
    if MAKE_GIF and len(dz_pngs) >= 2:
        try_make_gif(dz_pngs, GIF_DIR / 'dz_timeseries.gif', fps=GIF_FPS)
if __name__ == '__main__':
    main()
