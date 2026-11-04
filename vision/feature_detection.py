import os
from datetime import datetime
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

# ── Configuration ──────────────────────────────────────────────────
DEFAULT_THRESHOLD = 20
MIN_FEATURE_AREA = 1500
MAX_FEATURE_AREA = 60000
CLUSTER_DISTANCE = 300
MIN_CLUSTER_SIZE = 7
MAX_GROUPS = 4
GRID_ROWS = 2
GRID_COLS = 5

# Max extra blobs per row that the outlier-rejection stage will tolerate.
MAX_EXTRA_PER_ROW = 2

# ── Image Processing ──────────────────────────────────────────────

def extract_red_channel(rgb: np.ndarray) -> np.ndarray:
    """Pick the channel with the widest intensity range."""
    if rgb.ndim == 2:
        return rgb
    channels = cv2.split(rgb)
    best_idx = 0
    max_range = 0
    for i, ch in enumerate(channels[:3]):
        r = float(ch.max()) - float(ch.min())
        if r > max_range:
            max_range = r
            best_idx = i
    return channels[best_idx]

def threshold_image(channel: np.ndarray, thresh: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Dark pixels (< thresh) become 255; others become 0."""
    _, binary = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary

def clear_border_components(stats: np.ndarray, num_labels: int, width: int, height: int) -> List[int]:
    """Returns label IDs that do NOT touch the image edges."""
    valid_ids = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h) >= height - 1):
            continue
        valid_ids.append(i)
    return valid_ids

def find_components(binary: np.ndarray) -> List[Dict]:
    h, w = binary.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    valid_ids = clear_border_components(stats, num_labels, w, h)

    components = []
    for i in valid_ids:
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_FEATURE_AREA < area < MAX_FEATURE_AREA:
            components.append({
                'area': area,
                'centroid_x': float(centroids[i, 0]),
                'centroid_y': float(centroids[i, 1]),
                'x_min': int(stats[i, cv2.CC_STAT_LEFT]),
                'y_min': int(stats[i, cv2.CC_STAT_TOP]),
                'x_max': int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                'y_max': int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                'bbox_w': int(stats[i, cv2.CC_STAT_WIDTH]),
                'bbox_h': int(stats[i, cv2.CC_STAT_HEIGHT]),
            })
    return components

def erode_binary(binary: np.ndarray, iterations: int = 3) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(binary, kernel, iterations=iterations)

# ── Clustering ───────────────────────────────────────────────────

def cluster_components(components: List[Dict]) -> List[List[Dict]]:
    n = len(components)
    if n == 0: return []

    pts = np.array([[c['centroid_x'], c['centroid_y']] for c in components])
    assigned = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0: continue
        assigned[i] = cluster_id
        queue = [i]

        # BFS for neighbors within CLUSTER_DISTANCE
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            dists = np.linalg.norm(pts - pts[curr], axis=1)
            neighbors = np.where((assigned == -1) & (dists <= CLUSTER_DISTANCE))[0]
            for nb in neighbors:
                assigned[nb] = cluster_id
                queue.append(nb)
        cluster_id += 1

    clusters = [[] for _ in range(cluster_id)]
    for i, cid in enumerate(assigned):
        clusters[cid].append(components[i])

    valid = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]
    valid.sort(key=lambda g: (np.mean([p['centroid_y'] for p in g]), np.mean([p['centroid_x'] for p in g])))
    return valid[:MAX_GROUPS]

# ── Grid Fitting and Decoding ─────────────────────────────────────

def _split_rows_robust(cluster: List[Dict], debug: bool = False
                       ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split cluster blobs into 2 Y-bands via k=2 clustering. Blobs more
    than ~40% of the inter-band gap away from either band centre are
    rejected (returned as out-of-band).

    Returns (row0_blobs, row1_blobs, out_of_band), with row0 the lower-Y
    band. Row lists are sorted by X.
    """
    if len(cluster) < 2:
        return cluster, [], []

    sorted_y = sorted(cluster, key=lambda c: c['centroid_y'])
    y_vals = [c['centroid_y'] for c in sorted_y]
    diffs = np.diff(y_vals)
    if len(diffs) == 0:
        return sorted_y, [], []

    # Initial split at biggest Y gap, refine with k=2 means (3 iters).
    best_gap_idx = int(np.argmax(diffs)) + 1
    g0 = sorted_y[:best_gap_idx]
    g1 = sorted_y[best_gap_idx:]
    for _ in range(3):
        if not g0 or not g1:
            break
        c0 = float(np.mean([b['centroid_y'] for b in g0]))
        c1 = float(np.mean([b['centroid_y'] for b in g1]))
        n0, n1 = [], []
        for b in sorted_y:
            (n0 if abs(b['centroid_y'] - c0) <= abs(b['centroid_y'] - c1) else n1).append(b)
        if n0 == g0 and n1 == g1:
            break
        g0, g1 = n0, n1

    if not g0 or not g1:
        return sorted(sorted_y, key=lambda b: b['centroid_x']), [], []

    c0 = float(np.mean([b['centroid_y'] for b in g0]))
    c1 = float(np.mean([b['centroid_y'] for b in g1]))
    row_gap = abs(c1 - c0)
    y_tol = max(row_gap * 0.4, 20.0)

    out_of_band = []
    row0_clean, row1_clean = [], []
    for b in g0:
        if abs(b['centroid_y'] - c0) <= y_tol:
            row0_clean.append(b)
        else:
            out_of_band.append(b)
    for b in g1:
        if abs(b['centroid_y'] - c1) <= y_tol:
            row1_clean.append(b)
        else:
            out_of_band.append(b)

    # Ensure row0 = lower Y
    if (row0_clean and row1_clean and
            np.mean([b['centroid_y'] for b in row0_clean]) >
            np.mean([b['centroid_y'] for b in row1_clean])):
        row0_clean, row1_clean = row1_clean, row0_clean

    row0_clean.sort(key=lambda b: b['centroid_x'])
    row1_clean.sort(key=lambda b: b['centroid_x'])

    if debug and out_of_band:
        for b in out_of_band:
            print(f"  Y-band outlier rejected at "
                  f"px=({b['centroid_x']:.0f},{b['centroid_y']:.0f})")

    return row0_clean, row1_clean, out_of_band


def _select_5_by_template(row: List[Dict], template_xs: List[float]) -> List[Dict]:
    """Keep the 5 blobs whose X is closest to the other row's column
    template (greedy nearest-neighbour)."""
    if len(row) <= GRID_COLS:
        return row
    remaining = list(row)
    selected = []
    for tx in template_xs:
        if not remaining:
            break
        best = min(remaining, key=lambda b: abs(b['centroid_x'] - tx))
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=lambda b: b['centroid_x'])


def _select_5_by_uniformity(row: List[Dict]) -> List[Dict]:
    """Brute-force pick the 5-subset whose inter-blob X gaps are most
    uniform (lowest coefficient of variation)."""
    from itertools import combinations
    if len(row) <= GRID_COLS:
        return row
    best_subset = None
    best_score = float('inf')
    for combo in combinations(row, GRID_COLS):
        sc = sorted(combo, key=lambda b: b['centroid_x'])
        xs = np.array([b['centroid_x'] for b in sc])
        gaps = np.diff(xs)
        mean_gap = float(gaps.mean())
        if mean_gap < 1:
            continue
        cv = float(gaps.std()) / mean_gap
        if cv < best_score:
            best_score = cv
            best_subset = list(sc)
    if best_subset is None:
        return sorted(row, key=lambda b: b['centroid_x'])[:GRID_COLS]
    return best_subset


def fit_grid_to_cluster(cluster: List[Dict], binary: np.ndarray,
                        eroded: np.ndarray, debug: bool = False) -> Optional[Dict]:
    if len(cluster) < MIN_CLUSTER_SIZE: return None

    row0_blobs, row1_blobs, _outliers = _split_rows_robust(cluster, debug=debug)

    widths = sorted([b['bbox_w'] for b in cluster])
    ref_w = widths[len(widths)//4]
    bridge_thresh = ref_w * 1.4

    def process_row(blobs):
        """Bridge-split blobs that look like merged neighbours, then
        erode-classify each split piece."""
        res = []
        for b in blobs:
            if b['bbox_w'] > bridge_thresh:
                n = max(2, round(b['bbox_w'] / (ref_w * 1.2)))
                pw = (b['x_max'] - b['x_min']) / n
                for p in range(n):
                    x_start = int(b['x_min'] + p * pw)
                    x_end = int(b['x_min'] + (p+1) * pw)
                    crop = eroded[b['y_min']:b['y_max'], x_start:x_end]
                    ys, _ = np.where(crop == 255)
                    if len(ys) > 0:
                        res.append({
                            'centroid_x': (x_start + x_end) / 2.0,
                            'centroid_y': b['y_min'] + ys.mean(),
                            'bbox_w': x_end - x_start,
                            'bbox_h': ys.max() - ys.min(),
                            'area': len(ys),
                            'y_min': b['y_min'] + ys.min(), 'y_max': b['y_min'] + ys.max(),
                            'x_min': x_start, 'x_max': x_end
                        })
            else:
                crop = eroded[b['y_min']:b['y_max'], b['x_min']:b['x_max']]
                ys, _ = np.where(crop == 255)
                if len(ys) > 0:
                    b['area'] = len(ys)
                    b['bbox_h'] = ys.max() - ys.min()
                    res.append(b)
        return sorted(res, key=lambda x: x['centroid_x'])

    r0_proc = process_row(row0_blobs)
    r1_proc = process_row(row1_blobs)

    if (len(r0_proc) > GRID_COLS + MAX_EXTRA_PER_ROW or
            len(r1_proc) > GRID_COLS + MAX_EXTRA_PER_ROW):
        if debug:
            print(f"  Too many blobs after row-split (r0={len(r0_proc)}, "
                  f"r1={len(r1_proc)}) — refusing to guess")
        return None

    # Outlier rejection: prefer cross-row X-template; fall back to uniformity.
    if len(r0_proc) == GRID_COLS and len(r1_proc) > GRID_COLS:
        r0 = r0_proc
        r1 = _select_5_by_template(r1_proc, [b['centroid_x'] for b in r0])
        if debug:
            print(f"  Outlier reject: r1 {len(r1_proc)}→5 via r0 template")
    elif len(r1_proc) == GRID_COLS and len(r0_proc) > GRID_COLS:
        r1 = r1_proc
        r0 = _select_5_by_template(r0_proc, [b['centroid_x'] for b in r1])
        if debug:
            print(f"  Outlier reject: r0 {len(r0_proc)}→5 via r1 template")
    elif len(r0_proc) > GRID_COLS and len(r1_proc) > GRID_COLS:
        r0 = _select_5_by_uniformity(r0_proc)
        r1_uni = _select_5_by_uniformity(r1_proc)
        if r0 and r1_uni:
            r1 = _select_5_by_template(r1_proc, [b['centroid_x'] for b in r0])
        else:
            r1 = r1_uni
        if debug:
            print(f"  Outlier reject (both): r0 {len(r0_proc)}→5, "
                  f"r1 {len(r1_proc)}→5 by uniformity+template")
    else:
        r0 = r0_proc
        r1 = r1_proc

    if len(r0) < GRID_COLS or len(r1) < GRID_COLS: return None

    # Classify each blob by aspect + area → a 2×5 binary grid → (col_id, row_id).
    all_data = []
    for r_idx, row in enumerate([r0, r1]):
        for c_idx, b in enumerate(row):
            aspect = b['bbox_w'] / max(1, b['bbox_h'])
            all_data.append((r_idx, c_idx, aspect, b['area']))

    asp_min, asp_max = min(d[2] for d in all_data), max(d[2] for d in all_data)
    are_min, are_max = min(d[3] for d in all_data), max(d[3] for d in all_data)

    asp_range = asp_max - asp_min
    are_range = are_max - are_min
    avg_area = np.mean([d[3] for d in all_data])
    avg_aspect = np.mean([d[2] for d in all_data])

    # Uniform shapes → all blobs same type: (0,0) if small, (31,31) if big.
    is_uniform = (are_range < 4000) and (asp_range < 0.25)

    if is_uniform:
        if avg_area < 10000:
            col_id, row_id = 0, 0
        else:
            col_id, row_id = 31, 31
        confidence = 0.0
        if debug:
            print(f"  Uniform shapes: asp_range={asp_range:.3f} are_range={are_range:.0f} "
                  f"avg_area={avg_area:.0f} avg_asp={avg_aspect:.2f} -> ({col_id},{row_id})")
    else:
        scores = {}
        for r, c, asp, area in all_data:
            n_asp = (asp - asp_min) / max(0.001, (asp_max - asp_min))
            n_are = 1.0 - (area - are_min) / max(0.001, (are_max - are_min))
            scores[(r, c)] = n_asp + n_are

        sorted_sc = sorted(scores.values())
        split_idx = np.argmax(np.diff(sorted_sc))
        confidence = float(sorted_sc[split_idx+1] - sorted_sc[split_idx])

        thresh = (sorted_sc[split_idx] + sorted_sc[split_idx+1]) / 2.0
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        for (r, c), s in scores.items():
            grid[r, c] = 1 if s < thresh else 0

        col_id, row_id = 0, 0
        for c in range(GRID_COLS):
            col_id |= (grid[0, c] << c)
            row_id |= (grid[1, c] << c)

        if debug:
            print(f"  Mixed: asp_range={asp_range:.3f} are_range={are_range:.0f} "
                  f"confidence={confidence:.3f} -> ({col_id},{row_id})")

    return {
        'center': (np.mean([b['centroid_x'] for b in cluster]), np.mean([b['centroid_y'] for b in cluster])),
        'col_id': col_id, 'row_id': row_id,
        'confidence': confidence,
        'bbox': (min(b['x_min'] for b in cluster), min(b['y_min'] for b in cluster),
                 max(b['x_max'] for b in cluster), max(b['y_max'] for b in cluster))
    }

def detect_markers(rgb: np.ndarray, threshold: int = DEFAULT_THRESHOLD, debug: bool = False) -> List[Dict]:
    channel = extract_red_channel(rgb)
    binary = threshold_image(channel, threshold)
    components = find_components(binary)
    if not components: return []

    clusters = cluster_components(components)
    eroded = erode_binary(binary)

    markers = []
    for c in clusters:
        m = fit_grid_to_cluster(c, binary, eroded, debug)
        if m: markers.append(m)
    return markers

FAILED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_images")

def save_failed_frame(frame: np.ndarray, tag: str = "") -> Optional[str]:
    """Save a frame that failed marker detection for offline diagnosis."""
    if frame is None:
        return None
    try:
        os.makedirs(FAILED_IMAGES_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(FAILED_IMAGES_DIR, f"fail_{ts}{suffix}.jpg")
        from PIL import Image
        Image.fromarray(frame).save(path, quality=90)
        return path
    except Exception as e:
        print(f"[save_failed_frame] error: {e}")
        return None


if __name__ == '__main__':
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python feature_detection.py <image_path> [threshold]")
        sys.exit(1)

    img_path = sys.argv[1]
    thresh = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    rgb = np.array(Image.open(img_path).convert('RGB'))
    h, w = rgb.shape[:2]
    print(f"Image: {img_path} ({w}x{h})  threshold={thresh}")

    channel = extract_red_channel(rgb)
    binary = threshold_image(channel, thresh)
    components = find_components(binary)
    print(f"Components: {len(components)}")

    clusters = cluster_components(components)
    print(f"Clusters:   {len(clusters)}")

    eroded = erode_binary(binary)
    markers = []
    for i, cl in enumerate(clusters):
        m = fit_grid_to_cluster(cl, binary, eroded, debug=True)
        if m:
            markers.append(m)

    print(f"\n=== {len(markers)} markers detected ===")
    for m in markers:
        print(f"  ({m['col_id']},{m['row_id']}) at px "
              f"({m['center'][0]:.0f},{m['center'][1]:.0f}) "
              f"conf={m['confidence']:.3f}")
