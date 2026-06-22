"""
probe_dhc.py — Session 12

Scores each Hough candidate using the Basalamah (2012) Distance Histogram
method and compares acc-sort vs DHC-sort sphere ranking against ground truth.

DHC scoring per candidate:
  For candidate (cx, cy, r), compute distances from (cx,cy) to all edge pixels.
  Build histogram of distances in band [r-30%, r+30%].
  Score = peak_count / (2pi*r) — fraction of circumference with edge support.
  Real circles score high. Arc-fragment phantoms score low.

Cost: O(N_candidates x E) per frame — fast.

Usage:
    python3 probe_dhc.py <tiff_folder> <profile_json>
"""

import sys, os, json, glob, math
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import time

_CANNY_SIGMA           = 0.5
_CANNY_LOW             = 0.005
_CANNY_HIGH            = 0.02
_HOUGH_ACCUMULATOR_MIN = 0.25
_HOUGH_NUM_PEAKS       = 50
_HOUGH_MIN_DIST        = 20
_RADIUS_MIN_RATIO      = 0.02
_RADIUS_MAX_RATIO      = 0.15
_PF_RADIUS_MIN         = 0.018
_DETECTION_TARGET_DIM  = 1080
_DHC_BAND_FRACTION     = 0.30

def _load_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB","RGBA"): img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0,0,0))
        bg.paste(img, mask=img.split()[3]); img = bg
    return img

def _resize_for_detection(img):
    w, h = img.size
    scale = _DETECTION_TARGET_DIM / max(w, h)
    if scale >= 1.0: return img, 1.0
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS), scale

def _to_gray(arr):
    return 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]

def dhc_score(edge_pts, cx, cy, r, r_min, r_max):
    if len(edge_pts) == 0: return 0.0, 0
    diffs = edge_pts - np.array([cx, cy], dtype=np.float32)
    dists = np.sqrt((diffs**2).sum(axis=1))
    band = r * _DHC_BAND_FRACTION
    lo = max(r_min, r - band); hi = min(r_max, r + band)
    mask = (dists >= lo) & (dists <= hi)
    d_in = dists[mask]
    band_count = len(d_in)
    if band_count < 4: return 0.0, 0
    bins = np.arange(int(lo), int(hi) + 2)
    if len(bins) < 2: return 0.0, 0
    hist, _ = np.histogram(d_in, bins=bins)
    peak_val = int(hist.max())
    # Score = fraction of band edges that agree on the same radius.
    # This is radius-invariant: a real circle scores high regardless of size.
    # Large r/w phantoms with high raw counts score low because their band
    # contains thousands of edges spread across many radii.
    score = peak_val / band_count
    return score, peak_val

def load_profile(path):
    with open(os.path.expanduser(path)) as f: return json.load(f)

def get_gt(profile, label):
    cam = profile.get("cameras", {}).get(label, {})
    samples = [s for s in cam.get("samples",[]) if s.get("trust")=="verified_manual"]
    if not samples: samples = cam.get("samples",[])
    if not samples: return None
    g = samples[-1].get("geometry",{})
    return g.get("cx_norm"), g.get("cy_norm"), g.get("radius_ratio")

def camera_label(fname):
    parts = os.path.basename(fname).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts)>=2 else None

def find_rank(cands, gt):
    if gt is None: return None
    for i,c in enumerate(cands):
        if math.hypot(c['cx']-gt['cx'], c['cy']-gt['cy']) < 0.5*gt['r']: return i
    return None

def analyze(tiff_path, profile):
    label = camera_label(tiff_path)
    gt_raw = get_gt(profile, label) if label else None
    img = _load_image(tiff_path)
    img_det, scale = _resize_for_detection(img)
    arr = np.array(img_det, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    gray = _to_gray(arr)
    edges = canny(gray, sigma=_CANNY_SIGMA, low_threshold=_CANNY_LOW, high_threshold=_CANNY_HIGH)
    eys, exs = np.where(edges)
    edge_pts = np.stack([exs,eys],axis=1).astype(np.float32) if len(exs) else np.zeros((0,2),np.float32)
    gt_det = None
    if gt_raw and all(v is not None for v in gt_raw):
        gt_det = {'cx': gt_raw[0]*w, 'cy': gt_raw[1]*h, 'r': gt_raw[2]*w}
    r_min = max(6, int(w*_RADIUS_MIN_RATIO)); r_max = min(h//2, int(w*_RADIUS_MAX_RATIO))
    radii = np.arange(r_min, r_max+1)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii, min_xdistance=_HOUGH_MIN_DIST, min_ydistance=_HOUGH_MIN_DIST,
        num_peaks=_HOUGH_NUM_PEAKS, threshold=_HOUGH_ACCUMULATOR_MIN,
    )
    t0 = time.time()
    pf_r_min = int(w*_PF_RADIUS_MIN); pf_r_max = int(w*_RADIUS_MAX_RATIO)
    candidates = []
    for cx,cy,r in zip(cx_arr.tolist(),cy_arr.tolist(),r_arr.tolist()):
        r_idx = np.clip(np.searchsorted(radii,r),0,len(radii)-1)
        acc = float(hough_res[r_idx,int(cy),int(cx)])
        ds, pk = dhc_score(edge_pts, cx, cy, r, pf_r_min, pf_r_max)
        candidates.append({'cx':cx,'cy':cy,'r':r,'acc':acc,'dhc':ds,'peak':pk,'r_w':r/w})
    dhc_time = time.time()-t0
    by_acc = sorted(candidates, key=lambda c: -c['acc'])
    by_dhc = sorted(candidates, key=lambda c: -c['dhc'])  # peak/band_count — radius-invariant
    return {
        'label':label,'w':w,'h':h,'scale':scale,'gt_det':gt_det,
        'candidates':candidates,'by_acc':by_acc,'by_dhc':by_dhc,
        'acc_rank':find_rank(by_acc,gt_det),'dhc_rank':find_rank(by_dhc,gt_det),
        'dhc_time':dhc_time,'edge_count':int(edges.sum()),
    }

def report(res):
    cam = res['label'] or '?'
    n = len(res['candidates'])
    ar,dr = res['acc_rank'],res['dhc_rank']
    w,scale = res['w'],res['scale']
    gt = res['gt_det']
    ar_s = f"rank {ar+1}/{n}" if ar is not None else "MISSED"
    dr_s = f"rank {dr+1}/{n}" if dr is not None else "MISSED"
    if dr==0 and (ar is None or ar>0): v=" ✓ DHC FIXES"
    elif dr==0 and ar==0: v=" both rank 1"
    elif dr is not None and ar is not None and dr<ar: v=f" DHC improves ({ar+1}→{dr+1})"
    elif ar==0 and (dr is None or dr>0): v=" ✗ DHC BREAKS"
    else: v=" ✗ neither"
    print(f"\n{'='*70}")
    print(f"  {cam}  |  {n} candidates  edges={res['edge_count']}  DHC:{res['dhc_time']*1000:.0f}ms")
    print(f"  acc-rank: {ar_s}   peak-rank: {dr_s}{v}")
    if gt: print(f"  GT: cx={gt['cx']/scale:.0f} cy={gt['cy']/scale:.0f} r={gt['r']/scale:.0f} r/w={gt['r']/w:.3f}")
    print(f"{'='*70}")
    sphere = next((c for c in res['candidates'] if gt and math.hypot(c['cx']-gt['cx'],c['cy']-gt['cy'])<0.5*gt['r']),None)
    if sphere: print(f"  Sphere: acc={sphere['acc']:.3f}  dhc={sphere['dhc']:.3f}  peak={sphere['peak']}  r/w={sphere['r_w']:.3f}")
    if ar is not None and ar>0:
        print(f"  Above by acc (top 5/{ar}):")
        for i,c in enumerate(res['by_acc'][:min(5,ar)]):
            print(f"    [{i+1}] acc={c['acc']:.3f} dhc={c['dhc']:.3f} peak={c['peak']} r/w={c['r_w']:.3f}")
    if dr is not None and dr>0:
        print(f"  Above by DHC (top 5/{dr}):")
        for i,c in enumerate(res['by_dhc'][:min(5,dr)]):
            print(f"    [{i+1}] dhc={c['dhc']:.3f} acc={c['acc']:.3f} peak={c['peak']} r/w={c['r_w']:.3f}")

def summary(results):
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Camera':<14} {'Cands':>6} {'Acc-Rank':>10} {'Peak-Rank':>10} {'Result':>12}")
    print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*10} {'-'*12}")
    dhc_fixes=dhc_breaks=both_1=neither=0
    for r in results:
        cam=(r['label'] or '?')[:14]; n=len(r['candidates'])
        ar,dr=r['acc_rank'],r['dhc_rank']
        ar_s=f"{ar+1}/{n}" if ar is not None else "MISS"
        dr_s=f"{dr+1}/{n}" if dr is not None else "MISS"
        if ar==0 and dr==0: res_s="both rank 1"; both_1+=1
        elif dr==0 and ar!=0: res_s="DHC fixes"; dhc_fixes+=1
        elif ar==0 and dr!=0: res_s="DHC breaks"; dhc_breaks+=1
        else: res_s="neither"; neither+=1
        print(f"  {cam:<14} {n:>6} {ar_s:>10} {dr_s:>10} {res_s:>12}")
    print(f"\n  Both rank 1: {both_1}  DHC fixes: {dhc_fixes}  DHC breaks: {dhc_breaks}  Neither: {neither}")
    sphere_cands=[c for res in results for c in res['candidates']
                  if res['gt_det'] and math.hypot(c['cx']-res['gt_det']['cx'],c['cy']-res['gt_det']['cy'])<0.5*res['gt_det']['r']]
    non_sphere=[c for res in results for c in res['candidates']
                if not (res['gt_det'] and math.hypot(c['cx']-res['gt_det']['cx'],c['cy']-res['gt_det']['cy'])<0.5*res['gt_det']['r'])]
    if sphere_cands and non_sphere:
        print(f"  DHC — sphere: mean={np.mean([c['dhc'] for c in sphere_cands]):.3f} "
              f"min={np.min([c['dhc'] for c in sphere_cands]):.3f}")
        print(f"  DHC — others: mean={np.mean([c['dhc'] for c in non_sphere]):.3f} "
              f"max={np.max([c['dhc'] for c in non_sphere]):.3f}")
    print(f"  Avg DHC time: {sum(r['dhc_time'] for r in results)/len(results)*1000:.0f}ms/frame")

def main(tiff_folder, profile_path):
    profile = load_profile(profile_path)
    tiffs = sorted(glob.glob(os.path.join(tiff_folder,"*.tif")))
    if not tiffs: print(f"No TIFFs in {tiff_folder}"); sys.exit(1)
    print(f"Profile: {profile_path}\nTIFFs:   {len(tiffs)}")
    results=[]
    for tiff in tiffs:
        sys.stdout.write(f"  {os.path.basename(tiff)[:42]}... "); sys.stdout.flush()
        res=analyze(tiff,profile)
        ar=res['acc_rank']; dr=res['dhc_rank']
        print(f"acc={ar+1 if ar is not None else 'X'}  dhc={dr+1 if dr is not None else 'X'}  ({res['dhc_time']*1000:.0f}ms)")
        results.append(res)
    for res in results: report(res)
    summary(results)

if __name__=="__main__":
    if len(sys.argv)<3: print("Usage: python3 probe_dhc.py <tiff_folder> <profile_json>"); sys.exit(1)
    main(sys.argv[1],sys.argv[2])
