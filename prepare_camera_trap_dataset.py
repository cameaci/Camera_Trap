"""
prepare_camera_trap_dataset.py

Builds a balanced, leakage-free badger/otter/other dataset from WSP camera
trap footage, ready for SpeciesNet fine-tuning.

Finds the source folders itself. Just run it:

    python prepare_camera_trap_dataset.py

Then fill the 'label' column in review_and_label.csv and run it again.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
CLASSES = ("badger", "otter", "other")

LABEL_FILE = "review_and_label.csv"
DATASET_DIR = "dataset_uk_species"
WORK_DIR = "prep_work"

# ---------------------------------------------------------------------------
# Known sources, from the data offers received May-Sep 2026.
# 'markers' are folder-name fragments used to locate the data automatically.
# ---------------------------------------------------------------------------
KNOWN_SOURCES = [
    {
        "provider": "Pugh_Bethan",
        "project": "SEWUE_badger",
        "class_hint": "badger",
        "markers": ["sewue", "badger monitoring", "badger_monitoring"],
        "known_path": r"https://wsponlinegbr.sharepoint.com/sites/GB-SEWUE/Shared Documents",
        "note": "3 main setts + 2 subsidiaries. Tracker xlsx lists badger clips.",
    },
    {
        "provider": "White_Lydia",
        "project": "Otter_holts",
        "class_hint": "mixed",
        "markers": ["camera trapping", "otter rest", "holt"],
        "known_path": None,
        "note": "Otter, mink, polecat, mice, birds. Two monitoring periods.",
    },
    {
        "provider": "Blyth_Laura",
        "project": "BridgeOfAllan_70097308",
        "class_hint": "mixed",
        "markers": ["otter and beaver survey footage", "70097308",
                    "bridge of allan"],
        "known_path": (r"\\uk.wspgroup.com\central data\Projects\70097xxx"
                       r"\70097308 - Bridge of Allan FPS Engagement\03 WIP"
                       r"\Ecology\Data\Otter and Beaver survey footage 2026"),
        "note": "Network path, confirmed accessible.",
    },
    {
        "provider": "Laycock_Charlotte",
        "project": "NationalGrid",
        "class_hint": "other",
        "markers": ["national grid", "nationalgrid"],
        "known_path": None,
        "note": "~300 files, no badger, voles + vegetation triggers. "
                "PM approval pending.",
    },
    {
        "provider": "Brown_Richard",
        "project": "SWS_Anton_CSR",
        "class_hint": "mixed",
        "markers": ["anton", "sws anton"],
        "known_path": None,
        "note": "Otter videos shared previously.",
    },
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def candidate_roots() -> List[Path]:
    """Places worth searching on a WSP Windows machine."""
    roots: List[Path] = []
    home = Path.home()

    # OneDrive / SharePoint sync roots
    for var in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        v = os.environ.get(var)
        if v:
            roots.append(Path(v))
    for p in home.glob("OneDrive*"):
        roots.append(p)
    for p in home.glob("WSP*"):
        roots.append(p)

    # Common local working areas
    for name in ("Documents", "Downloads", "Desktop"):
        p = home / name
        if p.exists():
            roots.append(p)

    # Network share
    unc = Path(r"\\uk.wspgroup.com\central data\Projects")
    try:
        if unc.exists():
            roots.append(unc)
    except OSError:
        pass

    seen, out = set(), []
    for r in roots:
        try:
            rp = r.resolve()
        except OSError:
            continue
        if rp.exists() and str(rp).lower() not in seen:
            seen.add(str(rp).lower())
            out.append(rp)
    return out


def has_media(folder: Path, limit: int = 400) -> int:
    n = 0
    try:
        for i, p in enumerate(folder.rglob("*")):
            if i > limit * 40:
                break
            if p.is_file() and p.suffix.lower() in (VIDEO_EXT | IMAGE_EXT):
                n += 1
                if n >= limit:
                    break
    except (OSError, PermissionError):
        pass
    return n


def find_source(src: dict, roots: Sequence[Path], max_depth: int = 6
                ) -> Optional[Path]:
    """Locate a source folder by matching folder-name markers."""
    # 1. Try the known literal path first
    kp = src.get("known_path")
    if kp and not str(kp).startswith("http"):
        p = Path(kp)
        try:
            if p.exists() and has_media(p):
                return p
        except OSError:
            pass

    # 2. Walk the candidate roots looking for marker folder names
    markers = [m.lower() for m in src["markers"]]
    best: Optional[Path] = None
    best_n = 0
    for root in roots:
        base_depth = len(root.parts)
        try:
            for dirpath, dirnames, _ in os.walk(root, topdown=True):
                d = Path(dirpath)
                if len(d.parts) - base_depth > max_depth:
                    dirnames[:] = []
                    continue
                dirnames[:] = [x for x in dirnames
                               if not x.startswith((".", "$", "~"))]
                name = d.name.lower()
                if any(m in name for m in markers):
                    n = has_media(d)
                    if n > best_n:
                        best, best_n = d, n
        except (OSError, PermissionError):
            continue
    return best


def split_into_deployments(folder: Path, max_deployments: int = 12
                           ) -> List[tuple]:
    """A source folder usually contains one subfolder per camera or period.
    Those subfolders are the deployments. If there are none, the folder
    itself is a single deployment."""
    subs = []
    try:
        for d in sorted(folder.iterdir()):
            if d.is_dir() and not d.name.startswith((".", "$", "~")):
                n = has_media(d)
                if n:
                    subs.append((d.name, d, n))
    except (OSError, PermissionError):
        pass
    if len(subs) >= 2:
        subs.sort(key=lambda t: -t[2])
        return [(n, p) for n, p, _ in subs[:max_deployments]]
    return [(folder.name, folder)]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_frames(video: Path, out_dir: Path, every_sec: float,
                   max_frames: int, prefix: str) -> List[Path]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_sec)))
    out, idx, saved = [], 0, 0
    while saved < max_frames:
        if total and idx >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        dest = out_dir / f"{prefix}_{video.stem}_f{idx:06d}.jpg"
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(dest)
        saved += 1
        idx += step
    cap.release()
    return out


def load_detector(weights: str):
    p = Path(weights)
    if not p.exists():
        print(f"\n  {weights} not found - using whole frames instead of crops.")
        print("  SpeciesNet expects crops, so put MegaDetectorV6.pt in this"
              " folder and rerun\n  with --redo-extract for best results.\n")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ultralytics not installed - using whole frames.")
        print("  pip install ultralytics, then rerun with --redo-extract.\n")
        return None
    return YOLO(weights)


def crop_frames(detector, frames: Sequence[Path], out_dir: Path,
                conf: float, min_px: int) -> List[dict]:
    from PIL import Image
    recs = []
    for fp in frames:
        if detector is None:
            recs.append({"crop": fp, "frame": fp, "conf": -1.0, "blank": False})
            continue
        try:
            res = detector(str(fp), verbose=False)[0]
        except Exception:
            continue
        boxes = getattr(res, "boxes", None)
        kept = 0
        if boxes is not None and len(boxes) > 0:
            img = Image.open(fp).convert("RGB")
            for bi in range(len(boxes)):
                c = float(boxes.conf[bi])
                if c < conf:
                    continue
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[bi].tolist()]
                if (x2 - x1) < min_px or (y2 - y1) < min_px:
                    continue
                dest = out_dir / f"{fp.stem}_c{bi}.jpg"
                img.crop((x1, y1, x2, y2)).save(dest, quality=95)
                recs.append({"crop": dest, "frame": fp, "conf": c,
                             "blank": False})
                kept += 1
        if kept == 0:
            dest = out_dir / f"{fp.stem}_blank.jpg"
            shutil.copy2(fp, dest)
            recs.append({"crop": dest, "frame": fp, "conf": 0.0, "blank": True})
    return recs


# ---------------------------------------------------------------------------
# Split and build
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assign_splits(dep_counts: Dict[str, Counter], seed: int = 42
                  ) -> Dict[str, str]:
    """Whole deployments go to one split each. Near-identical frames from the
    same camera must never straddle train and test."""
    rng = random.Random(seed)
    totals = Counter()
    for c in dep_counts.values():
        totals.update(c)
    targets = {s: {k: totals[k] * r for k in totals}
               for s, r in zip(SPLITS, (0.70, 0.15, 0.15))}
    current = {s: Counter() for s in SPLITS}
    assign: Dict[str, str] = {}
    deps = sorted(dep_counts.items(), key=lambda kv: -sum(kv[1].values()))

    # Guarantee every class appears in every split
    for cls in sorted(totals, key=lambda c: totals[c]):
        for s in SPLITS:
            if current[s][cls] > 0:
                continue
            cand = [(k, c) for k, c in deps
                    if k not in assign and c.get(cls, 0) > 0]
            if not cand:
                continue
            pick = max if s == "train" else min
            k, c = pick(cand, key=lambda kv: kv[1].get(cls, 0))
            assign[k] = s
            current[s].update(c)

    # Everything else fills the biggest deficit
    for k, c in deps:
        if k in assign or not c:
            continue
        dom = c.most_common(1)[0][0]
        best, bd = None, None
        for s in SPLITS:
            d = targets[s].get(dom, 0) - current[s][dom] + rng.random() * 1e-6
            if bd is None or d > bd:
                bd, best = d, s
        assign[k] = best
        current[best].update(c)
    return assign


def build(work: Path, out: Path, args) -> int:
    label_path = Path(LABEL_FILE)
    rows = []
    with label_path.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lab = (r.get("label") or "").strip().lower()
            if lab in CLASSES:
                r["label"] = lab
                rows.append(r)
    if not rows:
        print(f"\nNo labels filled in yet in {LABEL_FILE}.")
        print("Fill the 'label' column with badger / otter / other, then rerun.")
        return 1

    dep_counts = defaultdict(Counter)
    for r in rows:
        dep_counts[r["deployment"]][r["label"]] += 1

    assign = assign_splits(dep_counts, args.seed)

    print("\nDeployment -> split")
    for k in sorted(assign):
        print(f"  {k:50s} {assign[k]:6s}  {dict(dep_counts[k])}")

    buckets = defaultdict(list)
    for r in rows:
        buckets[(assign[r["deployment"]], r["label"])].append(r)

    rng = random.Random(args.seed)
    chosen, gaps = [], []
    for s in SPLITS:
        sizes = {c: len(buckets[(s, c)]) for c in CLASSES}
        absent = [c for c, n in sizes.items() if n == 0]
        if absent:
            gaps.append((s, absent))
        present = [n for n in sizes.values() if n > 0]
        cap = int(min(present) * args.max_ratio) if present else 0
        for c in CLASSES:
            pool = buckets[(s, c)]
            rng.shuffle(pool)
            chosen.extend(pool[:cap])

    if gaps:
        print("\n*** CLASS COVERAGE PROBLEM ***")
        for s, absent in gaps:
            print(f"  '{s}' split has no {', '.join(absent)}")
        print("  A class missing from a split cannot be measured there.")
        print("  Usually means one class sits in a single deployment.")

    if out.exists() and args.reset:
        shutil.rmtree(out)
    for s in SPLITS:
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)

    seen, manifest, dupes = set(), [], 0
    for r in chosen:
        src = Path(r["crop_path"])
        if not src.exists():
            continue
        dg = sha256_file(src)
        if dg in seen:
            dupes += 1
            continue
        seen.add(dg)
        s = assign[r["deployment"]]
        dest = out / s / r["label"] / src.name
        shutil.copy2(src, dest)
        manifest.append({
            "split": s, "label": r["label"],
            "file": str(dest.relative_to(out)),
            "deployment": r["deployment"], "provider": r["provider"],
            "source_frame": r["source_frame"], "det_conf": r["det_conf"],
            "blank": r["blank"], "sha256": dg,
        })

    if not manifest:
        print("\nNothing selected. Check labels.")
        return 1

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    counts = defaultdict(Counter)
    for m in manifest:
        counts[m["split"]][m["label"]] += 1

    print(f"\nDuplicates removed: {dupes}")
    print("\n" + f"{'split':8s}" + "".join(f"{c:>10s}" for c in CLASSES)
          + f"{'total':>10s}")
    print("-" * 50)
    mins = {"train": args.min_train, "val": args.min_val, "test": args.min_test}
    ok = True
    for s in SPLITS:
        print(f"{s:8s}" + "".join(f"{counts[s][c]:10d}" for c in CLASSES)
              + f"{sum(counts[s].values()):10d}")
        if any(counts[s][c] < mins[s] for c in CLASSES):
            ok = False
    print("-" * 50)

    (out / "prep_report.json").write_text(json.dumps({
        "splits": {s: dict(counts[s]) for s in SPLITS},
        "deployment_assignment": assign,
        "duplicates_removed": dupes,
        "meets_minimums": ok,
    }, indent=2), encoding="utf-8")

    print(f"\nDataset ready: {out.resolve()}")
    print("Minimum counts met." if ok else
          f"Below minimums {mins} - more labelled data needed.")
    return 0


# ---------------------------------------------------------------------------

def discover_and_extract(work: Path, args) -> int:
    print("Searching for source folders...")
    roots = candidate_roots()
    for r in roots:
        print(f"  root: {r}")
    if not roots:
        print("  No searchable roots found.")

    found = []
    for src in KNOWN_SOURCES:
        p = find_source(src, roots)
        tag = f"{src['provider']} / {src['project']}"
        if p:
            print(f"  FOUND  {tag:42s} {p}")
            found.append((src, p))
        else:
            hint = src.get("known_path") or "not shared yet"
            print(f"  ---    {tag:42s} ({hint})")

    if args.extra:
        for raw in args.extra:
            p = Path(raw)
            if p.exists():
                found.append(({"provider": "Manual", "project": p.name,
                               "class_hint": "mixed"}, p))
                print(f"  ADDED  {'Manual / ' + p.name:42s} {p}")

    if not found:
        print("\nNo source data located on this machine.")
        print("Sync the SharePoint folders to OneDrive, or pass a folder:")
        print("  python prepare_camera_trap_dataset.py --extra \"C:\\path\\to\\footage\"")
        return 1

    detector = load_detector(args.detector)
    work.mkdir(parents=True, exist_ok=True)
    index = []

    for src, folder in found:
        for dep_name, dep_path in split_into_deployments(folder):
            key = f"{src['provider']}__{src['project']}__{dep_name}"
            key = "".join(ch if ch.isalnum() or ch in "_-" else "_"
                          for ch in key)
            fdir = work / "frames" / key
            cdir = work / "crops" / key
            fdir.mkdir(parents=True, exist_ok=True)
            cdir.mkdir(parents=True, exist_ok=True)

            vids = [p for p in dep_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT]
            imgs = [p for p in dep_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT]

            frames = []
            for v in vids:
                frames += extract_frames(v, fdir, args.every_sec,
                                         args.max_frames_per_clip, key)
            for im in imgs:
                dest = fdir / f"{key}_{im.stem}{im.suffix.lower()}"
                if not dest.exists():
                    shutil.copy2(im, dest)
                frames.append(dest)

            recs = crop_frames(detector, frames, cdir, args.det_conf,
                               args.min_crop_px)
            print(f"  {key:50s} {len(vids):4d} clips -> {len(recs):5d} crops")

            hint = src.get("class_hint", "mixed")
            for r in recs:
                index.append({
                    "label": hint if hint in CLASSES else "",
                    "deployment": key,
                    "provider": src["provider"],
                    "crop_path": str(r["crop"]),
                    "source_frame": str(r["frame"]),
                    "det_conf": f"{r['conf']:.4f}",
                    "blank": "1" if r["blank"] else "0",
                })

    if not index:
        print("\nNo frames produced.")
        return 1

    with Path(LABEL_FILE).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)

    prefilled = sum(1 for r in index if r["label"])
    print(f"\nWrote {LABEL_FILE} ({len(index)} rows, {prefilled} pre-labelled"
          " from folder origin)")
    print("Check the pre-filled labels, fill the blanks, then run again.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
          formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", nargs="*", help="Extra source folders")
    ap.add_argument("--detector", default="MegaDetectorV6.pt")
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--max-frames-per-clip", type=int, default=10)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--min-crop-px", type=int, default=48)
    ap.add_argument("--max-ratio", type=float, default=1.0)
    ap.add_argument("--min-train", type=int, default=400)
    ap.add_argument("--min-val", type=int, default=100)
    ap.add_argument("--min-test", type=int, default=100)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--redo-extract", action="store_true",
                    help="Re-extract even if review_and_label.csv exists")
    args = ap.parse_args()

    work = Path(WORK_DIR)
    out = Path(DATASET_DIR)

    if Path(LABEL_FILE).exists() and not args.redo_extract:
        print(f"{LABEL_FILE} found - building dataset.\n")
        return build(work, out, args)
    return discover_and_extract(work, args)


if __name__ == "__main__":
    raise SystemExit(main())
