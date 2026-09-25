#!/usr/bin/env python3
"""
Build and maintain the WSP model library (the OneDrive folder the app
installs models from). Standard library only, except `add-model`, which
reads class names from the checkpoint with torch when --classes is not
given.

Library layout (point the app at the "models" folder):

    WSP CameraTrap/models/models.json
    WSP CameraTrap/models/det/MD5A-0-0/md_v5a.0.0.pt
    WSP CameraTrap/models/cls/SPECIESNET-v4-0-2-A/...
    WSP CameraTrap/models/cls/WSP-UK-v1/{model.pt, inference.py, taxonomy.csv}

Examples:

    python wsp/tools/wsp_library.py init "C:/Users/me/OneDrive - WSP/WSP CameraTrap/models"
    python wsp/tools/wsp_library.py add-md LIB path/to/md_v5a.0.0.pt
    python wsp/tools/wsp_library.py add-speciesnet LIB path/to/speciesnet-pytorch-v4.0.2a
    python wsp/tools/wsp_library.py add-model LIB --checkpoint training/wsp_uk_v1.pth \\
        --id WSP-UK-v1 --name "WSP UK mammals v1" --taxonomy uk_taxonomy.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CATALOG = REPO_ROOT / "wsp" / "models.json"
SPECIESNET_INFERENCE = REPO_ROOT / "wsp" / "models" / "SPECIESNET-v4-0-2-A" / "inference.py"
TORCHVISION_INFERENCE = REPO_ROOT / "wsp" / "models" / "templates" / "torchvision_inference.py"
PROJECT_URL = "https://github.com/cameaci/Camera_Trap"

SPECIESNET_ID = "SPECIESNET-v4-0-2-A"
MD_ID = "MD5A-0-0"
TAXONOMY_FIELDS = ["model_class", "class", "order", "family", "genus", "species"]


def _load_catalog(lib: Path) -> dict:
    path = lib / "models.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": {"det": [], "cls": [], "emb": []}}


def _save_catalog(lib: Path, catalog: dict) -> None:
    path = lib / "models.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _upsert(catalog: dict, model_type: str, entry: dict) -> None:
    entries = catalog["models"].setdefault(model_type, [])
    entries[:] = [e for e in entries if e["model_id"] != entry["model_id"]]
    entries.append(entry)


def _shipped_entry(model_type: str, model_id: str) -> dict:
    catalog = json.loads(SHIPPED_CATALOG.read_text(encoding="utf-8"))
    for entry in catalog["models"][model_type]:
        if entry["model_id"] == model_id:
            return entry
    raise KeyError(model_id)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {dst}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def speciesnet_taxonomy_rows(labels_path: Path) -> list[dict]:
    """taxonomy.csv rows with the app's name de-duplication rule."""
    rows, seen = [], set()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(";")
        if len(parts) < 7:
            continue
        name = parts[6]
        if not name or name in seen:
            taxonomy = [p for p in parts[1:6] if p]
            if taxonomy:
                name = taxonomy[-1]
        if name in seen:
            name = f"{name} ({parts[0][:8]})"
        seen.add(name)
        rows.append(dict(zip(TAXONOMY_FIELDS, [name, *parts[1:6]])))
    return rows


def _write_taxonomy(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TAXONOMY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path} ({len(rows)} classes)")


# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    lib = Path(args.library)
    for t in ("det", "cls", "emb"):
        (lib / t).mkdir(parents=True, exist_ok=True)
    if not (lib / "models.json").is_file():
        _save_catalog(lib, {"models": {"det": [], "cls": [], "emb": []}})
    print(f"Library ready at {lib}")
    return 0


def cmd_add_md(args) -> int:
    lib, weights = Path(args.library), Path(args.weights)
    if weights.name != "md_v5a.0.0.pt":
        print("Expected md_v5a.0.0.pt (MegaDetector v5a)", file=sys.stderr)
        return 1
    _copy(weights, lib / "det" / MD_ID / weights.name)
    catalog = _load_catalog(lib)
    _upsert(catalog, "det", _shipped_entry("det", MD_ID))
    _save_catalog(lib, catalog)
    return 0


def cmd_add_speciesnet(args) -> int:
    lib, src = Path(args.library), Path(args.speciesnet_dir)
    entry = _shipped_entry("cls", SPECIESNET_ID)
    weights = src / entry["model_fname"]
    labels = sorted(src.glob("*.labels*.txt"))
    if not weights.is_file() or not labels:
        print(
            f"{src} does not look like SpeciesNet PyTorch v4.0.2a: expected "
            f"{entry['model_fname']} and a *.labels.txt file",
            file=sys.stderr,
        )
        return 1
    dst = lib / "cls" / SPECIESNET_ID
    for f in src.iterdir():
        if f.is_file():
            _copy(f, dst / f.name)
    _copy(SPECIESNET_INFERENCE, dst / "inference.py")
    _write_taxonomy(dst / "taxonomy.csv", speciesnet_taxonomy_rows(labels[-1]))
    catalog = _load_catalog(lib)
    _upsert(catalog, "cls", entry)
    _save_catalog(lib, catalog)
    return 0


def _class_names(args) -> list[str]:
    if args.classes:
        return [c.strip() for c in args.classes.split(",") if c.strip()]
    import torch  # only needed when --classes is not given

    names = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["class_names"]
    return [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)


def cmd_add_model(args) -> int:
    lib, checkpoint = Path(args.library), Path(args.checkpoint)
    names = _class_names(args)
    dst = lib / "cls" / args.id
    if dst.exists() and not args.replace:
        print(
            f"{dst} already exists. Publish a new id (e.g. WSP-UK-v2) so installed "
            f"copies update cleanly, or pass --replace.",
            file=sys.stderr,
        )
        return 1
    _copy(checkpoint, dst / "model.pt")
    _copy(TORCHVISION_INFERENCE, dst / "inference.py")

    rows = {n: {"model_class": n} for n in names}
    if args.taxonomy:
        with open(args.taxonomy, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("model_class") in rows:
                    rows[row["model_class"]] = row
    _write_taxonomy(dst / "taxonomy.csv", [
        {k: rows[n].get(k, "") for k in TAXONOMY_FIELDS} for n in names
    ])

    entry = {
        "model_id": args.id,
        "friendly_name": args.name or args.id,
        "emoji": "🇬🇧",
        "env": "pytorch",
        "model_fname": "model.pt",
        "description": args.description
        or f"WSP classifier trained on WSP camera trap data. Classes: {', '.join(names)}.",
        "description_short": f"WSP • {len(names)} classes",
        "developer": "WSP",
        "info_url": PROJECT_URL,
        "min_app_version": "0.1.0",
        "region": args.region,
        "species_list": names,
    }
    catalog = _load_catalog(lib)
    _upsert(catalog, "cls", entry)
    _save_catalog(lib, catalog)
    print(f"Published {args.id} (sha256 {_sha256(dst / 'model.pt')[:12]}…)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create an empty library")
    p.add_argument("library")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add-md", help="add MegaDetector v5a weights")
    p.add_argument("library")
    p.add_argument("weights")
    p.set_defaults(func=cmd_add_md)

    p = sub.add_parser("add-speciesnet", help="add Google's SpeciesNet PyTorch v4.0.2a folder")
    p.add_argument("library")
    p.add_argument("speciesnet_dir")
    p.set_defaults(func=cmd_add_speciesnet)

    p = sub.add_parser("add-model", help="publish a WSP classifier checkpoint")
    p.add_argument("library")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id", required=True, help="e.g. WSP-UK-v1; bump for every new version")
    p.add_argument("--name")
    p.add_argument("--description")
    p.add_argument("--classes", help="comma-separated, in model output order")
    p.add_argument("--taxonomy", help="CSV with model_class,class,order,family,genus,species")
    p.add_argument("--region", default="europe",
                   choices=["global", "africa", "americas", "asia", "europe", "oceania"])
    p.add_argument("--replace", action="store_true")
    p.set_defaults(func=cmd_add_model)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
