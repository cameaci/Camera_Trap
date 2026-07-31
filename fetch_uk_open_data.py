"""Fetch UK open-licensed wildlife images from iNaturalist into dataset_uk_species."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib import error as url_error
from urllib import parse, request

from PIL import Image, UnidentifiedImageError

INAT_API_BASE = "https://api.inaturalist.org/v1"
SPLITS = ("train", "val", "test")
DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-nc")
SIZE_TOKEN_RE = re.compile(r"/(square|small|medium|large|original)\.")

DEFAULT_CLASS_TAXA = {
    "badger": [("Meles meles", 855297)],
    "otter": [("Lutra lutra", 41850)],
    "other": [
        ("Vulpes vulpes", 42069),
        ("Capreolus capreolus", 42184),
        ("Erinaceus europaeus", 43042),
        ("Lepus europaeus", 43128),
        ("Sciurus carolinensis", 46017),
    ],
}


@dataclass
class PhotoRecord:
    class_name: str
    species: str
    observation_id: int
    photo_id: int
    license_code: str
    url: str


@dataclass
class ObservationRecord:
    class_name: str
    species: str
    observation_id: int
    photos: List[PhotoRecord]


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str, expected_len: int, label: str) -> List[int]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != expected_len:
        raise ValueError(f"{label} expects {expected_len} comma-separated integers")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise ValueError(f"{label} contains non-integer values") from exc


def api_get_json(url: str, timeout: float = 30.0, retries: int = 5) -> Dict[str, object]:
    backoff = 1.5
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (url_error.URLError, url_error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = backoff ** attempt
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed API request after {retries} attempts: {url}") from last_error


def download_bytes(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = request.Request(url, headers={"User-Agent": "CameraTrapUK/1.0"})
            with request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (url_error.URLError, url_error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(1.2 ** attempt)
    raise RuntimeError(f"Failed to download image after {retries} attempts: {url}") from last_error


def normalize_photo_url(url: str, preferred_size: str = "large") -> str:
    return SIZE_TOKEN_RE.sub(f"/{preferred_size}.", url)


def image_extension_from_format(fmt: str | None) -> str:
    mapping = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }
    return mapping.get((fmt or "").upper(), ".jpg")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dataset_dirs(root: Path) -> None:
    for split in SPLITS:
        for class_name in DEFAULT_CLASS_TAXA:
            (root / split / class_name).mkdir(parents=True, exist_ok=True)


def count_images(root: Path) -> Dict[str, Dict[str, int]]:
    counts = {split: {class_name: 0 for class_name in DEFAULT_CLASS_TAXA} for split in SPLITS}
    for split in SPLITS:
        for class_name in DEFAULT_CLASS_TAXA:
            class_dir = root / split / class_name
            if class_dir.exists():
                counts[split][class_name] = len([path for path in class_dir.glob("*") if path.is_file()])
    return counts


def load_existing_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], set[str], Dict[str, Dict[str, int]]]:
    rows: list[dict[str, str]] = []
    hashes: set[str] = set()
    counts = {split: {class_name: 0 for class_name in DEFAULT_CLASS_TAXA} for split in SPLITS}
    if not manifest_path.exists():
        return rows, hashes, counts

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            sha_value = row.get("sha256", "")
            if sha_value:
                hashes.add(sha_value)
            split = row.get("split", "")
            class_name = row.get("class_name", "")
            if split in counts and class_name in counts[split]:
                counts[split][class_name] += 1
    return rows, hashes, counts


def fetch_observations_for_taxon(
    taxon_id: int,
    place_id: int,
    licenses_csv: str,
    per_page: int,
    max_observations: int,
    request_sleep: float,
) -> List[dict]:
    observations: List[dict] = []
    page = 1

    while True:
        params = {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": licenses_csv,
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }
        url = f"{INAT_API_BASE}/observations?{parse.urlencode(params)}"
        try:
            payload = api_get_json(url)
        except RuntimeError as exc:
            if observations:
                print(
                    f"WARNING: API paging interrupted for taxon_id={taxon_id} page={page}. "
                    f"Proceeding with {len(observations)} observations collected so far. Error: {exc}"
                )
                break
            raise
        results = payload.get("results", [])
        if not results:
            break

        observations.extend(results)
        if max_observations > 0 and len(observations) >= max_observations:
            observations = observations[:max_observations]
            break

        if len(results) < per_page:
            break

        page += 1
        if request_sleep > 0:
            time.sleep(request_sleep)

    return observations


def build_observation_records(
    class_name: str,
    species_name: str,
    observations: Sequence[dict],
    allowed_licenses: set[str],
) -> List[ObservationRecord]:
    grouped: List[ObservationRecord] = []
    for obs in observations:
        observation_id = int(obs.get("id"))
        photos = []
        for photo in obs.get("photos", []):
            raw_license = (photo.get("license_code") or "").strip().lower()
            if raw_license and raw_license not in allowed_licenses:
                continue

            photo_id = int(photo.get("id"))
            url = photo.get("original_url") or photo.get("url")
            if not url:
                continue
            url = normalize_photo_url(url, preferred_size="large")
            photos.append(
                PhotoRecord(
                    class_name=class_name,
                    species=species_name,
                    observation_id=observation_id,
                    photo_id=photo_id,
                    license_code=raw_license or "unknown",
                    url=url,
                )
            )

        if photos:
            grouped.append(
                ObservationRecord(
                    class_name=class_name,
                    species=species_name,
                    observation_id=observation_id,
                    photos=photos,
                )
            )
    return grouped


def interleave_observations(records: Sequence[ObservationRecord], rng: random.Random) -> List[ObservationRecord]:
    by_species: Dict[str, List[ObservationRecord]] = {}
    for rec in records:
        by_species.setdefault(rec.species, []).append(rec)

    for species in by_species:
        rng.shuffle(by_species[species])

    species_order = sorted(by_species.keys())
    interleaved: List[ObservationRecord] = []
    while True:
        pushed = 0
        for species in species_order:
            bucket = by_species[species]
            if bucket:
                interleaved.append(bucket.pop())
                pushed += 1
        if pushed == 0:
            break
    return interleaved


def choose_target_split(class_counts: Dict[str, int], split_targets: Dict[str, int]) -> str | None:
    candidates = [split for split in SPLITS if class_counts[split] < split_targets[split]]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda split: (
            (split_targets[split] - class_counts[split]) / max(1, split_targets[split]),
            split_targets[split] - class_counts[split],
        ),
    )


def validate_image_content(content: bytes, min_short_edge: int) -> tuple[bool, str, str]:
    try:
        with Image.open(BytesIO(content)) as img:
            width, height = img.size
            fmt = img.format or "JPEG"
            if min(width, height) < min_short_edge:
                return False, "", fmt
            return True, image_extension_from_format(fmt), fmt
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "", ""


def write_manifest(manifest_path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = ["class_name", "source", "observation_id", "photo_id", "license", "url", "sha256", "split"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch UK open data for badger/otter classifier training.")
    parser.add_argument("--out", default="dataset_uk_species", help="Output dataset root")
    parser.add_argument("--licenses", default="cc0,cc-by,cc-by-nc", help="Comma-separated iNat photo licenses")
    parser.add_argument("--uk-place-id", type=int, default=6857, help="iNaturalist place ID for UK")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--request-sleep", type=float, default=0.03, help="Sleep between API pages (seconds)")
    parser.add_argument("--per-page", type=int, default=200, help="iNat API per_page value")
    parser.add_argument("--max-observations-per-species", type=int, default=0, help="Optional cap per species; 0 means unlimited")
    parser.add_argument("--targets", default="600,150,150", help="Split targets as train,val,test counts per class")
    parser.add_argument("--min-short-edge", type=int, default=224, help="Minimum short-edge pixels")
    parser.add_argument("--reset", action="store_true", help="Delete existing dataset images and manifest before fetching")
    parser.add_argument("--manifest", default="", help="Manifest path; default is <out>/manifest.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    out_root = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out_root / "manifest.csv"
    split_values = parse_int_csv(args.targets, expected_len=3, label="--targets")
    split_targets = {split: split_values[idx] for idx, split in enumerate(SPLITS)}
    licenses = [item.lower() for item in parse_csv(args.licenses)]
    allowed_licenses = set(licenses)

    if args.reset:
        for split in SPLITS:
            split_dir = out_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        if manifest_path.exists():
            manifest_path.unlink()

    ensure_dataset_dirs(out_root)
    existing_rows, seen_hashes, existing_counts = load_existing_manifest(manifest_path)
    manifest_rows: list[dict[str, str]] = list(existing_rows)

    counts = {
        class_name: {split: existing_counts[split][class_name] for split in SPLITS}
        for class_name in DEFAULT_CLASS_TAXA
    }

    print("Starting fetch with settings:")
    print(
        json.dumps(
            {
                "out": str(out_root),
                "licenses": licenses,
                "uk_place_id": args.uk_place_id,
                "targets": split_targets,
                "min_short_edge": args.min_short_edge,
                "reset": args.reset,
            },
            indent=2,
        )
    )

    class_observations: Dict[str, List[ObservationRecord]] = {name: [] for name in DEFAULT_CLASS_TAXA}
    for class_name, taxa in DEFAULT_CLASS_TAXA.items():
        for species_name, taxon_id in taxa:
            observations = fetch_observations_for_taxon(
                taxon_id=taxon_id,
                place_id=args.uk_place_id,
                licenses_csv=",".join(licenses),
                per_page=args.per_page,
                max_observations=args.max_observations_per_species,
                request_sleep=args.request_sleep,
            )
            grouped = build_observation_records(
                class_name=class_name,
                species_name=species_name,
                observations=observations,
                allowed_licenses=allowed_licenses,
            )
            class_observations[class_name].extend(grouped)
            photo_total = sum(len(item.photos) for item in grouped)
            print(
                f"Collected {len(grouped)} observations / {photo_total} photos "
                f"for class={class_name} species={species_name}"
            )

    for class_name in class_observations:
        class_observations[class_name] = interleave_observations(class_observations[class_name], rng=rng)

    downloaded = 0
    skipped_duplicates = 0
    skipped_quality = 0
    skipped_download_errors = 0

    for class_name, observations in class_observations.items():
        rng.shuffle(observations)
        for obs in observations:
            split = choose_target_split(counts[class_name], split_targets=split_targets)
            if split is None:
                break

            class_dir = out_root / split / class_name
            saved_from_observation = 0
            for photo in obs.photos:
                if counts[class_name][split] >= split_targets[split]:
                    break

                try:
                    content = download_bytes(photo.url)
                except RuntimeError:
                    skipped_download_errors += 1
                    continue

                digest = sha256_bytes(content)
                if digest in seen_hashes:
                    skipped_duplicates += 1
                    continue

                ok, ext, _fmt = validate_image_content(content, min_short_edge=args.min_short_edge)
                if not ok:
                    skipped_quality += 1
                    continue

                file_name = f"{photo.observation_id}_{photo.photo_id}_{digest[:12]}{ext}"
                output_path = class_dir / file_name
                output_path.write_bytes(content)

                seen_hashes.add(digest)
                downloaded += 1
                saved_from_observation += 1
                counts[class_name][split] += 1

                manifest_rows.append(
                    {
                        "class_name": class_name,
                        "source": "inaturalist",
                        "observation_id": str(photo.observation_id),
                        "photo_id": str(photo.photo_id),
                        "license": photo.license_code,
                        "url": photo.url,
                        "sha256": digest,
                        "split": split,
                    }
                )

            if saved_from_observation == 0:
                continue

    write_manifest(manifest_path, manifest_rows)

    print("Fetch complete.")
    print(f"Downloaded new images: {downloaded}")
    print(f"Skipped duplicate images: {skipped_duplicates}")
    print(f"Skipped low-quality/corrupt images: {skipped_quality}")
    print(f"Skipped download errors: {skipped_download_errors}")
    print("Per-class split counts:")
    print(json.dumps(counts, indent=2))

    unmet = []
    for class_name in DEFAULT_CLASS_TAXA:
        for split in SPLITS:
            if counts[class_name][split] < split_targets[split]:
                unmet.append(f"{class_name}/{split}: {counts[class_name][split]} < {split_targets[split]}")

    if unmet:
        print("WARNING: Some split targets were not met.")
        for row in unmet:
            print(f"- {row}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
