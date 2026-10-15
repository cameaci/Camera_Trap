"""Benchmark single-crop classifier inference latency on CPU/GPU."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from PIL import Image

from wildlife_classifier import WildlifeClassificationAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark species classifier latency.")
    parser.add_argument("--weights", default="wildlife_model_badger_otter.pth", help="Model checkpoint path")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--image", default="", help="Optional image path for benchmark input")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=200, help="Measured iterations")
    return parser


def load_image(path: str) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(array)


def main() -> int:
    args = build_parser().parse_args()
    model = WildlifeClassificationAdapter(weights_path=args.weights, device=args.device)
    image = load_image(args.image)

    for _ in range(args.warmup):
        model.single_image_classification(image)

    timings_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        model.single_image_classification(image)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    median_ms = statistics.median(timings_ms)
    p95_ms = sorted(timings_ms)[int(0.95 * (len(timings_ms) - 1))]
    mean_ms = statistics.fmean(timings_ms)

    print(f"median_ms={median_ms:.2f}")
    print(f"p95_ms={p95_ms:.2f}")
    print(f"mean_ms={mean_ms:.2f}")
    print("pass_under_80ms=" + str(median_ms < 80.0).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

