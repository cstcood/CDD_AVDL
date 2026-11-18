"""Batch-extract audio and visual features for a directory of recordings.

    python scripts/extract_features.py --manifest manifest.csv --root data \
        --audio-dir audio --visual-dir visual

The manifest has the columns sample_id,wav,video.  Audio features come from
wav2vec2 and visual features from OpenGraphAU; both extractors are optional
dependencies, so this script only runs on a machine that has them installed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdd_avdl.data.features import extract_audio_features, extract_visual_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract CDD-AVDL features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--visual-dir", default="visual")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--graphau-root", default=None)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    audio_out = root / args.audio_dir
    visual_out = root / args.visual_dir
    audio_out.mkdir(parents=True, exist_ok=True)
    visual_out.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("manifest %s is empty" % args.manifest)

    for row in rows:
        sample_id = str(row["sample_id"]).strip()
        if not args.skip_audio and row.get("wav"):
            array = extract_audio_features(Path(row["wav"]), args.model, args.device)
            np.save(audio_out / (sample_id + ".npy"), array)
            print("audio %s %s" % (sample_id, array.shape))
        if not args.skip_visual and row.get("video"):
            array = extract_visual_features(
                Path(row["video"]),
                Path(args.graphau_root) if args.graphau_root else None,
                args.device,
            )
            np.save(visual_out / (sample_id + ".npy"), array)
            print("visual %s %s" % (sample_id, array.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
