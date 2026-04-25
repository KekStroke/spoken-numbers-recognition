"""
Normalize competition audio: mono WAV @ 16 kHz. Clip dev/test to max duration (default 7 s).

Typical usage (from repo root, after downloading data):

  uv run python src/preprocess_audio.py \\
    --data-root data/competitions/asr-2026-spoken-numbers-recognition-challenge \\
    --output-dir data/processed_16k

Point training code at --output-dir (CSVs there reference the same relative paths).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder containing train.csv, dev.csv, test.csv and train/, dev/, test/ audio.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Write mirrored tree of .wav files here (and copy/update CSVs).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help="Comma-separated splits to process (default: train,dev,test).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000).",
    )
    p.add_argument(
        "--clip-seconds",
        type=float,
        default=7.0,
        help="Max duration in seconds for dev and test after resample (default: 7).",
    )
    p.add_argument(
        "--clip-splits",
        type=str,
        default="train,dev,test",
        help="Comma-separated splits to apply clipping (default: train,dev,test).",
    )
    p.add_argument(
        "--copy-csv",
        action="store_true",
        help="Copy train/dev/test.csv into output-dir (paths stay relative; use output-dir as DATA_ROOT).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output WAVs.",
    )
    return p.parse_args()


def dest_wav_path(output_root: Path, rel_file: str) -> Path:
    p = Path(rel_file)
    return output_root / p.with_suffix(".wav")


def load_resample_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def maybe_clip(y: np.ndarray, sr: int, max_sec: float | None) -> np.ndarray:
    if max_sec is None or max_sec <= 0:
        return y
    max_samples = int(max_sec * sr)
    if len(y) > max_samples:
        return y[:max_samples]
    return y


def process_one(
    src: Path,
    dst: Path,
    *,
    target_sr: int,
    clip_max_sec: float | None,
    overwrite: bool,
) -> str | None:
    if dst.is_file() and not overwrite:
        return "skip_exists"
    try:
        y, sr = load_resample_mono(src, target_sr)
        y = maybe_clip(y, sr, clip_max_sec)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), y, sr, subtype="PCM_16")
    except Exception as e:
        return f"error: {e}"
    return None


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.output_dir.resolve()
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    valid_splits = {"train", "dev", "test"}
    invalid = [s for s in splits if s not in valid_splits]
    if invalid:
        print(f"Invalid --splits values: {invalid}", file=sys.stderr)
        return 1
    if not splits:
        print("No splits provided via --splits", file=sys.stderr)
        return 1
    clip_splits = {s.strip().lower() for s in args.clip_splits.split(",") if s.strip()}

    for split in splits:
        name = f"{split}.csv"
        p = data_root / name
        if not p.is_file():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    n_ok = n_skip = n_err = 0
    errors: list[str] = []

    for split in splits:
        csv_path = data_root / f"{split}.csv"
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            print(f"No 'filename' column in {csv_path}", file=sys.stderr)
            return 1

        clip_here = split in clip_splits
        max_sec = args.clip_seconds if clip_here else None

        for rel in tqdm(df["filename"].astype(str).unique(), desc=f"{split} audio"):
            src = data_root / rel
            if not src.is_file():
                msg = f"missing source: {src}"
                errors.append(msg)
                n_err += 1
                continue
            dst = dest_wav_path(out_root, rel)
            err = process_one(
                src,
                dst,
                target_sr=args.sample_rate,
                clip_max_sec=max_sec,
                overwrite=args.overwrite,
            )
            if err == "skip_exists":
                n_skip += 1
            elif err:
                errors.append(f"{rel}: {err}")
                n_err += 1
            else:
                n_ok += 1

    print(f"Done. wrote={n_ok} skipped={n_skip} errors={n_err}")

    if args.copy_csv:
        out_root.mkdir(parents=True, exist_ok=True)
        for split in splits:
            name = f"{split}.csv"
            src_c = data_root / name
            dst_c = out_root / name
            shutil.copy2(src_c, dst_c)
            df = pd.read_csv(dst_c)

            # Normalize filename extensions to .wav under the same relative paths
            def to_wav(rel: str) -> str:
                return str(Path(rel).with_suffix(".wav"))

            df["filename"] = df["filename"].map(to_wav)
            df.to_csv(dst_c, index=False)
        print(f"Copied CSVs to {out_root} (filename column uses .wav paths).")

    if errors:
        print("First errors:", file=sys.stderr)
        for line in errors[:20]:
            print(line, file=sys.stderr)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
