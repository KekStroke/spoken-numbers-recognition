from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.asr.data import AudioConfig, SpokenNumbersDataset, collate_batch
from src.asr.model import ConvBiGRUCTC
from src.asr.tokenizer import NumberTokenizer, build_tokenizer
from src.train_baseline import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Kaggle submission from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_16k"))
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_model(
    checkpoint: dict[str, object],
    tokenizer: NumberTokenizer,
    device: torch.device,
) -> ConvBiGRUCTC:
    args = checkpoint["args"]
    audio_config = checkpoint["audio_config"]
    model = ConvBiGRUCTC(
        n_mels=int(audio_config["n_mels"]),
        vocab_size=tokenizer.vocab_size,
        encoder_dim=int(args["encoder_dim"]),
        encoder_layers=int(args["encoder_layers"]),
        dropout=float(args["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _posix_rel_path(path_like: str) -> str:
    return Path(path_like).as_posix()


def restore_original_filename(processed_filename: str, original_ext: str) -> str:
    ext = original_ext.removeprefix(".")
    return Path(processed_filename).with_suffix(f".{ext}").as_posix()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(checkpoint.get("tokenizer"))
    audio_config = AudioConfig(**checkpoint["audio_config"])

    dataset = SpokenNumbersDataset(
        data_root=args.data_root,
        split="test",
        tokenizer=tokenizer,
        audio_config=audio_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )

    model = build_model(checkpoint, tokenizer, device)
    test_df = pd.read_csv(args.data_root / "test.csv")
    ext_by_filename = {
        _posix_rel_path(str(row.filename)): str(row.ext)
        for row in test_df.itertuples(index=False)
    }

    predictions: list[dict[str, str]] = []
    for batch in tqdm(loader, desc="submit:test"):
        features = batch["features"].to(device)  # type: ignore[assignment]
        feature_lengths = batch["feature_lengths"].to(device)  # type: ignore[assignment]
        log_probs, output_lengths = model(features, feature_lengths)

        predicted_ids = log_probs.argmax(dim=-1).cpu().tolist()
        predicted_lengths = output_lengths.cpu().tolist()
        filenames = batch["filenames"]  # type: ignore[assignment]

        for idx, processed_filename in enumerate(filenames):
            frame_ids = predicted_ids[idx][: predicted_lengths[idx]]
            transcription = tokenizer.ctc_collapse(frame_ids)
            key = _posix_rel_path(str(processed_filename))
            original_filename = restore_original_filename(
                key,
                ext_by_filename[key],
            )
            predictions.append(
                {
                    "filename": original_filename,
                    "transcription": transcription or "0",
                }
            )

    submission_df = pd.DataFrame(predictions)
    sample_df = pd.read_csv(args.sample_submission)
    order_source = "test.csv"
    if len(sample_df) == len(submission_df):
        order_source = "sample_submission.csv"
        sample_order = sample_df["filename"].tolist()
        submission_df = (
            submission_df.set_index("filename").reindex(sample_order).reset_index()
        )
        if submission_df["transcription"].isna().any():
            missing = submission_df[submission_df["transcription"].isna()][
                "filename"
            ].tolist()[:10]
            raise ValueError(f"Missing predictions for filenames: {missing}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "rows": len(submission_df),
        "output": str(output_path),
        "order_source": order_source,
        "preview": submission_df.head(5).to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
