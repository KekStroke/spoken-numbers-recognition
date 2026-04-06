from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.asr.data import AudioConfig, SpokenNumbersDataset, collate_batch
from src.asr.metrics import cer
from src.asr.model import ConvBiGRUCTC
from src.asr.tokenizer import RussianNumberTokenizer, build_tokenizer
from src.train_baseline import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the baseline CTC model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_16k"))
    parser.add_argument(
        "--split", type=str, default="dev", choices=["train", "dev", "test"]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_model(
    checkpoint: dict[str, object],
    tokenizer: RussianNumberTokenizer,
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


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(checkpoint.get("tokenizer"))
    audio_config = AudioConfig(**checkpoint["audio_config"])

    dataset = SpokenNumbersDataset(
        data_root=args.data_root,
        split=args.split,
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.split}_predictions.csv"

    rows: list[dict[str, object]] = []
    total_cer = 0.0
    total_examples = 0
    speaker_scores: dict[str, list[float]] = {}

    for batch in tqdm(loader, desc=f"infer:{args.split}"):
        features = batch["features"].to(device)  # type: ignore[assignment]
        feature_lengths = batch["feature_lengths"].to(device)  # type: ignore[assignment]
        log_probs, output_lengths = model(features, feature_lengths)

        predicted_ids = log_probs.argmax(dim=-1).cpu().tolist()
        predicted_lengths = output_lengths.cpu().tolist()
        filenames = batch["filenames"]  # type: ignore[assignment]
        speakers = batch["speakers"]  # type: ignore[assignment]
        texts = batch.get("texts", [])
        token_texts = batch.get("token_texts", [])

        for idx, filename in enumerate(filenames):
            frame_ids = predicted_ids[idx][: predicted_lengths[idx]]
            prediction = tokenizer.ctc_collapse(frame_ids)
            row: dict[str, object] = {
                "filename": str(filename),
                "speaker": str(speakers[idx]),
                "prediction": prediction,
                "prediction_len": len(prediction),
                "prediction_words": tokenizer.ctc_collapse_words(frame_ids),
                "prediction_words_raw": tokenizer.ctc_collapse_words_raw(frame_ids),
            }
            if texts:
                target = str(texts[idx])
                score = cer(target, prediction)
                row["target"] = target
                row["target_len"] = len(target)
                row["target_words"] = str(token_texts[idx])
                row["cer"] = score
                total_cer += score
                total_examples += 1
                speaker_scores.setdefault(str(speakers[idx]), []).append(score)
            rows.append(row)

    fieldnames = (
        list(rows[0].keys())
        if rows
        else ["filename", "speaker", "prediction", "prediction_len"]
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "num_rows": len(rows),
        "predictions_path": str(predictions_path),
    }
    if total_examples:
        summary["mean_cer"] = total_cer / total_examples
        summary["speaker_cer"] = {
            speaker: sum(values) / len(values)
            for speaker, values in sorted(speaker_scores.items())
        }

        worst_rows = sorted(rows, key=lambda row: float(row["cer"]), reverse=True)[:20]
        worst_path = output_dir / f"{args.split}_worst_examples.json"
        with worst_path.open("w", encoding="utf-8") as fp:
            json.dump(worst_rows, fp, ensure_ascii=False, indent=2)
        summary["worst_examples_path"] = str(worst_path)

    summary_path = output_dir / f"{args.split}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
