from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.asr.data import AudioConfig, SpokenNumbersDataset, collate_batch
from src.asr.metrics import cer, compute_domain_cer_summary
from src.asr.model import ConvBiGRUCTC
from src.asr.tokenizer import NumberTokenizer, build_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline CTC model for spoken numbers."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_16k"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/baseline_ctc")
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--encoder-dim", type=int, default=192)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-parameters",
        type=int,
        default=5_000_000,
        help="Fail if model has more trainable params than this limit.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save checkpoint after every epoch in output-dir/checkpoints.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="russian_number_words",
        choices=["russian_number_words", "russian_number_compact"],
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested != "auto":
        return torch.device(requested)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloader(
    *,
    data_root: Path,
    split: str,
    tokenizer: NumberTokenizer,
    audio_config: AudioConfig,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = SpokenNumbersDataset(
        data_root=data_root,
        split=split,
        tokenizer=tokenizer,
        audio_config=audio_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
    )


def move_batch_to_device(
    batch: dict[str, object], device: torch.device
) -> tuple[torch.Tensor, ...]:
    features = batch["features"].to(device)  # type: ignore[assignment]
    feature_lengths = batch["feature_lengths"].to(device)  # type: ignore[assignment]
    targets = batch["targets"].to(device)  # type: ignore[assignment]
    target_lengths = batch["target_lengths"].to(device)  # type: ignore[assignment]
    return features, feature_lengths, targets, target_lengths


def compute_ctc_loss(
    criterion: nn.CTCLoss,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if device.type == "mps":
        # PyTorch does not currently implement aten::_ctc_loss on MPS, so we
        # keep the model on MPS and execute only the loss on CPU.
        return criterion(
            log_probs.float().cpu().transpose(0, 1),
            targets.cpu(),
            output_lengths.cpu(),
            target_lengths.cpu(),
        )

    return criterion(
        log_probs.transpose(0, 1),
        targets,
        output_lengths,
        target_lengths,
    )


def train_one_epoch(
    model: ConvBiGRUCTC,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        features, feature_lengths, targets, target_lengths = move_batch_to_device(
            batch, device
        )
        optimizer.zero_grad(set_to_none=True)
        log_probs, output_lengths = model(features, feature_lengths)
        loss = compute_ctc_loss(
            criterion,
            log_probs,
            targets,
            output_lengths,
            target_lengths,
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: ConvBiGRUCTC,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: NumberTokenizer,
    device: torch.device,
    in_domain_speakers: set[str] | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_cer = 0.0
    speaker_cers: dict[str, list[float]] = defaultdict(list)
    preview: list[dict[str, str]] = []

    for batch in tqdm(loader, desc="dev", leave=False):
        features, feature_lengths, targets, target_lengths = move_batch_to_device(
            batch, device
        )
        log_probs, output_lengths = model(features, feature_lengths)
        loss = compute_ctc_loss(
            criterion,
            log_probs,
            targets,
            output_lengths,
            target_lengths,
            device,
        )

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        predicted_ids = log_probs.argmax(dim=-1).cpu().tolist()
        predicted_lengths = output_lengths.cpu().tolist()
        texts = batch["texts"]  # type: ignore[assignment]
        token_texts = batch["token_texts"]  # type: ignore[assignment]
        speakers = batch["speakers"]  # type: ignore[assignment]
        filenames = batch["filenames"]  # type: ignore[assignment]

        for idx, ref in enumerate(texts):
            frame_ids = predicted_ids[idx][: predicted_lengths[idx]]
            hyp = tokenizer.ctc_collapse(frame_ids)
            sample_cer = cer(str(ref), hyp)
            total_cer += sample_cer
            speaker_cers[str(speakers[idx])].append(sample_cer)
            if len(preview) < 8:
                preview.append(
                    {
                        "filename": str(filenames[idx]),
                        "speaker": str(speakers[idx]),
                        "reference": str(ref),
                        "reference_words": str(token_texts[idx]),
                        "prediction": hyp,
                        "prediction_words": tokenizer.ctc_collapse_words(frame_ids),
                        "prediction_words_raw": tokenizer.ctc_collapse_words_raw(
                            frame_ids
                        ),
                    }
                )

    mean_speaker_cer = {
        speaker: sum(values) / len(values)
        for speaker, values in sorted(speaker_cers.items())
    }
    metrics: dict[str, object] = {
        "loss": total_loss / max(total_examples, 1),
        "cer": total_cer / max(total_examples, 1),
        "speaker_cer": mean_speaker_cer,
        "preview": preview,
    }
    if in_domain_speakers:
        domain_summary = compute_domain_cer_summary(speaker_cers, in_domain_speakers)
        metrics["in_domain_cer"] = domain_summary.in_domain_cer
        metrics["out_of_domain_cer"] = domain_summary.out_of_domain_cer
        metrics["primary_hmean_cer"] = domain_summary.harmonic_mean_cer
        metrics["in_domain_samples"] = domain_summary.in_domain_count
        metrics["out_of_domain_samples"] = domain_summary.out_of_domain_count
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def format_model_info(model: nn.Module, device: torch.device) -> dict[str, object]:
    num_parameters = count_parameters(model)
    return {
        "device": str(device),
        "num_parameters": num_parameters,
        "num_parameters_millions": round(num_parameters / 1_000_000, 3),
        "fits_5m_limit": num_parameters <= 5_000_000,
    }


def cer_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def speaker_cer_to_percent(speaker_cer: dict[str, float]) -> dict[str, float]:
    return {speaker: float(cer_val) * 100.0 for speaker, cer_val in speaker_cer.items()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    tokenizer = build_tokenizer({"type": args.tokenizer, "blank_id": 0})
    audio_config = AudioConfig(n_mels=args.n_mels)
    train_loader = build_dataloader(
        data_root=args.data_root,
        split="train",
        tokenizer=tokenizer,
        audio_config=audio_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dev_loader = build_dataloader(
        data_root=args.data_root,
        split="dev",
        tokenizer=tokenizer,
        audio_config=audio_config,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    train_speakers = {
        str(speaker)
        for speaker in train_loader.dataset.df["spk_id"].dropna().astype(str).tolist()
    }

    model = ConvBiGRUCTC(
        n_mels=args.n_mels,
        vocab_size=tokenizer.vocab_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model_info = format_model_info(model, device)
    print(json.dumps(model_info, ensure_ascii=False, indent=2))
    if not model_info["fits_5m_limit"] or int(model_info["num_parameters"]) > int(
        args.max_parameters
    ):
        raise ValueError(
            "Model exceeds parameter limit: "
            f"{model_info['num_parameters']} > {args.max_parameters}"
        )

    best_primary = float("inf")
    best_ood = float("inf")
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_metrics = evaluate(
            model,
            dev_loader,
            criterion,
            tokenizer,
            device,
            in_domain_speakers=train_speakers,
        )
        primary_hmean = dev_metrics.get("primary_hmean_cer")
        in_domain_cer = dev_metrics.get("in_domain_cer")
        out_domain_cer = dev_metrics.get("out_of_domain_cer")
        speaker_cer_raw = dev_metrics["speaker_cer"]
        assert isinstance(speaker_cer_raw, dict)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_cer": cer_to_percent(float(dev_metrics["cer"])),
            "dev_primary_hmean_cer": cer_to_percent(
                float(primary_hmean) if primary_hmean is not None else None
            ),
            "dev_in_domain_cer": cer_to_percent(
                float(in_domain_cer) if in_domain_cer is not None else None
            ),
            "dev_out_of_domain_cer": cer_to_percent(
                float(out_domain_cer) if out_domain_cer is not None else None
            ),
            "speaker_cer": speaker_cer_to_percent(
                {str(k): float(v) for k, v in speaker_cer_raw.items()}
            ),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if args.save_all_checkpoints:
            checkpoints_dir = output_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint = {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.config,
                "audio_config": audio_config.__dict__,
                "args": vars(args),
                "metrics": dev_metrics,
                "epoch": epoch,
                "model_info": model_info,
            }
            torch.save(epoch_checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pt")

        current_primary = dev_metrics.get("primary_hmean_cer")
        if current_primary is None:
            current_primary = float(dev_metrics["cer"])
        current_primary = float(current_primary)
        current_ood = float(dev_metrics.get("out_of_domain_cer") or float("inf"))

        if (
            current_primary < best_primary
            or (
                abs(current_primary - best_primary) <= 1e-12
                and current_ood < best_ood
            )
        ):
            best_primary = current_primary
            best_ood = current_ood
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.config,
                "audio_config": audio_config.__dict__,
                "args": vars(args),
                "metrics": dev_metrics,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            with (output_dir / "best_preview.json").open("w", encoding="utf-8") as fp:
                json.dump(dev_metrics["preview"], fp, ensure_ascii=False, indent=2)

    with (output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
