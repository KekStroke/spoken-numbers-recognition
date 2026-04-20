from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.asr.features import compute_log_mel_spectrogram
from src.asr.tokenizer import NumberTokenizer


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 20.0
    f_max: float | None = 7600.0
    # Augmentation — only applied when the dataset split is "train".
    aug_enabled: bool = False
    # Audio speed perturbation (librosa.effects.time_stretch, pitch-preserving).
    aug_speed_prob: float = 0.5
    aug_speed_min: float = 0.9
    aug_speed_max: float = 1.1
    # SpecAugment: freq masks (rows along n_mels axis).
    aug_freq_mask_num: int = 2
    aug_freq_mask_width: int = 15
    # SpecAugment: time masks (columns along time axis).
    aug_time_mask_num: int = 2
    aug_time_mask_width: int = 30
    aug_time_mask_ratio: float = 0.2  # cap mask width at this fraction of T


def spec_augment(
    features: np.ndarray,
    *,
    freq_mask_num: int,
    freq_mask_width: int,
    time_mask_num: int,
    time_mask_width: int,
    time_mask_ratio: float,
) -> np.ndarray:
    if freq_mask_num <= 0 and time_mask_num <= 0:
        return features
    feats = features.copy()
    n_mels, t = feats.shape
    fill = float(feats.min())
    for _ in range(max(0, freq_mask_num)):
        w = random.randint(0, max(0, freq_mask_width))
        if w <= 0 or n_mels - w <= 0:
            continue
        f0 = random.randint(0, n_mels - w)
        feats[f0 : f0 + w, :] = fill
    t_cap = max(1, min(time_mask_width, int(t * time_mask_ratio)))
    for _ in range(max(0, time_mask_num)):
        w = random.randint(0, t_cap)
        if w <= 0 or t - w <= 0:
            continue
        t0 = random.randint(0, t - w)
        feats[:, t0 : t0 + w] = fill
    return feats


class SpokenNumbersDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        tokenizer: NumberTokenizer,
        audio_config: AudioConfig,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.df = pd.read_csv(self.data_root / f"{split}.csv").copy()
        self.has_targets = "transcription" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        audio_path = self.data_root / str(row["filename"])
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != self.audio_config.sample_rate:
            raise ValueError(
                f"Expected {self.audio_config.sample_rate} Hz, got {sample_rate} for {audio_path}"
            )

        augment_active = (
            self.split == "train" and self.audio_config.aug_enabled
        )

        # Speed perturbation (audio-level). time_stretch preserves pitch and
        # operates on the waveform; the mel is recomputed after stretching.
        if augment_active and random.random() < self.audio_config.aug_speed_prob:
            speed = random.uniform(
                self.audio_config.aug_speed_min,
                self.audio_config.aug_speed_max,
            )
            if abs(speed - 1.0) > 1e-3:
                audio = librosa.effects.time_stretch(y=audio, rate=speed)

        features = compute_log_mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            n_mels=self.audio_config.n_mels,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length,
            win_length=self.audio_config.win_length,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
        )

        if augment_active:
            features = spec_augment(
                features,
                freq_mask_num=self.audio_config.aug_freq_mask_num,
                freq_mask_width=self.audio_config.aug_freq_mask_width,
                time_mask_num=self.audio_config.aug_time_mask_num,
                time_mask_width=self.audio_config.aug_time_mask_width,
                time_mask_ratio=self.audio_config.aug_time_mask_ratio,
            )

        item: dict[str, object] = {
            "features": torch.from_numpy(features),
            "feature_length": features.shape[1],
            "filename": str(row["filename"]),
            "spk_id": str(row.get("spk_id", "unknown")),
        }
        if self.has_targets:
            transcription = self.tokenizer.normalize_text(row["transcription"])
            target_ids = self.tokenizer.encode(transcription)
            item["text"] = transcription
            if hasattr(self.tokenizer, "encode_as_text"):
                item["token_text"] = self.tokenizer.encode_as_text(transcription)
            item["target"] = torch.tensor(target_ids, dtype=torch.long)
            item["target_length"] = len(target_ids)
        return item


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    feature_lengths = torch.tensor(
        [int(sample["feature_length"]) for sample in batch],
        dtype=torch.long,
    )
    max_frames = int(feature_lengths.max().item())
    n_mels = int(batch[0]["features"].shape[0])  # type: ignore[index]
    features = torch.zeros(len(batch), n_mels, max_frames, dtype=torch.float32)

    filenames: list[str] = []
    speakers: list[str] = []
    texts: list[str] = []
    token_texts: list[str] = []
    targets: list[torch.Tensor] = []
    target_lengths: list[int] = []

    for idx, sample in enumerate(batch):
        current = sample["features"]  # type: ignore[assignment]
        frames = int(sample["feature_length"])
        features[idx, :, :frames] = current  # type: ignore[index]
        filenames.append(str(sample["filename"]))
        speakers.append(str(sample["spk_id"]))
        if "target" in sample:
            texts.append(str(sample["text"]))
            token_texts.append(str(sample.get("token_text", sample["text"])))
            targets.append(sample["target"])  # type: ignore[arg-type]
            target_lengths.append(int(sample["target_length"]))

    result: dict[str, object] = {
        "features": features,
        "feature_lengths": feature_lengths,
        "filenames": filenames,
        "speakers": speakers,
    }
    if targets:
        result["texts"] = texts
        result["token_texts"] = token_texts
        result["targets"] = torch.cat(targets)
        result["target_lengths"] = torch.tensor(target_lengths, dtype=torch.long)
    return result
