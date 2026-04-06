from __future__ import annotations

import torch
from torch import nn


def conv_time_length(
    lengths: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    return (
        torch.div(
            lengths + 2 * padding - (kernel_size - 1) - 1, stride, rounding_mode="floor"
        )
        + 1
    )


class ConvBiGRUCTC(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int,
        vocab_size: int,
        conv_channels: tuple[int, int] = (32, 64),
        encoder_dim: int = 192,
        encoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2 = conv_channels
        self.frontend = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )

        reduced_mels = self._reduced_freq_bins(n_mels)
        gru_input_dim = c2 * reduced_mels
        self.encoder = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=encoder_dim,
            num_layers=encoder_layers,
            dropout=dropout if encoder_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim * 2, vocab_size)

    def _reduced_freq_bins(self, n_mels: int) -> int:
        length = torch.tensor([n_mels], dtype=torch.long)
        length = conv_time_length(length, kernel_size=3, stride=2, padding=1)
        length = conv_time_length(length, kernel_size=3, stride=2, padding=1)
        return int(length.item())

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.unsqueeze(1)
        x = self.frontend(x)

        output_lengths = feature_lengths.clone()
        output_lengths = conv_time_length(
            output_lengths, kernel_size=3, stride=2, padding=1
        )
        output_lengths = conv_time_length(
            output_lengths, kernel_size=3, stride=2, padding=1
        )

        batch_size, channels, freq_bins, time_steps = x.shape
        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size, time_steps, channels * freq_bins)
        )
        x, _ = self.encoder(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs, output_lengths
