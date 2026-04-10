"""
LSTM Encoder-Decoder with Bahdanau (Additive) Attention.

Architecture:
  Encoder:
    - Stacked LSTM over the input sequence (enc_len timesteps, n_features each)
    - Outputs a hidden state + all encoder hidden states (for attention)

  Decoder:
    - Unrolled step-by-step for dec_len timesteps
    - At each step, computes attention over all encoder outputs
    - Context vector is concatenated with the previous prediction and fed into
      the decoder LSTM cell
    - A small FC head maps hidden state -> scalar discharge prediction

  Output:
    - (batch, dec_len) -- predicted log-discharge values (normalised)
    - Inverse-transform with tgt_scaler then expm1() to get m³/s

This model is compact but powerful -- Bahdanau attention helps the model
focus on the most relevant past rainfall/discharge windows when predicting
each future step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.

    Computes alignment scores between decoder hidden state and all encoder
    hidden states, returns a weighted context vector.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, enc_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expand decoder hidden to match encoder sequence length
        dec_h = decoder_hidden.unsqueeze(1)                 # (batch, 1, hidden)
        energy = torch.tanh(
            self.W_enc(encoder_outputs) + self.W_dec(dec_h) # (batch, enc_len, hidden)
        )
        scores = self.v(energy).squeeze(-1)                 # (batch, enc_len)
        weights = F.softmax(scores, dim=-1)                 # (batch, enc_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden)
        context = context.squeeze(1)                                 # (batch, hidden)
        return context, weights


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, enc_len, input_size)
        outputs, (h_n, c_n) = self.lstm(x)
        # outputs: (batch, enc_len, hidden_size) -- all timestep hidden states
        # h_n, c_n: (num_layers, batch, hidden_size)
        return outputs, h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        fc_hidden: int,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.hidden_size = hidden_size

        # Decoder input = [previous prediction (1) + context vector (hidden_size if attention)]
        decoder_input_size = 1 + (hidden_size if use_attention else 0)

        self.lstm = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if use_attention:
            self.attention = BahdanauAttention(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward_step(
        self,
        prev_pred: torch.Tensor,       # (batch, 1)
        h: torch.Tensor,               # (num_layers, batch, hidden)
        c: torch.Tensor,
        encoder_outputs: torch.Tensor, # (batch, enc_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_attention:
            # Use top-layer hidden state for attention query
            context, _ = self.attention(h[-1], encoder_outputs)  # (batch, hidden)
            dec_input = torch.cat([prev_pred, context], dim=-1)   # (batch, 1+hidden)
        else:
            dec_input = prev_pred  # (batch, 1)

        dec_input = dec_input.unsqueeze(1)  # (batch, 1, input_size)
        out, (h, c) = self.lstm(dec_input, (h, c))
        pred = self.fc(out.squeeze(1))  # (batch, 1)
        return pred, h, c

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        dec_len: int,
        teacher_forcing_ratio: float = 0.0,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = encoder_outputs.size(0)
        prev_pred = torch.zeros(batch_size, 1, device=encoder_outputs.device)
        predictions = []

        for t in range(dec_len):
            pred, h, c = self.forward_step(prev_pred, h, c, encoder_outputs)
            predictions.append(pred)

            # Teacher forcing: occasionally use the true value as next input
            if (
                teacher_forcing_ratio > 0.0
                and targets is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                prev_pred = targets[:, t].unsqueeze(1)
            else:
                prev_pred = pred

        return torch.cat(predictions, dim=1)  # (batch, dec_len)


class FloodForecastModel(nn.Module):
    """
    Full encoder-decoder model for multi-step discharge forecasting.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate (applied between LSTM layers and in FC head).
        encoder_len: Length of the input (encoder) sequence.
        decoder_len: Number of future timesteps to predict.
        fc_hidden: Hidden dimension of the FC output head.
        use_attention: Whether to use Bahdanau attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        encoder_len: int = 30,
        decoder_len: int = 7,
        fc_hidden: int = 64,
        use_attention: bool = True,
    ):
        super().__init__()
        self.decoder_len = decoder_len

        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, num_layers, dropout, fc_hidden, use_attention)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, enc_len, input_size)
        enc_outputs, h_n, c_n = self.encoder(x)
        preds = self.decoder(
            enc_outputs, h_n, c_n,
            dec_len=self.decoder_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
            targets=targets,
        )
        return preds  # (batch, dec_len)

    @classmethod
    def from_config(cls, cfg, input_size: int) -> "FloodForecastModel":
        """Instantiate from a Hydra model config node."""
        return cls(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            encoder_len=cfg.encoder_len,
            decoder_len=cfg.decoder_len,
            fc_hidden=cfg.fc_hidden,
            use_attention=cfg.attention,
        )
