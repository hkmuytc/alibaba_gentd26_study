import math
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.pe(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class CNNLSTMForecaster(nn.Module):
    """
    Hybrid: 1D dilated convolutions extract local patterns, then a small GRU
    captures temporal dependencies. More parameter-efficient than pure RNN for
    high-resolution inputs.
    """

    def __init__(self, input_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=2, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(16, hidden_dim, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, T, F) → conv expects (B, F, T)
        c = self.conv(x.transpose(1, 2))  # (B, 16, T)
        c = c.transpose(1, 2)  # (B, T, 16)
        out, _ = self.gru(c)
        return self.head(out[:, -1, :]).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerHistoryOnly(nn.Module):
    """Transformer using only GPU utilization and its derived features (11 of 27).
    Excludes exogenous workload telemetry (QPS, pod ratios, memory, time-of-day).
    Ablation: tests whether power history dynamics (lags, rolling stats, rate of change,
    fractional differencing) suffice without workload-side features."""

    GPU_COLS = list(range(10))  # gpu_util group: columns 0–9 (pure GPU features, no exogenous data)

    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(len(self.GPU_COLS), d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = x[:, :, self.GPU_COLS]  # (B, T, 10)
        x = self.proj(x)
        x = self.pe(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def build_model(name, input_dim, hidden_dim=32, **kwargs):
    if name == "lstm":
        model = LSTMForecaster(input_dim, hidden_dim=hidden_dim)
    elif name == "gru":
        model = GRUForecaster(input_dim, hidden_dim=hidden_dim)
    elif name == "transformer":
        model = TransformerForecaster(input_dim, d_model=hidden_dim)
    elif name == "transformer_hist":
        model = TransformerHistoryOnly(input_dim, d_model=hidden_dim)
    elif name == "cnn_lstm":
        model = CNNLSTMForecaster(input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model: {name}")
    print(f"  {name}: {count_params(model):,} parameters")
    return model
