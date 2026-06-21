"""
Training and evaluation pipeline for time-series forecasting models.
Handles dataset creation, training loop with early stopping, and metric computation.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

from ..data_processing.loading import clean_processed_power_frame
from ..models.architectures import get_model

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "saved"


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, window_size=24):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window_size]
        y = self.targets[idx + self.window_size]
        return x, y


def prepare_data(df, target_col="power_total_kw", feature_cols=None,
                 window_size=24, train_ratio=0.8, clean_identifiers=True):
    df = clean_processed_power_frame(df).copy() if clean_identifiers else df.copy()

    if feature_cols is None:
        exclude = ["timestamp", "container_id", "worker_name", "machine",
                    "start_time", "end_time", "time_offset"]
        preferred = [
            "gpu_util_frac",
            "gpu_mem_util",
            "mem_util_frac",
            "qps",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            target_col,
            f"{target_col}_roc",
            f"{target_col}_roll_mean_12",
            f"{target_col}_roll_std_12",
            f"{target_col}_roll_mean_72",
        ]
        feature_cols = [c for c in preferred if c in df.columns and c not in exclude]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in exclude and c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]

    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    # Temporal split
    split_idx = int(len(features) * train_ratio)

    # Scale features based on training set only
    scaler = StandardScaler()
    features[:split_idx] = scaler.fit_transform(features[:split_idx])
    features[split_idx:] = scaler.transform(features[split_idx:])

    # Scale target separately
    target_scaler = StandardScaler()
    targets_scaled = targets.copy()
    targets_scaled[:split_idx] = target_scaler.fit_transform(
        targets[:split_idx].reshape(-1, 1)
    ).flatten()
    targets_scaled[split_idx:] = target_scaler.transform(
        targets[split_idx:].reshape(-1, 1)
    ).flatten()

    train_dataset = TimeSeriesDataset(
        features[:split_idx], targets_scaled[:split_idx], window_size
    )
    test_dataset = TimeSeriesDataset(
        features[split_idx:], targets_scaled[split_idx:], window_size
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "feature_names": feature_cols,
        "input_dim": len(feature_cols),
        "split_idx": split_idx,
        "window_size": window_size,
        "raw_targets_test": targets[split_idx + window_size:],
        "timestamps_test": df["timestamp"].values[split_idx + window_size:] if "timestamp" in df.columns else None,
    }


def train_model(model, train_loader, val_loader=None, epochs=100,
                lr=1e-3, patience=10, grad_clip=1.0, device="cpu",
                progress_cb=None):
    """
    Train a model with early stopping, LR scheduling, and gradient clipping.
    Returns training history.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
        else:
            avg_val_loss = avg_train_loss

        history["val_loss"].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Early stopping
        stopped = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped = True

        if progress_cb is not None:
            progress_cb({
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "best_val_loss": float(best_val_loss),
                "wait": wait,
                "patience": patience,
                "stopped": stopped,
            })

        if stopped:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(model, test_loader, target_scaler, device="cpu"):
    """
    Evaluate model and return predictions + metrics.
    Metrics: MAE, RMSE, MAPE, R²
    """
    model = model.to(device)
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # Inverse transform to original scale
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    # Compute metrics
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    # MAPE: avoid division by zero
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    else:
        mape = float("inf")
    r2 = r2_score(targets, preds)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }

    return preds, targets, metrics


def save_model(model, model_name, dataset_name, metrics, history,
               window_size=24, train_ratio=0.8, input_dim=None,
               feature_names=None):
    """Save model weights, metrics, history, and a manifest with metadata.

    File naming: {model}_{dataset}_w{window}_{YYYYMMDD}_{HHMMSS}_*
    Also writes a _manifest.json with all training metadata.
    """
    from datetime import datetime
    os.makedirs(MODELS_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_name}_{dataset_name}_w{window_size}_{ts}"

    torch.save(model.state_dict(), MODELS_DIR / f"{prefix}_weights.pt")

    with open(MODELS_DIR / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(MODELS_DIR / f"{prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    manifest = {
        "model_name": model_name,
        "dataset": dataset_name,
        "window_size": window_size,
        "train_ratio": train_ratio,
        "input_dim": input_dim,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "created": ts,
        "epochs_trained": len(history.get("train_loss", [])),
        "metrics": metrics,
    }
    with open(MODELS_DIR / f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save as the "latest" shortcut (overwrites previous)
    latest_prefix = f"{model_name}_{dataset_name}"
    torch.save(model.state_dict(), MODELS_DIR / f"{latest_prefix}_weights.pt")
    with open(MODELS_DIR / f"{latest_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / f"{latest_prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)


def load_model(model_name, dataset_name, input_dim, **model_kwargs):
    """Load a saved model."""
    prefix = f"{model_name}_{dataset_name}"
    weights_path = MODELS_DIR / f"{prefix}_weights.pt"

    model = get_model(model_name, input_dim=input_dim, **model_kwargs)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    metrics_path = MODELS_DIR / f"{prefix}_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return model, metrics


def run_full_pipeline(df, model_names=None, dataset_label="genai",
                      target_col="power_total_kw", window_size=24,
                      epochs=100, device="cpu"):
    """
    Run the full training + evaluation pipeline for all models on a dataset.
    Returns a dict of results per model.
    """
    if model_names is None:
        model_names = ["LSTM", "GRU", "Transformer"]

    data = prepare_data(df, target_col=target_col, window_size=window_size)
    results = {}

    for name in model_names:
        print(f"\n{'='*50}")
        print(f"Training {name} on {dataset_label}...")
        print(f"{'='*50}")

        model = get_model(name, input_dim=data["input_dim"])

        history = train_model(
            model,
            data["train_loader"],
            val_loader=data["test_loader"],
            epochs=epochs,
            device=device,
        )

        preds, targets, metrics = evaluate_model(
            model, data["test_loader"], data["target_scaler"], device=device
        )

        save_model(
            model,
            name,
            dataset_label,
            metrics,
            history,
            window_size=window_size,
            train_ratio=0.8,
            input_dim=data["input_dim"],
            feature_names=data["feature_names"],
        )

        results[name] = {
            "model": model,
            "predictions": preds,
            "actuals": targets,
            "metrics": metrics,
            "history": history,
            "timestamps": data["timestamps_test"][:len(preds)] if data["timestamps_test"] is not None else None,
        }

        print(f"\n{name} Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return results
