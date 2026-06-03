"""
Transfer learning helpers: pretrain a model on a source dataset (v2020),
then fine-tune on a target dataset (GenTD26).

Approach: train a deep model on the source dataset's power series. Save
encoder weights. On the target dataset, re-initialize the head, freeze
or warm-start the encoder, and fine-tune. Compare against from-scratch.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import TimeSeriesDataset, prepare_data, evaluate_model
from ..models.architectures import get_model


def _train(model, train_loader, val_loader, epochs, lr, patience, device, freeze_prefix=None):
    model = model.to(device)
    if freeze_prefix:
        for n, p in model.named_parameters():
            if any(n.startswith(pref) for pref in freeze_prefix):
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best = float("inf"); best_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
        history["train_loss"].append(float(np.mean(tl)))
        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vl.append(crit(model(xb.to(device)), yb.to(device)).item())
        v = float(np.mean(vl)) if vl else float("inf")
        history["val_loss"].append(v)
        if v < best:
            best = v; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    # unfreeze again so caller can continue fine-tuning if desired
    for p in model.parameters():
        p.requires_grad = True
    return model, history


def _intersect_features(source_df, target_df, target_col):
    """Find feature columns common to both datasets (by name)."""
    exclude = {"timestamp", "container_id", "worker_name", "machine",
               "start_time", "end_time", "time_offset", target_col}
    src_cols = [c for c in source_df.columns if c not in exclude
                and pd.api.types.is_numeric_dtype(source_df[c])]
    tgt_cols = [c for c in target_df.columns if c not in exclude
                and pd.api.types.is_numeric_dtype(target_df[c])]
    common = [c for c in src_cols if c in tgt_cols]
    return common


def pretrain_then_finetune(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_name: str = "GRU",
    target_col: str = "power_total_kw",
    window_size: int = 24,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 50,
    lr_pretrain: float = 1e-3,
    lr_finetune: float = 5e-4,
    patience: int = 8,
    train_ratio_target: float = 0.8,
    device: str = "cpu",
    freeze_encoder: bool = False,
    model_kwargs=None,
    progress_cb=None,
):
    """Pretrain on source_df, fine-tune on target_df. Also trains a from-scratch
    model on target_df only as a baseline. Returns metrics for both."""
    common = _intersect_features(source_df, target_df, target_col)
    if not common:
        raise ValueError("No overlapping numeric feature columns between source and target.")

    # Use only common features
    src_data = prepare_data(source_df[["timestamp"] + common + [target_col]],
                             target_col=target_col, feature_cols=common,
                             window_size=window_size, train_ratio=0.9)
    tgt_data = prepare_data(target_df[["timestamp"] + common + [target_col]],
                             target_col=target_col, feature_cols=common,
                             window_size=window_size, train_ratio=train_ratio_target)

    kwargs = dict(model_kwargs or {})
    if model_name == "Transformer":
        kwargs.setdefault("d_model", kwargs.pop("hidden_dim", 64))
        kwargs.setdefault("nhead", 4)
        kwargs.setdefault("dim_feedforward", kwargs["d_model"] * 2)

    if progress_cb: progress_cb("Pretraining on source...")
    src_model = get_model(model_name, input_dim=len(common), **kwargs)
    src_model, src_hist = _train(
        src_model, src_data["train_loader"], src_data["test_loader"],
        epochs=pretrain_epochs, lr=lr_pretrain, patience=patience, device=device,
    )

    # Fine-tune on target — start from pretrained weights
    if progress_cb: progress_cb("Fine-tuning on target...")
    ft_model = get_model(model_name, input_dim=len(common), **kwargs)
    ft_model.load_state_dict(src_model.state_dict())
    freeze = ["lstm.", "gru.", "transformer.", "input_proj.", "pos_enc."] if freeze_encoder else None
    ft_model, ft_hist = _train(
        ft_model, tgt_data["train_loader"], tgt_data["test_loader"],
        epochs=finetune_epochs, lr=lr_finetune, patience=patience, device=device,
        freeze_prefix=freeze,
    )
    ft_preds, ft_targets, ft_metrics = evaluate_model(
        ft_model, tgt_data["test_loader"], tgt_data["target_scaler"], device=device)

    # From-scratch baseline on target
    if progress_cb: progress_cb("Training from scratch on target...")
    sc_model = get_model(model_name, input_dim=len(common), **kwargs)
    sc_model, sc_hist = _train(
        sc_model, tgt_data["train_loader"], tgt_data["test_loader"],
        epochs=finetune_epochs, lr=lr_pretrain, patience=patience, device=device,
    )
    sc_preds, sc_targets, sc_metrics = evaluate_model(
        sc_model, tgt_data["test_loader"], tgt_data["target_scaler"], device=device)

    return {
        "common_features": common,
        "pretrain_history": src_hist,
        "finetune": {"metrics": ft_metrics, "history": ft_hist,
                     "preds": ft_preds.tolist(), "targets": ft_targets.tolist()},
        "from_scratch": {"metrics": sc_metrics, "history": sc_hist,
                          "preds": sc_preds.tolist(), "targets": sc_targets.tolist()},
        "improvement": {k: sc_metrics[k] - ft_metrics[k] for k in ft_metrics
                        if k in ("MAE", "RMSE", "MAPE")},
    }
