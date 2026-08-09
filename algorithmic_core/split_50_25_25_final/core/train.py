import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=150,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    clip_norm=1.0,
):
    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    history["epochs_trained"] = epoch + 1
    return model, history


def evaluate_model(model, test_loader, tgt_scaler, residual_mode=True, yp_test_orig=None, y_test_orig=None):
    """
    Evaluate model on test set. Reconstructs level predictions from residuals.
    Returns metrics in original scale.

    In residual mode:
      - Model outputs scaled deltas
      - Inverse transform → actual delta
      - Level = y_prev + delta
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds_scaled.append(pred.cpu().numpy())
            all_targets_scaled.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds_scaled)
    targets_scaled = np.concatenate(all_targets_scaled)

    if residual_mode and yp_test_orig is not None:
        # Inverse transform scaled residuals → actual deltas
        pred_deltas = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        target_deltas = tgt_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

        # Reconstruct levels
        preds_orig = yp_test_orig + pred_deltas
        targets_orig = yp_test_orig + target_deltas  # = y_test_orig
    else:
        # Direct level prediction
        preds_orig = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        targets_orig = tgt_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    return compute_metrics(targets_orig, preds_orig)


def compute_metrics(actuals, preds):
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    # MAPE with zero-masking
    mask = np.abs(actuals) > 1e-6
    mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100 if mask.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def persistence_baseline(yp_test_orig, y_test_orig):
    """
    Persistence baseline: predict y[t] = y[t-1].
    Works directly in original scale (no inverse transform needed).
    """
    preds = yp_test_orig  # predict last observed value
    targets = y_test_orig  # actual next value
    return compute_metrics(targets, preds)


def linear_baseline(data_dict):
    """Ridge regression baseline on flattened window features."""
    X_train, y_train = [], []
    for X_batch, y_batch in data_dict["train_loader"]:
        X_train.append(X_batch.numpy())
        y_train.append(y_batch.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    X_test, y_test_scaled = [], []
    for X_batch, y_batch in data_dict["test_loader"]:
        X_test.append(X_batch.numpy())
        y_test_scaled.append(y_batch.numpy())
    X_test = np.concatenate(X_test)
    y_test_scaled = np.concatenate(y_test_scaled)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = Ridge(alpha=1.0)
    model.fit(X_train_flat, y_train)
    preds_scaled = model.predict(X_test_flat)

    tgt_scaler = data_dict["tgt_scaler"]
    yp_test_orig = data_dict["yp_test_orig"]

    if data_dict["residual_mode"]:
        pred_deltas = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        preds_orig = yp_test_orig + pred_deltas
    else:
        preds_orig = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    return compute_metrics(data_dict["y_test_orig"], preds_orig)
