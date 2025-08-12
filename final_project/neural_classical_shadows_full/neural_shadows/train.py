
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .model import MLPRegressor

def train_model(X: np.ndarray, Y: np.ndarray, epochs: int = 60, batch_size: int = 128,
                lr: float = 1e-3, weight_decay: float = 0.0, width: int = 256, depth: int = 3,
                dropout: float = 0.0, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_train = int(0.8 * len(X))
    idx = np.random.permutation(len(X))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    mean = Xtr.mean(axis=0, keepdims=True); std = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_n = (Xtr - mean) / std; Xte_n = (Xte - mean) / std

    model = MLPRegressor(in_dim=X.shape[1], out_dim=Y.shape[1], width=width, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.tensor(Xtr_n, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    hist = {"train_mse": [], "test_mse": []}
    for _ in range(epochs):
        model.train(); total = 0.0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        total /= len(ds)
        model.eval()
        with torch.no_grad():
            te_pred = model(torch.tensor(Xte_n, dtype=torch.float32, device=device))
            te_loss = loss_fn(te_pred, torch.tensor(Yte, dtype=torch.float32, device=device)).item()
        hist["train_mse"].append(total); hist["test_mse"].append(te_loss)

    return model, {"model_state": model.state_dict(), "scaler_mean": mean, "scaler_std": std,
                   "hist": hist, "split": {"train_idx": tr_idx, "test_idx": te_idx}, "device": device}
