
import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_shadows.data import generate_dataset
from neural_shadows.train import train_model
from neural_shadows.baseline import estimate_properties_from_r

OUTDIR = "./outputs/demo"; os.makedirs(OUTDIR, exist_ok=True)

n_qubits, n_samples, shots, seed = 3, 600, 100, 123
X, Y, meta = generate_dataset(n_qubits, n_samples, shots, seed, mix=(0.34,0.33,0.33))
model, art = train_model(X, Y, epochs=25, batch_size=128, lr=1e-3, width=192, depth=2, dropout=0.0)

te_idx = art["split"]["test_idx"]
Xte, Yte = X[te_idx], Y[te_idx]
mean, std = art["scaler_mean"], art["scaler_std"]
Xte_n = (Xte - mean) / std

model.eval()
with torch.no_grad():
    pred = model(torch.tensor(Xte_n, dtype=torch.float32)).cpu().numpy()

mse_model = ((pred - Yte)**2).mean(axis=0).tolist()
mae_model = (np.abs(pred - Yte)).mean(axis=0).tolist()

R_full = np.concatenate([np.ones((Xte.shape[0],1), dtype=float), Xte], axis=1)
preds_bl = []
for i in range(R_full.shape[0]):
    pr, ent, ghz = estimate_properties_from_r(n_qubits, R_full[i])
    preds_bl.append([pr, ent, ghz])
preds_bl = np.array(preds_bl, dtype=float)
mse_bl = ((preds_bl - Yte)**2).mean(axis=0).tolist()
mae_bl = (np.abs(preds_bl - Yte)).mean(axis=0).tolist()

plt.figure(); plt.plot(art["hist"]["train_mse"], label="train MSE"); plt.plot(art["hist"]["test_mse"], label="test MSE")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Learning curves (tiny demo)")
plt.savefig(os.path.join(OUTDIR, "learning_curves.png"), bbox_inches="tight", dpi=150)

targets = ["purity", "entropy", "ghz_fidelity"]
for j, name in enumerate(targets):
    plt.figure()
    plt.scatter(Yte[:,j], pred[:,j], s=12)
    lo = float(min(Yte[:,j].min(), pred[:,j].min())); hi = float(max(Yte[:,j].max(), pred[:,j].max()))
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(f"True {name}"); plt.ylabel(f"Predicted {name}"); plt.title(f"Pred vs True — {name}")
    plt.savefig(os.path.join(OUTDIR, f"scatter_{name}.png"), bbox_inches="tight", dpi=150)

out = {"n_qubits": n_qubits, "shots": shots, "n_samples": n_samples, "targets": targets,
       "model": {"mse": mse_model, "mae": mae_model}, "baseline": {"mse": mse_bl, "mae": mae_bl}}
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f: json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)); print("[OK] Wrote outputs under", OUTDIR)
