
import argparse, os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from neural_shadows.data import generate_dataset
from neural_shadows.train import train_model

def run(n_qubits=3, n_samples=800, shots=200, seed=0, epochs=35, outdir="./outputs/prediction_panels"):
    os.makedirs(outdir, exist_ok=True)
    targets = ["purity", "entropy", "ghz_fidelity"]
    X, Y, meta = generate_dataset(n, n_samples, shots, seed, mix=(0.34,0.33,0.33))

    model, art = train_model(X, Y, epochs=epochs, batch_size=128, lr=1e-3, width=192, depth=2, dropout=0.0)
    te_idx = art["split"]["test_idx"]
    Xte = X[te_idx]; Yte = Y[te_idx]
    mean = art["scaler_mean"]; std = art["scaler_std"]
    Xte_n = (Xte - mean) / std

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte_n, dtype=torch.float32)).cpu().numpy()

    # 1) Scatter parity per target
    for j, name in enumerate(targets):
        plt.figure()
        plt.scatter(Yte[:,j], pred[:,j], s=12)
        lo = float(min(Yte[:,j].min(), pred[:,j].min()))
        hi = float(max(Yte[:,j].max(), pred[:,j].max()))
        plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Pred vs True — {name}")
        plt.savefig(os.path.join(outdir, f"scatter_{name}.png"), bbox_inches="tight", dpi=160)

    # 2) Residual histograms per target
    resid = pred - Yte
    for j, name in enumerate(targets):
        plt.figure()
        plt.hist(resid[:,j], bins=40)
        plt.xlabel(f"Residual ({name})")
        plt.ylabel("count")
        plt.title(f"Residual distribution — {name}")
        plt.savefig(os.path.join(outdir, f"residual_hist_{name}.png"), bbox_inches="tight", dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--outdir", type=str, default="./outputs/prediction_panels")
    args = ap.parse_args()
    n = args.n_qubits
    run(args.n_qubits, args.n_samples, args.shots, args.seed, args.epochs, args.outdir)
