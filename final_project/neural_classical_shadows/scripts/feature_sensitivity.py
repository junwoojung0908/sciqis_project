
import argparse, os, json, numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_shadows.data import generate_dataset
from neural_shadows.train import train_model
from neural_shadows.utils import all_pauli_strings

def topk_sensitivity(n_qubits=3, n_samples=800, shots=200, seed=0, epochs=25, k=20, outdir="./outputs/feature_sensitivity"):
    os.makedirs(outdir, exist_ok=True)
    X, Y, meta = generate_dataset(n_qubits, n_samples, shots, seed, mix=(0.34,0.33,0.33))
    model, art = train_model(X, Y, epochs=epochs, batch_size=128, lr=1e-3, width=192, depth=2, dropout=0.0)
    mean = art["scaler_mean"]; std = art["scaler_std"]
    Xn = (X - mean) / std
    import torch.nn as nn
    model.eval()
    Xtensor = torch.tensor(Xn, dtype=torch.float32, requires_grad=True)
    out = model(Xtensor)
    # Aggregate sensitivity across outputs (L2 norm of grad wrt inputs)
    grads = []
    for j in range(out.shape[1]):
        model.zero_grad()
        out[:,j].sum().backward(retain_graph=True)
        grads.append(Xtensor.grad.detach().abs().mean(dim=0).numpy())
        Xtensor.grad.zero_()
    sens = np.sqrt(np.sum(np.stack(grads, axis=0)**2, axis=0))
    idx = np.argsort(-sens)[:k]
    values = sens[idx]
    # Map feature indices to Pauli strings (skip identity at 0)
    paulis = all_pauli_strings(n_qubits)[1:]  # drop identity
    labels = [paulis[i] for i in idx]
    # Bar plot
    plt.figure(figsize=(10,4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=60, ha="right")
    plt.ylabel("sensitivity")
    plt.title(f"Top-{k} feature sensitivity (across targets)")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"top{k}_sensitivity.png"), bbox_inches="tight", dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="./outputs/feature_sensitivity")
    args = ap.parse_args()
    topk_sensitivity(args.n_qubits, args.n_samples, args.shots, args.seed, args.epochs, args.topk, args.outdir)
