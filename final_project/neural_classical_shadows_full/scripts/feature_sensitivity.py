
import argparse, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_shadows.data import generate_dataset
from neural_shadows.train import train_model
from neural_shadows.utils import all_pauli_strings

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

    os.makedirs(args.outdir, exist_ok=True)
    X, Y, _ = generate_dataset(args.n_qubits, args.n_samples, args.shots, args.seed, mix=(0.34,0.33,0.33))
    model, art = train_model(X, Y, epochs=args.epochs, batch_size=128, lr=1e-3, width=192, depth=2, dropout=0.0)
    mean, std = art["scaler_mean"], art["scaler_std"]
    Xn = (X - mean) / std

    model.eval()
    Xtensor = torch.tensor(Xn, dtype=torch.float32, requires_grad=True)
    out = model(Xtensor)
    grads = []
    for j in range(out.shape[1]):
        model.zero_grad()
        out[:,j].sum().backward(retain_graph=True)
        grads.append(Xtensor.grad.detach().abs().mean(dim=0).numpy())
        Xtensor.grad.zero_()
    sens = np.sqrt(np.sum(np.stack(grads, axis=0)**2, axis=0))
    idx = np.argsort(-sens)[:args.topk]
    values = sens[idx]
    paulis = all_pauli_strings(args.n_qubits)[1:]
    labels = [paulis[i] for i in idx]

    plt.figure(figsize=(10,4))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=60, ha="right")
    plt.ylabel("sensitivity"); plt.title(f"Top-{args.topk} feature sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"top{args.topk}_sensitivity.png"), bbox_inches="tight", dpi=160)
