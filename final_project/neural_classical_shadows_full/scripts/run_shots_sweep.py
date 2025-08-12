
import argparse, os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_shadows.data import generate_dataset
from neural_shadows.train import train_model
from neural_shadows.baseline import estimate_properties_from_r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=1200)
    ap.add_argument("--shots", type=str, default="50,100,200,400")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--outdir", type=str, default="./outputs/sweep")
    args = ap.parse_args()

    shots_list = [int(x) for x in args.shots.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    targets = ["purity", "entropy", "ghz_fidelity"]
    results = {"shots": shots_list, "targets": targets, "model_mse": [], "baseline_mse": []}

    for s in shots_list:
        X, Y, _ = generate_dataset(args.n_qubits, args.n_samples, s, args.seed, mix=(0.34,0.33,0.33))
        model, art = train_model(X, Y, epochs=args.epochs, batch_size=args.batch, lr=args.lr, width=args.width, depth=args.depth, dropout=args.dropout)
        te_idx = art["split"]["test_idx"]
        Xte, Yte = X[te_idx], Y[te_idx]
        mean, std = art["scaler_mean"], art["scaler_std"]
        Xte_n = (Xte - mean) / std
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte_n, dtype=torch.float32)).cpu().numpy()
        mse_model = ((pred - Yte)**2).mean(axis=0)
        R_full = np.concatenate([np.ones((Xte.shape[0],1), dtype=float), Xte], axis=1)
        preds_bl = []
        for i in range(R_full.shape[0]):
            pr, ent, ghz = estimate_properties_from_r(args.n_qubits, R_full[i])
            preds_bl.append([pr, ent, ghz])
        preds_bl = np.array(preds_bl, dtype=float)
        mse_bl = ((preds_bl - Yte)**2).mean(axis=0)
        results["model_mse"].append(mse_model.tolist())
        results["baseline_mse"].append(mse_bl.tolist())

    with open(os.path.join(args.outdir, "sweep_results.json"), "w") as f: json.dump(results, f, indent=2)
    shots = np.array(shots_list, dtype=float)
    for j, name in enumerate(targets):
        plt.figure()
        plt.plot(shots, np.array(results["model_mse"])[:,j], marker="o", label="Neural")
        plt.plot(shots, np.array(results["baseline_mse"])[:,j], marker="o", label="Pauli-LS")
        plt.xlabel("shots per sample"); plt.ylabel("MSE"); plt.title(f"Sample efficiency — {name}")
        plt.legend(); plt.savefig(os.path.join(args.outdir, f"mse_vs_shots_{name}.png"), bbox_inches="tight", dpi=160)
