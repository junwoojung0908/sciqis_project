
from typing import Tuple, List
import numpy as np
from qutip import Qobj, basis, qeye, sigmax, sigmay, sigmaz, tensor, ket2dm, rand_ket
from .utils import ghz_fidelity, von_neumann_entropy, purity, pauli_index_map

def eigenkets_of(pauli: str):
    if pauli == 'Z':
        return [basis(2,0), basis(2,1)]
    elif pauli == 'X':
        plus  = (basis(2,0) + basis(2,1)).unit()
        minus = (basis(2,0) - basis(2,1)).unit()
        return [plus, minus]
    elif pauli == 'Y':
        plus  = (basis(2,0) + 1j*basis(2,1)).unit()
        minus = (basis(2,0) - 1j*basis(2,1)).unit()
        return [plus, minus]
    else:
        raise ValueError("pauli must be X/Y/Z")

def outcome_projector(basis_letters: str, outcomes_pm: np.ndarray) -> Qobj:
    kets = []
    for letter, ev in zip(basis_letters, outcomes_pm):
        k0, k1 = eigenkets_of(letter)
        kets.append(k0 if ev == +1 else k1)
    ket = tensor(kets)
    return ket * ket.dag()

def sample_one_shot(rho: Qobj, basis_letters: str, rng: np.random.Generator) -> np.ndarray:
    n = len(basis_letters)
    outcomes = []
    probs = []
    for bits in range(1<<n):
        pm = np.array([+1 if ((bits >> j) & 1) == 0 else -1 for j in range(n)], dtype=int)
        P = outcome_projector(basis_letters, pm)
        p = float((P * rho).tr().real)
        outcomes.append(pm)
        probs.append(p)
    probs = np.array(probs, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum()
    idx = rng.choice(len(outcomes), p=probs)
    return outcomes[idx]

def estimate_pauli_expectations_from_shots(n: int, bases: List[str], outcomes: List[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    paulis, _ = pauli_index_map(n)
    sums = np.zeros(len(paulis), dtype=float)
    cnts = np.zeros(len(paulis), dtype=int)
    for B, e in zip(bases, outcomes):
        for i, P in enumerate(paulis):
            use = True
            prod = 1
            for k, (p_letter, b_letter) in enumerate(zip(P, B)):
                if p_letter == 'I':
                    continue
                if p_letter != b_letter:
                    use = False; break
                prod *= int(e[k])
            if use:
                sums[i] += prod
                cnts[i] += 1
    idx_I = 0
    sums[idx_I] = cnts[idx_I]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    r[idx_I] = 1.0
    return r, paulis

def random_pure_state(n: int, rng: np.random.Generator) -> Qobj:
    # QuTiP variants: some don't accept dims=; set dims after creation
    ket = rand_ket(2**n, density=0.75, seed=int(rng.integers(1, 2**31-1)))
    ket.dims = [[2]*n, [1]*n]  # ensure compatibility with tensor-built ops
    return ket2dm(ket)

def depolarized_pure_state(n: int, rng: np.random.Generator, lam_max: float = 0.6) -> Qobj:
    rho_p = random_pure_state(n, rng)
    lam = float(rng.uniform(0.0, lam_max))
    I = tensor([qeye(2) for _ in range(n)])
    d = 2**n
    return (1.0 - lam) * rho_p + lam * (I / d)

def random_thermal_state(n: int, rng: np.random.Generator) -> Qobj:
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H = 0
    for i in range(n):
        H += float(rng.normal(0,1)) * tensor([sx if k==i else qeye(2) for k in range(n)])
        H += float(rng.normal(0,1)) * tensor([sy if k==i else qeye(2) for k in range(n)])
        H += float(rng.normal(0,1)) * tensor([sz if k==i else qeye(2) for k in range(n)])
    for i in range(n-1):
        zz = tensor([sz if k in (i,i+1) else qeye(2) for k in range(n)])
        H += float(rng.normal(0,1)) * zz
    beta = float(rng.uniform(0.2, 2.0))
    rho = (-beta * H).expm()
    return rho / rho.tr()

def draw_state(n: int, rng: np.random.Generator, family: str) -> Qobj:
    f = family.lower()
    if f == 'pure': return random_pure_state(n, rng)
    if f == 'depolarized': return depolarized_pure_state(n, rng)
    if f == 'thermal': return random_thermal_state(n, rng)
    raise ValueError("family must be pure/depolarized/thermal")

def simulate_sample(n: int, shots: int, rng: np.random.Generator, state_family: str):
    rho = draw_state(n, rng, state_family)
    bases, outcomes = [], []
    letters = np.array(['X','Y','Z'])
    for _ in range(shots):
        B = ''.join(rng.choice(letters, size=n))
        e = sample_one_shot(rho, B, rng)
        bases.append(B); outcomes.append(e)
    r, _ = estimate_pauli_expectations_from_shots(n, bases, outcomes)
    y = np.array([purity(rho), von_neumann_entropy(rho), ghz_fidelity(rho, n)], dtype=float)
    X = r[1:].astype(float)
    return X, y

def generate_dataset(n: int, n_samples: int, shots: int, seed: int, mix=(1/3,1/3,1/3)):
    rng = np.random.default_rng(seed)
    fams = ['pure','depolarized','thermal']
    weights = np.array(mix, dtype=float); weights /= weights.sum()
    X, Y, labels = [], [], []
    for _ in range(n_samples):
        family = rng.choice(fams, p=weights)
        Xi, yi = simulate_sample(n, shots, rng, family)
        X.append(Xi); Y.append(yi); labels.append(family)
    X = np.stack(X, axis=0); Y = np.stack(Y, axis=0)
    meta = {"n_qubits": n, "shots": shots, "features": int(4**n - 1),
            "targets": ["purity","entropy","ghz_fidelity"], "families": labels}
    return X, Y, meta
