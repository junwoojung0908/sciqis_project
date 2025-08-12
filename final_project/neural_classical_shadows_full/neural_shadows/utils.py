
import numpy as np
from typing import List
from qutip import tensor, basis, qeye, sigmax, sigmay, sigmaz, Qobj, ket2dm, expect

def ghz_state(n: int) -> Qobj:
    zero = tensor([basis(2,0) for _ in range(n)])
    one  = tensor([basis(2,1) for _ in range(n)])
    return (zero + one).unit()

def ghz_fidelity(rho: Qobj, n: int) -> float:
    g = ghz_state(n)
    val = float(np.real(expect(ket2dm(g), rho)))  # Tr[|GHZ><GHZ| rho]
    return float(np.clip(val, 0.0, 1.0))

def von_neumann_entropy(rho: Qobj, eps: float = 1e-12) -> float:
    evals = np.linalg.eigvalsh((rho + rho.dag()).full() / 2.0)
    evals = np.clip(evals.real, 0.0, 1.0)
    evals = evals[evals > eps]
    return float(-np.sum(evals * np.log2(evals)))

def purity(rho: Qobj) -> float:
    return float((rho * rho).tr().real)

def pauli_tensor_from_string(s: str) -> Qobj:
    mapping = {'I': qeye(2), 'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}
    return tensor([mapping[ch] for ch in s])

def all_pauli_strings(n: int) -> list:
    from itertools import product
    return [''.join(p) for p in product('IXYZ', repeat=n)]

def pauli_index_map(n: int):
    strings = all_pauli_strings(n)
    to_idx = {s:i for i,s in enumerate(strings)}
    return strings, to_idx
