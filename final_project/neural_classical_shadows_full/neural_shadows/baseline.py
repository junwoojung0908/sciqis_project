
from typing import Tuple, List
import numpy as np
from qutip import Qobj
from .utils import pauli_tensor_from_string, all_pauli_strings, ghz_fidelity, purity, von_neumann_entropy

def reconstruct_density_from_paulis(n: int, r: np.ndarray, paulis: List[str]) -> Qobj:
    d = 2**n
    op = 0
    for coeff, P in zip(r, paulis):
        op = op + float(coeff) * pauli_tensor_from_string(P)
    rho = (1.0 / d) * op
    return (rho + rho.dag()) / 2.0

def estimate_properties_from_r(n: int, r_full: np.ndarray) -> Tuple[float,float,float]:
    paulis = all_pauli_strings(n)
    rho = reconstruct_density_from_paulis(n, r_full, paulis)
    return (purity(rho), von_neumann_entropy(rho), ghz_fidelity(rho, n))
