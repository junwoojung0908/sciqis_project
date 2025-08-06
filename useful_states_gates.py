import numpy as np
from state_gate_class import State, Gate

zero_state = State(np.array([1, 0]))  # |0>
one_state  = State(np.array([0, 1]))  # |1>

phi_plus  = State(np.array([1, 0, 0, 1]) / np.sqrt(2))  # |00> + |11>
phi_minus = State(np.array([1, 0, 0, -1]) / np.sqrt(2)) # |00> - |11>
psi_plus  = State(np.array([0, 1, 1, 0]) / np.sqrt(2))  # |01> + |10>
psi_minus = State(np.array([0, 1, -1, 0]) / np.sqrt(2)) # |01> - |10>


I = Gate(np.array([[1, 0], [0, 1]]))
X = Gate(np.array([[0, 1], [1, 0]]))    # Pauli-X gate
Y = Gate(np.array([[0, -1j], [1j, 0]])) # Pauli-Y gate
Z = Gate(np.array([[1, 0], [0, -1]]))   # Pauli-Z gate
H = Gate((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))    # Hadamard gate
T = Gate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])) # T gate
CNOT = Gate(np.array([[1, 0, 0, 0], 
                 [0, 1, 0, 0], 
                 [0, 0, 0, 1], 
                 [0, 0, 1, 0]]))  # CNOT gate
proj_zero  = Gate(np.array([[1, 0], [0, 0]]))  # Projector onto |0>
proj_one  = Gate(np.array([[0, 0], [0, 1]]))  # Projector onto |1>