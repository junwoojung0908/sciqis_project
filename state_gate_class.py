import numpy as np

class State:
    def __init__(self, vector):
        self.vector = vector

    def apply(self, gate):
        return State(gate.matrix@self.vector)

    def norm(self):
        return np.linalg.norm(self.vector)
    
    def normalise(self):
        return State(self.vector/self.norm())
    
    def __str__(self):
        return str(self.vector)
    
    def __pow__(self, other):
        if isinstance(self, State):
            return State(np.kron(self.vector,other.vector))
    
    def full_measurement(state):
        # Calculate probabilities for each basis state
        probs = np.abs(state)**2

        # Randomly select a basis state based on probabilities
        outcome_index = np.random.choice(len(state), p=probs)

        # Collapse the state to the selected basis state
        collapsed_state = np.zeros_like(state)
        collapsed_state[outcome_index] = 1

        return outcome_index, collapsed_state
    
    def subsystem_measurement(self, qubit_index, num_qubits):
        """
        Perform measurement on a specific qubit in a multi-qubit system and remove the measured dimension.
        """
        # Reshape the state vector into a tensor product form
        reshaped_state = self.vector.reshape([2] * num_qubits)

        # Calculate probabilities for the specified qubit
        probs = np.zeros(2)
        for i in range(2):
            mask = [slice(None)] * num_qubits
            mask[qubit_index] = i
            probs[i] = np.sum(np.abs(reshaped_state[tuple(mask)])**2)

        # Randomly select a measurement outcome
        outcome = np.random.choice([0, 1], p=probs)

        # Collapse the state based on the measurement outcome
        mask = [slice(None)] * num_qubits
        mask[qubit_index] = outcome
        collapsed_state = reshaped_state[tuple(mask)]

        # Remove the measured dimension
        final_state = State(collapsed_state.flatten())
        new_state_vector = final_state.normalise()

        return outcome, new_state_vector
    
    def __str__(self):
        return str(self.vector.round(4))
    
class Gate:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def __matmul__(self, other):
        if isinstance(other, Gate):
            return Gate(other.matrix@self.matrix)
        elif isinstance(other, State):
            return other.apply(self)
        else:
            raise TypeError("Unsupported type for multiplication with Gate")
        
    def __mul__(self, other):
        return Gate(self.matrix * other)
        
    def __add__(self, other):
        if isinstance(other, Gate):
            return Gate(self.matrix + other.matrix)
        else:
            raise TypeError("Unsupported type for addition with Gate")
    
    def __pow__(self, other):
        return Gate(np.kron(self.matrix, other.matrix))
    
    def __xor__(self, int):
        a = self
        for _ in range(int):
            a = a ** self
        return a
    
    def __str__(self):
        return str(self.matrix)

I = Gate(np.array([[1, 0], [0, 1]]))

print(I*2)
