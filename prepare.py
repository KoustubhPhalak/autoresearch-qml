import torch
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split

# --- NTangled Configuration (Based on 2109.03400) ---
n_qubits = 4
n_samples = 1000
dev = qml.device("default.qubit", wires=n_qubits)

def get_concentratable_entanglement(state):
    """
    Calculates CE: The average linear entropy across all subsystems.
    Using the 2026 PennyLane 'qml.math' interface.
    """
    ce_val = 0
    for i in range(n_qubits):
        # ── NEW 2026 METHOD ──
        # Computes the reduced density matrix for qubit i from the state vector
        rho = qml.math.reduce_statevector(state, indices=[i])
        
        # Linear entropy: 1 - tr(rho^2)
        # qml.math is framework-agnostic (works with NumPy, Torch, etc.)
        tr_rho2 = qml.math.real(qml.math.trace(qml.math.matmul(rho, rho)))
        ce_val += (1 - tr_rho2)
        
    return ce_val / n_qubits

@qml.qnode(dev)
def generate_state(weights):
    """
    Strongly Entangling Ansatz (SEA) from the paper.
    Used to generate states with high entanglement complexity.
    """
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.state()

def generate_ntangled_dataset():
    states = []
    labels = []
    
    print("Generating NTangled states...")
    while len(states) < n_samples:
        # Random weights for the SEA (3 parameters per qubit per layer)
        # Low layers = Low CE, High layers = High CE
        depth = np.random.choice([1, 4]) 
        weights = np.random.uniform(0, 2*np.pi, (depth, n_qubits, 3))
        
        state_vec = generate_state(weights)
        ce = get_concentratable_entanglement(state_vec)
        
        # Binary Classification Task: Shallow vs. Deep Entanglement
        # This is the "Hard" version of the task from the paper.
        if depth == 1:
            states.append(state_vec)
            labels.append(0) # Class 0: Low complexity
        else:
            states.append(state_vec)
            labels.append(1) # Class 1: High complexity
            
    return np.array(states), np.array(labels)

if __name__ == "__main__":
    X, y = generate_ntangled_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Save as complex64 state vectors
    torch.save({
        'X_train': torch.tensor(X_train, dtype=torch.complex64),
        'y_train': torch.tensor(y_train, dtype=torch.long),
        'X_test': torch.tensor(X_test, dtype=torch.complex64),
        'y_test': torch.tensor(y_test, dtype=torch.long)
    }, 'ntangled_states.pt')
    print(f"Saved {n_samples} NTangled state vectors (4-qubits).")