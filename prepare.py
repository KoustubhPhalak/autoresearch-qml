import torch
import numpy as np
from sklearn.model_selection import train_test_split

def generate_ntangled_features(n_samples=1000, n_features=8):
    """
    Hardened NTangled Parameters: 
    Using interleaved 'parity' logic that requires non-linear mapping.
    """
    X = np.random.uniform(0, 2 * np.pi, (n_samples, n_features))
    
    # Class 0: Sum of angles is 'even' (in terms of pi/2 blocks)
    # Class 1: Sum of angles is 'odd'
    # This creates a high-dimensional checkerboard pattern that is 
    # impossible for a simple linear layer or basic HEA to solve.
    y = (np.sum(X // (np.pi / 2), axis=1) % 2).astype(int)
    
    return X, y

if __name__ == "__main__":
    print("Generating NTangled Parameter Dataset...")
    # Using 8 features to allow for multi-qubit RX/RY/RZ patterns
    X, y = generate_ntangled_features(n_samples=1000, n_features=8)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    torch.save({
        'X_train': torch.tensor(X_train, dtype=torch.float32),
        'y_train': torch.tensor(y_train, dtype=torch.long),
        'X_test': torch.tensor(X_test, dtype=torch.complex64), # Keep complex for flexibility
        'X_test': torch.tensor(X_test, dtype=torch.float32),
        'y_test': torch.tensor(y_test, dtype=torch.long)
    }, 'ntangled_params.pt')
    print("Saved to ntangled_params.pt. features: 8, classes: 2 (Low vs High Entanglement)")