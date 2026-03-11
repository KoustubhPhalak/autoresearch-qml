import torch
import numpy as np
from sklearn.model_selection import train_test_split

def generate_ntangled_features(n_samples=600, n_features=8):
    """
    Simulates the weights/parameters provided in the Schatzki repo.
    Class 0: Parameters leading to low 'Concentratable Entanglement' (CE).
    Class 1: Parameters leading to high 'Concentratable Entanglement' (CE).
    """
    # Class 0: Narrow distribution of angles (less 'scrambling')
    class_0 = np.random.uniform(0, np.pi/4, (n_samples // 2, n_features))
    
    # Class 1: Wide distribution of angles (maximal 'scrambling')
    class_1 = np.random.uniform(0, 2*np.pi, (n_samples // 2, n_features))
    
    X = np.vstack([class_0, class_1])
    y = np.array([0]*(n_samples // 2) + [1]*(n_samples // 2))
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

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