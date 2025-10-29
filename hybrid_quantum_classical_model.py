"""
Hybrid Quantum-Classical Neural Network (HQNN) for Crop Yield Prediction
Combines classical neural networks with quantum computing using PyTorch + PennyLane
Dynamically adapts to any number of input features
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')


class HybridQuantumClassicalNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network with dynamic feature adaptation
    
    Architecture:
    Input (n_features) → Classical Dense Layer (n_features → n_qubits) 
    → Quantum Layer (n_qubits qubits) → Classical Output Layer (n_qubits → 1) → Prediction
    """
    
    def __init__(self, n_features, n_qubits=4, n_layers=1, learning_rate=0.01):
        super(HybridQuantumClassicalNN, self).__init__()
        
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        self.fc1 = nn.Linear(n_features, n_qubits)
        self.activation1 = nn.Tanh()
        
        # Setup quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            """
            Quantum circuit with angle embedding and strongly entangling layers
            
            Args:
                inputs: Classical activations (n_qubits values)
                weights: Trainable quantum weights
            
            Returns:
                Expectation values of Pauli-Z operators
            """
            # Angle embedding: encode classical values as rotation angles
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Strongly entangling layers with trainable weights
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Measure expectation values of Pauli-Z operators
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Classical output layer: n_qubits quantum outputs → 1 prediction
        self.fc2 = nn.Linear(n_qubits, 1)
        
        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.training_losses = []
        self.validation_losses = []
    
    def forward(self, x):
        """
        Forward pass through hybrid network
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
        
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Classical layer 1: input → n_qubits neurons
        x = self.fc1(x)
        x = self.activation1(x)
        
        x = self.q_layer(x)
        
        # Classical output layer: n_qubits quantum outputs → 1 prediction
        x = self.fc2(x)
        
        return x
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16, verbose=True):
        """
        Train the hybrid model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Print training progress
        
        Returns:
            Training history
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                predictions = self(batch_X)
                loss = self.loss_fn(predictions, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average training loss
            avg_train_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_train_loss)
            
            # Validation loss
            self.eval()
            with torch.no_grad():
                val_predictions = self(X_val_tensor)
                val_loss = self.loss_fn(val_predictions, y_val_tensor)
                self.validation_losses.append(val_loss.item())
            self.train()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
        
        return {
            'train_losses': self.training_losses,
            'val_losses': self.validation_losses
        }
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        self.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions = self(X_tensor)
        
        return predictions.detach().cpu().numpy().flatten()


class HybridModelTrainer:
    """Trainer for Hybrid Quantum-Classical Neural Network"""
    
    def __init__(self, n_qubits=4, n_layers=1, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.n_features = None
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=16, validation_split=0.2):
        """
        Train hybrid model and evaluate on test set
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of training data for validation
        
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        
        X_train_np = np.array(X_train) if hasattr(X_train, 'values') else X_train
        y_train_np = np.array(y_train) if hasattr(y_train, 'values') else y_train
        X_test_np = np.array(X_test) if hasattr(X_test, 'values') else X_test
        y_test_np = np.array(y_test) if hasattr(y_test, 'values') else y_test
        
        # Store number of features
        self.n_features = X_train_np.shape[1]
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train_np)
        X_test_scaled = self.scaler_X.transform(X_test_np)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train_np.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test_np.reshape(-1, 1)).flatten()
        
        # Split validation set
        n_val = int(len(X_train_scaled) * validation_split)
        X_val_scaled = X_train_scaled[-n_val:]
        y_val_scaled = y_train_scaled[-n_val:]
        X_train_scaled = X_train_scaled[:-n_val]
        y_train_scaled = y_train_scaled[:-n_val]
        
        self.model = HybridQuantumClassicalNN(
            n_features=self.n_features,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            learning_rate=self.learning_rate
        )
        
        history = self.model.train_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        train_time = time.time() - start_time
        
        # Calculate metrics
        r2 = r2_score(y_test_np, y_pred)
        mae = mean_absolute_error(y_test_np, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
        
        return {
            'model': self.model,
            'predictions': y_pred,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'train_time': train_time,
            'history': history,
            'y_test': y_test_np,
            'X_test_scaled': X_test_scaled
        }
    
    def predict(self, X):
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_np = np.array(X) if hasattr(X, 'values') else X
        X_scaled = self.scaler_X.transform(X_np)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred


# Visualization functions
def visualize_quantum_outputs(model, sample_data):
    """
    Plot distribution of quantum layer outputs
    
    Args:
        model: Trained HybridQuantumClassicalNN model
        sample_data: Sample data to pass through quantum layer
    """
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        x = torch.FloatTensor(sample_data)
        x = torch.tanh(model.fc1(x))
        q_out = model.q_layer(x)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(q_out.flatten().detach().cpu().numpy(), bins=20, color='skyblue', edgecolor='black', linewidth=1.5)
    ax.set_title("Distribution of Quantum Layer Outputs", fontsize=14, fontweight='bold')
    ax.set_xlabel("Expectation value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def visualize_quantum_circuit_structure(n_qubits=4, n_layers=1):
    """
    Display information about the quantum circuit structure
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of entangling layers
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    circuit_info = f"""
    HYBRID QUANTUM-CLASSICAL NEURAL NETWORK ARCHITECTURE
    
    ┌─────────────────────────────────────────────────────┐
    │  Classical Input Layer                              │
    │  Input Features → Dense Layer (n_features → {n_qubits})  │
    │  Activation: Tanh                                   │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  Quantum Layer                                      │
    │  • Angle Embedding: Encode {n_qubits} classical values  │
    │  • Strongly Entangling Layers: {n_layers} layer(s)      │
    │  • Measurement: Pauli-Z expectation values          │
    │  • Output: {n_qubits} quantum measurements              │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │  Classical Output Layer                             │
    │  {n_qubits} Quantum Outputs → Dense Layer → 1 Prediction │
    └─────────────────────────────────────────────────────┘
    """
    
    ax.text(0.5, 0.5, circuit_info, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig
