import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt
import numpy as np_regular

# ===========================================
# 0. Helper Functions
# ===========================================

def to_numpy(tensor):
    """Safely convert PyTorch tensor (even with gradients) to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (list, tuple)):
        return np_regular.asarray([to_numpy(t) for t in tensor])
    else:
        return np_regular.asarray(tensor)

# ===========================================
# 1. Quantum Circuit Definition
# ===========================================

def create_quantum_circuit(n_qubits, n_layers):
    """Create a quantum circuit with specified number of qubits and layers"""
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        """Quantum circuit with angle embedding and entanglement"""
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        # Return measurements - using stack to ensure proper tensor output
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return qml.math.stack(measurements)
    
    weight_shapes = {"weights": (n_layers, n_qubits)}
    quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    return quantum_layer

# ===========================================
# 2. Advanced Hybrid Model Definition
# ===========================================

class AdvancedHybridNN(nn.Module):
    """Advanced Hybrid Quantum-Classical Neural Network"""
    
    def __init__(self, input_dim, quantum_input_dim=4, n_qubits=4, n_layers=2):
        super(AdvancedHybridNN, self).__init__()
        
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Create a unique quantum layer for this instance
        self.quantum_net = create_quantum_circuit(n_qubits, n_layers)
        self.quantum_input_dim = quantum_input_dim
        self.n_qubits = n_qubits
        
        self.combined = nn.Sequential(
            nn.Linear(32 + n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        """Forward pass through hybrid network"""
        classical_out = self.classical_net(x)
        
        quantum_input = x[:, :self.quantum_input_dim] if x.shape[1] >= self.quantum_input_dim else x
        if quantum_input.shape[1] < self.n_qubits:
            padding = torch.zeros(quantum_input.shape[0], self.n_qubits - quantum_input.shape[1], device=x.device)
            quantum_input = torch.cat([quantum_input, padding], dim=1)
        elif quantum_input.shape[1] > self.n_qubits:
            quantum_input = quantum_input[:, :self.n_qubits]
        
        quantum_out = self.quantum_net(quantum_input)
        
        # The quantum layer should return a torch.Tensor when interface="torch"
        # Ensure it's a tensor and on the correct device
        if not isinstance(quantum_out, torch.Tensor):
            # This should not happen with interface="torch", but handle gracefully
            raise TypeError(f"Quantum layer returned {type(quantum_out)}, expected torch.Tensor. "
                          f"Check quantum circuit definition and interface settings.")
        
        # Ensure it's on the correct device
        if quantum_out.device != x.device:
            quantum_out = quantum_out.to(x.device)
        
        # Ensure correct dimensions: should be [batch_size, n_qubits]
        if quantum_out.dim() == 1:
            # If batch_size == 1, it might be a 1D tensor
            if quantum_out.shape[0] == self.n_qubits:
                quantum_out = quantum_out.unsqueeze(0)
        elif quantum_out.dim() > 2:
            # Flatten extra dimensions
            quantum_out = quantum_out.view(quantum_out.shape[0], -1)
            if quantum_out.shape[1] > self.n_qubits:
                quantum_out = quantum_out[:, :self.n_qubits]
        
        combined_input = torch.cat((classical_out, quantum_out), dim=1)
        return self.combined(combined_input)

# ===========================================
# 3. Data Preprocessing Pipeline
# ===========================================

class DataPreprocessor:
    """Advanced data preprocessing with PCA and feature selection"""
    
    def __init__(self, use_pca=True, pca_variance=0.98, use_feature_selection=False):
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.use_feature_selection = use_feature_selection
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
    
    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        X_array = X.values if hasattr(X, 'values') else np_regular.asarray(X)
        X_scaled = self.scaler.fit_transform(X_array)
        
        y_array = None
        if y is not None:
            y_array = y.values if hasattr(y, 'values') else np_regular.asarray(y)
            # Flatten y if it's 2D
            if y_array.ndim > 1:
                y_array = y_array.flatten()
        
        if self.use_feature_selection and X_scaled.shape[1] > 6 and y_array is not None:
            self.feature_selector = SelectKBest(mutual_info_regression, k=6)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y_array)
        
        if self.use_pca and X_scaled.shape[1] > 4:
            self.pca = PCA(self.pca_variance)
            X_scaled = self.pca.fit_transform(X_scaled)
        
        y_scaled = None
        if y_array is not None:
            y_scaled = self.y_scaler.fit_transform(y_array.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def transform(self, X, y=None):
        """Transform data using fitted preprocessor"""
        X_array = X.values if hasattr(X, 'values') else np_regular.asarray(X)
        X_scaled = self.scaler.transform(X_array)
        
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        y_scaled = None
        if y is not None:
            y_array = y.values if hasattr(y, 'values') else np_regular.asarray(y)
            if y_array.ndim > 1:
                y_array = y_array.flatten()
            y_scaled = self.y_scaler.transform(y_array.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled

# ===========================================
# 4. Training Pipeline with Cross-Validation
# ===========================================

class AdvancedHybridTrainer:
    """Training pipeline for advanced hybrid model"""
    
    def __init__(self, n_qubits=4, n_layers=2, learning_rate=0.0005):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_fold(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train model on a single fold"""
        preprocessor = DataPreprocessor(use_pca=True, use_feature_selection=False)
        X_train_processed, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
        X_val_processed, y_val_scaled = preprocessor.transform(X_val, y_val)
        
        X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        quantum_input_dim = min(X_train_processed.shape[1], self.n_qubits)
        
        model = AdvancedHybridNN(
            input_dim=X_train_processed.shape[1], 
            quantum_input_dim=quantum_input_dim,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers
        ).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = loss_fn(preds, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = loss_fn(val_preds, y_val_tensor)
                val_losses.append(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            y_pred_scaled_tensor = model(X_val_tensor)
            y_pred_scaled = y_pred_scaled_tensor.detach().cpu().numpy().flatten()
        
        y_pred = preprocessor.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_val_original = y_val.values if hasattr(y_val, 'values') else y_val
        y_val_original = np_regular.asarray(y_val_original).flatten()
        
        r2 = r2_score(y_val_original, y_pred)
        rmse = mean_squared_error(y_val_original, y_pred, squared=False)
        mae = mean_absolute_error(y_val_original, y_pred)
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': y_pred
        }
    
    def train_cv(self, X, y, n_splits=5, epochs=100, batch_size=32):
        """Train model with k-fold cross-validation"""
        X_array = X.values if hasattr(X, 'values') else np_regular.asarray(X)
        y_array = y.values if hasattr(y, 'values') else np_regular.asarray(y)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        start_total = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_array)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            fold_result = self.train_fold(X_train, y_train, X_val, y_val, epochs, batch_size)
            results.append(fold_result)
        
        total_time = time.time() - start_total
        
        avg_r2 = np_regular.mean([r['r2'] for r in results])
        avg_rmse = np_regular.mean([r['rmse'] for r in results])
        avg_mae = np_regular.mean([r['mae'] for r in results])
        
        print(f"\n=== Cross-Validation Results ===")
        print(f"Average R²: {avg_r2:.4f}")
        print(f"Average RMSE: {avg_rmse:.2f}")
        print(f"Average MAE: {avg_mae:.2f}")
        print(f"Total Training Time: {total_time:.2f}s")
        
        return {
            'fold_results': results,
            'avg_r2': avg_r2,
            'avg_rmse': avg_rmse,
            'avg_mae': avg_mae,
            'train_time': total_time
        }

# ===========================================
# 5. Visualization Functions
# ===========================================

def visualize_quantum_circuit_structure(n_qubits=4, n_layers=2):
    """Visualize quantum circuit structure"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_height = 1.0
    qubit_spacing = 1.0
    
    for layer in range(n_layers + 1):
        x = layer * 2
        
        for qubit in range(n_qubits):
            y = qubit * qubit_spacing
            
            if layer == 0:
                ax.plot(x, y, 'o', markersize=15, color='lightblue', markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y, f'q{qubit}', ha='center', va='center', fontweight='bold', fontsize=10)
            elif layer <= n_layers:
                ax.plot(x, y, 's', markersize=12, color='lightcoral', markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y, 'U', ha='center', va='center', fontweight='bold', fontsize=9, color='white')
            else:
                ax.plot(x, y, 'o', markersize=15, color='lightgreen', markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y, f'<Z{qubit}>', ha='center', va='center', fontweight='bold', fontsize=9)
            
            if layer < n_layers:
                for next_qubit in range(n_qubits):
                    if next_qubit != qubit:
                        ax.plot([x + 0.3, x + 1.7], [qubit * qubit_spacing, next_qubit * qubit_spacing], 
                               'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlim(-0.5, (n_layers + 1) * 2 + 0.5)
    ax.set_ylim(-0.5, (n_qubits - 1) * qubit_spacing + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Quantum Circuit Structure\n(AngleEmbedding + BasicEntanglerLayers + Measurement)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def visualize_quantum_outputs(model, X_test, n_samples=50):
    """Visualize quantum layer outputs"""
    model.eval()
    device = next(model.parameters()).device
    n_qubits = model.n_qubits if hasattr(model, 'n_qubits') else 4
    
    X_test_tensor = torch.tensor(X_test[:n_samples], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        classical_out = model.classical_net(X_test_tensor)
        quantum_input = X_test_tensor[:, :min(n_qubits, X_test_tensor.shape[1])]
        
        if quantum_input.shape[1] < n_qubits:
            padding = torch.zeros(quantum_input.shape[0], n_qubits - quantum_input.shape[1], device=device)
            quantum_input = torch.cat([quantum_input, padding], dim=1)
        
        quantum_outputs = []
        for i in range(X_test_tensor.shape[0]):
            q_out = model.quantum_net(quantum_input[i:i+1])
            if isinstance(q_out, torch.Tensor):
                quantum_outputs.append(q_out.detach().cpu().numpy().flatten())
            else:
                quantum_outputs.append(np_regular.asarray(q_out).flatten())
        
        quantum_out_np = np_regular.array(quantum_outputs)
    
    # Dynamically create subplot grid based on n_qubits
    n_cols = min(n_qubits, 3)
    n_rows = (n_qubits + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_qubits == 1:
        axes = [axes]
    elif n_rows == 1:
        if hasattr(axes, '__len__') and not isinstance(axes, str):
            axes = list(axes)
        else:
            axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i in range(n_qubits):
        ax = axes[i] if n_qubits > 1 else axes[0]
        ax.hist(quantum_out_np[:, i], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(f'Qubit {i} Output', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Quantum Output Distribution - Qubit {i}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(n_qubits, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

