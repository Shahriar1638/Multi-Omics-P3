#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T06:37:08.900Z
"""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Load CSVs (features x samples format, so we need to transpose)
gene_df = pd.read_csv("../NewDatasets/processed_expression_4O.csv", index_col=0).T
meth_df = pd.read_csv("../NewDatasets/processed_methylation_4O.csv", index_col=0).T
cnv_df  = pd.read_csv("../NewDatasets/processed_cnv_4O.csv", index_col=0).T

# apply standard scaling to all data
scaler_meth = StandardScaler()
meth_df = pd.DataFrame(
    scaler_meth.fit_transform(meth_df.T).T,
    index=meth_df.index,
    columns=meth_df.columns
)

scaler_cnv = StandardScaler()
cnv_df = pd.DataFrame(
    scaler_cnv.fit_transform(cnv_df.T).T,
    index=cnv_df.index,
    columns=cnv_df.columns
)
scaler_gene = StandardScaler()
gene_df = pd.DataFrame(
    scaler_gene.fit_transform(gene_df.T).T,
    index=gene_df.index,
    columns=gene_df.columns
)

print("After transpose - Shapes (samples x features):")
print(f"gene_df: {gene_df.shape}")
print(f"meth_df: {meth_df.shape}")
print(f"cnv_df: {cnv_df.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Shapes:", gene_df.shape, meth_df.shape, cnv_df.shape)
phenotype_df = pd.read_csv("../NewDatasets/phenotype_data_clean_FXS_MOFA_3Omics.csv", index_col=0)
# Align samples across all three omics and phenotype data
common = gene_df.index.intersection(meth_df.index).intersection(cnv_df.index).intersection(phenotype_df.index)
gene_df = gene_df.loc[common]
meth_df = meth_df.loc[common]
cnv_df  = cnv_df.loc[common]
phenotype_df = phenotype_df.loc[common]

print("Shapes after alignment:", gene_df.shape, meth_df.shape, cnv_df.shape, phenotype_df.shape)
# Convert to float32 tensors
gene = torch.tensor(gene_df.values, dtype=torch.float32).to(device)
meth = torch.tensor(meth_df.values, dtype=torch.float32).to(device)
cnv  = torch.tensor(cnv_df.values, dtype=torch.float32).to(device)


def Split(data):
    from sklearn.model_selection import train_test_split
    import numpy as np
    # Load subtype labels
    labels_df = pd.read_csv("../NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)

    # Align labels with our data
    common_samples = data.index.intersection(labels_df.index)
    print(f"Samples with labels: {len(common_samples)}")

    # Filter data and labels to common samples
    labels = labels_df.loc[common_samples].values.ravel()
    # Prepare features and labels
    X = data  # Fused latent features
    y = labels  # Cancer subtype labels

    print("="*70)
    print("STRATIFIED DATA SPLITTING BY DISEASE SUBTYPE")
    print("="*70)
    print(f"\n📊 Total samples: {len(X)}")
    print(f"\n🔬 Disease subtype distribution:")

    # Get unique disease subtypes
    unique_subtypes = np.unique(y)
    for subtype_idx in unique_subtypes:
        count = np.sum(y == subtype_idx)
        print(f"   - Subtype {subtype_idx}: {count} samples")

    # ====================================================================
    # Split each disease subtype 80/20, then merge
    # ====================================================================
    print(f"\n{'='*70}")
    print("🔹 Train/Test Split - Stratified by Disease:")
    print(f"{'='*70}")

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for subtype_idx in unique_subtypes:
        # Get samples for this subtype
        subtype_mask = (y == subtype_idx)
        X_subtype = X[subtype_mask]
        y_subtype = y[subtype_mask]
        
        # Split this subtype 80/20
        X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
            X_subtype, y_subtype, test_size=0.2, random_state=42
        )
        
        X_train_list.append(X_train_sub)
        X_test_list.append(X_test_sub)
        y_train_list.append(y_train_sub)
        y_test_list.append(y_test_sub)
        
        print(f"   Subtype {subtype_idx}:")
        print(f"      Train: {len(X_train_sub)} samples ({len(X_train_sub)/len(X_subtype)*100:.1f}%)")
        print(f"      Test:  {len(X_test_sub)} samples ({len(X_test_sub)/len(X_subtype)*100:.1f}%)")

    # Merge all subtypes
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================================
# HyperDNN: Hypernetwork-based Deep Neural Network for Multi-Omics Integration
# ============================================================================
# Uses LOW-RANK hypernetwork to efficiently generate modality-specific weights
# This avoids memory issues with high-dimensional omics data
# ============================================================================

class LowRankHyperNetwork(nn.Module):
    """
    Low-rank hypernetwork that generates weights efficiently using factorization.
    Instead of generating W directly (in_dim × out_dim), generates:
    W = U @ V where U is (in_dim × rank) and V is (rank × out_dim)
    This dramatically reduces memory for high-dimensional inputs.
    """
    def __init__(self, embedding_dim, hidden_dim, in_features, out_features, rank=16):
        super(LowRankHyperNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Generate low-rank factors instead of full weight matrix
        # U: (in_features × rank), V: (rank × out_features)
        self.U_generator = nn.Linear(hidden_dim, in_features * rank)
        self.V_generator = nn.Linear(hidden_dim, rank * out_features)
        self.bias_generator = nn.Linear(hidden_dim, out_features)
        
        # Scaling factor
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, embedding):
        features = self.backbone(embedding)
        
        # Generate low-rank factors
        U = self.U_generator(features).view(self.in_features, self.rank)
        V = self.V_generator(features).view(self.rank, self.out_features)
        
        # Reconstruct weight matrix: W = U @ V (transposed for F.linear)
        weight = (U @ V).T * self.scale  # (out_features, in_features)
        bias = self.bias_generator(features)
        
        return weight, bias


class HyperEncoder(nn.Module):
    """
    Memory-efficient encoder using low-rank hypernetwork-generated weights.
    Only the first layer uses hypernetwork (most benefit for high-dim input).
    Subsequent layers use standard learnable weights.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, embedding_dim, 
                 hyper_hidden_dim=64, rank=16, dropout=0.3):
        super(HyperEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        # Hypernetwork only for the first (high-dimensional) layer
        self.hyper_layer = LowRankHyperNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hyper_hidden_dim,
            in_features=input_dim,
            out_features=hidden_dims[0],
            rank=rank
        )
        
        # Standard layers for the rest (much smaller, no need for hypernetwork)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Final layer to latent space
        self.final_layer = nn.Linear(prev_dim, latent_dim)
        self.first_bn = nn.BatchNorm1d(hidden_dims[0])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, embedding):
        # First layer: use hypernetwork-generated weights
        weight, bias = self.hyper_layer(embedding)
        x = F.linear(x, weight, bias)
        x = self.first_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Subsequent layers: standard weights
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final projection
        z = self.final_layer(x)
        return z


class HyperDecoder(nn.Module):
    """Standard decoder (low-dimensional, no hypernetwork needed)."""
    def __init__(self, latent_dim, hidden_dims, output_dim, dropout=0.3):
        super(HyperDecoder, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class HyperDNN(nn.Module):
    """
    Memory-Efficient HyperDNN for Multi-Omics Integration.
    
    Uses low-rank hypernetworks to generate modality-specific encoder weights
    based on learnable omics-type embeddings.
    """
    def __init__(self, input_dims, latent_dim=32, hyper_hidden_dim=64, 
                 embedding_dim=16, encoder_hidden_dims=[128, 64], 
                 rank=16, dropout=0.4):
        """
        Args:
            input_dims: Dict mapping omics names to input dimensions
            latent_dim: Dimension of the shared latent space
            hyper_hidden_dim: Hidden dimension of the hypernetwork
            embedding_dim: Dimension of omics-type embeddings
            encoder_hidden_dims: Hidden layer dimensions for encoders
            rank: Rank for low-rank weight approximation
            dropout: Dropout rate
        """
        super(HyperDNN, self).__init__()
        
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.omics_names = list(input_dims.keys())
        self.num_omics = len(self.omics_names)
        self.embedding_dim = embedding_dim
        
        # Learnable omics-type embeddings
        self.omics_embeddings = nn.Embedding(self.num_omics, embedding_dim)
        
        # Create encoders and decoders for each omics type
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for name, input_dim in input_dims.items():
            # Encoder with hypernetwork for first layer
            self.encoders[name] = HyperEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                latent_dim=latent_dim,
                embedding_dim=embedding_dim,
                hyper_hidden_dim=hyper_hidden_dim,
                rank=rank,
                dropout=dropout
            )
            
            # Standard decoder
            self.decoders[name] = HyperDecoder(
                latent_dim=latent_dim,
                hidden_dims=list(reversed(encoder_hidden_dims)),
                output_dim=input_dim,
                dropout=dropout
            )
        
        # Register omics indices as buffer (stays on correct device)
        for i, name in enumerate(self.omics_names):
            self.register_buffer(f'omics_idx_{name}', torch.tensor(i))
    
    def get_omics_embedding(self, omics_name, device):
        """Get the embedding for an omics type on the correct device."""
        idx = getattr(self, f'omics_idx_{omics_name}')
        return self.omics_embeddings(idx)
    
    def encode(self, x, omics_name):
        """Encode input from a specific omics type."""
        embedding = self.get_omics_embedding(omics_name, x.device)
        z = self.encoders[omics_name](x, embedding)
        return z
    
    def decode(self, z, omics_name):
        """Decode latent representation to a specific omics type."""
        return self.decoders[omics_name](z)
    
    def forward(self, inputs):
        """
        Forward pass for all omics types.
        
        Args:
            inputs: Dict mapping omics names to input tensors
        
        Returns:
            latents: Dict of latent representations
            reconstructions: Dict of reconstructed outputs
        """
        latents = {}
        reconstructions = {}
        
        for name, x in inputs.items():
            z = self.encode(x, name)
            recon = self.decode(z, name)
            latents[name] = z
            reconstructions[name] = recon
        
        return latents, reconstructions
    
    def get_fused_latent(self, inputs, fusion='concat'):
        """Get fused latent representation from all omics."""
        latents, _ = self.forward(inputs)
        latent_list = [latents[name] for name in self.omics_names]
        
        if fusion == 'concat':
            return torch.cat(latent_list, dim=1)
        elif fusion == 'mean':
            return torch.stack(latent_list, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")


# ------------------ Training with validation & early stopping ------------------
def train_hyperdnn(model, train_inputs, val_inputs, epochs=3000, patience=30, lr=1e-4):
    """Train HyperDNN with validation and early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = np.inf
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        _, train_recons = model(train_inputs)
        
        train_loss = 0
        for name in train_inputs.keys():
            train_loss += criterion(train_recons[name], train_inputs[name])
        train_loss /= len(train_inputs)

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            _, val_recons = model(val_inputs)
            val_loss = 0
            for name in val_inputs.keys():
                val_loss += criterion(val_recons[name], val_inputs[name])
            val_loss /= len(val_inputs)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss.item():.4f} | Val: {val_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("HyperDNN Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return model


# ============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
# Automatically finds the best hyperparameters for HyperDNN using Bayesian
# optimization with Tree-structured Parzen Estimator (TPE).
# ============================================================================

import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')


def create_hyperdnn_model(trial, input_dims, device):
    """
    Create a HyperDNN model with hyperparameters suggested by Optuna.
    
    Args:
        trial: Optuna trial object
        input_dims: Dict mapping omics names to input dimensions
        device: Torch device
    
    Returns:
        model: HyperDNN model with suggested hyperparameters
    """
    # Suggest hyperparameters
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 48, 64])
    
    # Encoder hidden dimensions (2-3 layers)
    n_encoder_layers = trial.suggest_int('n_encoder_layers', 2, 3)
    encoder_hidden_dims = []
    for i in range(n_encoder_layers):
        dim = trial.suggest_categorical(f'encoder_hidden_{i}', [64, 128, 256, 512])
        encoder_hidden_dims.append(dim)
    # Ensure decreasing dimensions
    encoder_hidden_dims = sorted(encoder_hidden_dims, reverse=True)
    
    # Hypernetwork parameters
    hyper_hidden_dim = trial.suggest_categorical('hyper_hidden_dim', [32, 64, 128])
    embedding_dim = trial.suggest_categorical('embedding_dim', [8, 16, 32])
    rank = trial.suggest_categorical('rank', [8, 16, 32, 48])
    
    # Regularization
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    
    # Create model
    model = HyperDNN(
        input_dims=input_dims,
        latent_dim=latent_dim,
        hyper_hidden_dim=hyper_hidden_dim,
        embedding_dim=embedding_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        rank=rank,
        dropout=dropout
    ).to(device)
    
    return model


def train_hyperdnn_for_optuna(model, train_inputs, val_inputs, trial, 
                               epochs=500, patience=20):
    """
    Training function optimized for Optuna with early stopping and pruning.
    
    Args:
        model: HyperDNN model
        train_inputs: Training data dict
        val_inputs: Validation data dict
        trial: Optuna trial for pruning
        epochs: Maximum training epochs
        patience: Early stopping patience
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    # Suggest optimizer hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = np.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        _, train_recons = model(train_inputs)
        
        train_loss = 0
        for name in train_inputs.keys():
            train_loss += criterion(train_recons[name], train_inputs[name])
        train_loss /= len(train_inputs)
        
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, val_recons = model(val_inputs)
            val_loss = 0
            for name in val_inputs.keys():
                val_loss += criterion(val_recons[name], val_inputs[name])
            val_loss /= len(val_inputs)
        
        scheduler.step(val_loss)
        
        # Report to Optuna for pruning
        trial.report(val_loss.item(), epoch)
        
        # Handle pruning (stop unpromising trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss


def objective(trial, input_dims, train_inputs, val_inputs, device):
    """
    Optuna objective function for HyperDNN hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        input_dims: Dict mapping omics names to input dimensions
        train_inputs: Training data dict
        val_inputs: Validation data dict
        device: Torch device
    
    Returns:
        val_loss: Validation loss (to minimize)
    """
    # Create model with suggested hyperparameters
    model = create_hyperdnn_model(trial, input_dims, device)
    
    # Train and evaluate
    val_loss = train_hyperdnn_for_optuna(
        model, train_inputs, val_inputs, trial,
        epochs=500, patience=20
    )
    
    return val_loss


def run_optuna_optimization(input_dims, train_inputs, val_inputs, device,
                           n_trials=50, timeout=3600, study_name="hyperdnn_optimization"):
    """
    Run Optuna hyperparameter optimization for HyperDNN.
    
    Args:
        input_dims: Dict mapping omics names to input dimensions
        train_inputs: Training data dict
        val_inputs: Validation data dict
        device: Torch device
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        study_name: Name for the study (for persistence)
    
    Returns:
        study: Optuna study object with results
        best_model: Best HyperDNN model found
    """
    print("="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR HYPERDNN")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of trials: {n_trials}")
    print(f"  Timeout: {timeout}s ({timeout/60:.1f} min)")
    print(f"  Device: {device}")
    
    # Create Optuna study with TPE sampler and MedianPruner
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
        interval_steps=10
    )
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize validation loss
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective function with fixed arguments
    def objective_wrapper(trial):
        return objective(trial, input_dims, train_inputs, val_inputs, device)
    
    # Run optimization
    print("\nStarting optimization...")
    print("-"*70)
    
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True  # Clean up GPU memory
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    # Trial statistics
    pruned_trials = len([t for t in study.trials if t.state == TrialState.PRUNED])
    complete_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    
    print(f"\nTrial Statistics:")
    print(f"  Completed trials: {complete_trials}")
    print(f"  Pruned trials: {pruned_trials}")
    print(f"  Total trials: {len(study.trials)}")
    
    # Best trial
    print(f"\nBest Trial:")
    print(f"  Value (Val Loss): {study.best_trial.value:.6f}")
    print(f"  Trial Number: {study.best_trial.number}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Reconstruct best model
    print("\nReconstructing best model...")
    best_params = study.best_trial.params
    
    # Build encoder hidden dims from best params
    n_layers = best_params['n_encoder_layers']
    encoder_hidden_dims = sorted(
        [best_params[f'encoder_hidden_{i}'] for i in range(n_layers)],
        reverse=True
    )
    
    best_model = HyperDNN(
        input_dims=input_dims,
        latent_dim=best_params['latent_dim'],
        hyper_hidden_dim=best_params['hyper_hidden_dim'],
        embedding_dim=best_params['embedding_dim'],
        encoder_hidden_dims=encoder_hidden_dims,
        rank=best_params['rank'],
        dropout=best_params['dropout']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in best_model.parameters())
    print(f"\nBest Model Parameters: {total_params:,}")
    
    return study, best_model


def visualize_optuna_results(study):
    """
    Visualize Optuna optimization results.
    
    Args:
        study: Completed Optuna study
    """
    try:
        import optuna.visualization as vis
        
        print("\n" + "="*70)
        print("OPTIMIZATION VISUALIZATION")
        print("="*70)
        
        # Create visualizations
        fig_history = vis.plot_optimization_history(study)
        fig_history.show()
        
        fig_importance = vis.plot_param_importances(study)
        fig_importance.show()
        
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.show()
        
        fig_contour = vis.plot_contour(study, params=['latent_dim', 'rank'])
        fig_contour.show()
        
    except ImportError:
        # Fallback to matplotlib if plotly not available
        print("\nPlotting optimization history with matplotlib...")
        
        trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        values = [t.value for t in trials]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        axes[0].plot(range(len(values)), values, 'b-o', alpha=0.6, markersize=4)
        best_values = [min(values[:i+1]) for i in range(len(values))]
        axes[0].plot(range(len(best_values)), best_values, 'r-', linewidth=2, label='Best Value')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Hyperparameter importance (simple estimate)
        param_values = {}
        for t in trials:
            for k, v in t.params.items():
                if k not in param_values:
                    param_values[k] = []
                param_values[k].append((v, t.value))
        
        # Calculate correlation with loss for numeric params
        importances = {}
        for param, vals in param_values.items():
            try:
                xs = [v[0] for v in vals]
                ys = [v[1] for v in vals]
                if isinstance(xs[0], (int, float)):
                    from scipy.stats import pearsonr
                    corr, _ = pearsonr(xs, ys)
                    importances[param] = abs(corr)
            except:
                pass
        
        if importances:
            sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            params, imps = zip(*sorted_params[:8])
            axes[1].barh(params, imps, color='steelblue', alpha=0.7)
            axes[1].set_xlabel('Importance (correlation with loss)')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def get_optimized_hyperdnn(input_dims, train_inputs, val_inputs, device,
                          n_trials=50, timeout=3600, retrain_epochs=3000):
    """
    Complete pipeline: Run Optuna optimization and retrain best model fully.
    
    Args:
        input_dims: Dict mapping omics names to input dimensions
        train_inputs: Training data dict
        val_inputs: Validation data dict
        device: Torch device
        n_trials: Number of Optuna trials
        timeout: Optimization timeout in seconds
        retrain_epochs: Epochs for final retraining
    
    Returns:
        best_model: Fully trained best HyperDNN model
        study: Optuna study for analysis
    """
    # Run optimization
    study, best_model = run_optuna_optimization(
        input_dims, train_inputs, val_inputs, device,
        n_trials=n_trials, timeout=timeout
    )
    
    # Visualize results
    visualize_optuna_results(study)
    
    # Retrain best model with full epochs
    print("\n" + "="*70)
    print("RETRAINING BEST MODEL WITH FULL EPOCHS")
    print("="*70)
    
    best_lr = study.best_trial.params['learning_rate']
    best_model = train_hyperdnn(
        best_model, train_inputs, val_inputs,
        epochs=retrain_epochs, patience=50, lr=best_lr
    )
    
    return best_model, study


# ------------------ Initialize and Train HyperDNN ------------------
print("="*70)
print("HYPERDNN: Low-Rank Hypernetwork for Multi-Omics Integration")
print("="*70)

# Get dimensions
gene_dim, meth_dim, cnv_dim = gene.shape[1], meth.shape[1], cnv.shape[1]
print(f"\nInput dimensions:")
print(f"  Gene Expression: {gene_dim}")
print(f"  Methylation: {meth_dim}")
print(f"  CNV: {cnv_dim}")
print(f"\nUsing device: {device}")

# Define input dimensions
input_dims = {
    'gene': gene_dim,
    'meth': meth_dim,
    'cnv': cnv_dim
}

# Create HyperDNN model with memory-efficient settings
hyperdnn = HyperDNN(
    input_dims=input_dims,
    latent_dim=32,
    hyper_hidden_dim=64,
    embedding_dim=16,
    encoder_hidden_dims=[128, 64],  # Smaller hidden dims
    rank=16,  # Low-rank approximation
    dropout=0.4
).to(device)

print(f"\nHyperDNN Architecture:")
print(f"  Latent dimension: 32")
print(f"  Hypernetwork hidden dim: 64")
print(f"  Omics embedding dim: 16")
print(f"  Encoder hidden layers: [128, 64]")
print(f"  Low-rank: 16")

# Count parameters
total_params = sum(p.numel() for p in hyperdnn.parameters())
trainable_params = sum(p.numel() for p in hyperdnn.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Verify model is on GPU
print(f"\nModel device check:")
for name, param in list(hyperdnn.named_parameters())[:2]:
    print(f"  {name}: {param.device}")

# Prepare train/val splits
n_samples = gene.shape[0]
n_train = int(0.8 * n_samples)

train_inputs = {
    'gene': gene[:n_train],
    'meth': meth[:n_train],
    'cnv': cnv[:n_train]
}

val_inputs = {
    'gene': gene[n_train:],
    'meth': meth[n_train:],
    'cnv': cnv[n_train:]
}

print(f"\nTraining samples: {n_train}")
print(f"Validation samples: {n_samples - n_train}")

# Train the model
print("\n" + "="*70)
print("Training HyperDNN...")
print("="*70)
hyperdnn = train_hyperdnn(hyperdnn, train_inputs, val_inputs, epochs=3000, patience=30)

# ------------------ Extract latents & fuse ------------------
print("\n" + "="*70)
print("Extracting fused latent representations...")
print("="*70)

hyperdnn.eval()
all_inputs = {
    'gene': gene,
    'meth': meth,
    'cnv': cnv
}

with torch.no_grad():
    z_fused = hyperdnn.get_fused_latent(all_inputs, fusion='concat')
    latents, _ = hyperdnn(all_inputs)

print(f"\nLatent representations:")
for name, z in latents.items():
    print(f"  {name}: {z.shape}")
print(f"\nFused latent shape: {z_fused.shape}")

from sklearn.metrics import silhouette_score
import seaborn as sns

# ------------------ HyperDNN Diagnostics ------------------
def hyperdnn_reconstruction_error(model, train_inputs, val_inputs, device):
    """Compute and plot per-sample reconstruction errors for train/val."""
    
    def compute_errors(inputs):
        model.eval()
        with torch.no_grad():
            _, recons = model(inputs)
        
        errors = {}
        for name in inputs.keys():
            err = ((recons[name].cpu().numpy() - inputs[name].cpu().numpy())**2).mean(axis=1)
            errors[name] = err
        
        # Average across all omics
        avg_errors = np.mean([errors[name] for name in inputs.keys()], axis=0)
        return errors, avg_errors
    
    train_errors, train_avg = compute_errors(train_inputs)
    val_errors, val_avg = compute_errors(val_inputs)
    
    # Plot per-omics reconstruction errors
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    for idx, name in enumerate(train_inputs.keys()):
        sns.kdeplot(train_errors[name], label="Train", shade=True, ax=axes[idx])
        sns.kdeplot(val_errors[name], label="Validation", shade=True, ax=axes[idx])
        axes[idx].set_xlabel("Reconstruction Error")
        axes[idx].set_ylabel("Density")
        axes[idx].set_title(f"{name.upper()} Reconstruction Errors")
        axes[idx].legend()
    
    # Average errors
    sns.kdeplot(train_avg, label="Train", shade=True, ax=axes[3])
    sns.kdeplot(val_avg, label="Validation", shade=True, ax=axes[3])
    axes[3].set_xlabel("Reconstruction Error")
    axes[3].set_ylabel("Density")
    axes[3].set_title("Average Reconstruction Errors")
    axes[3].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Reconstruction Error Summary:")
    print("-" * 60)
    for name in train_inputs.keys():
        print(f"{name.upper()}:")
        print(f"  Train Median: {np.median(train_errors[name]):.4f}")
        print(f"  Val Median:   {np.median(val_errors[name]):.4f}")
    print(f"\nAVERAGE:")
    print(f"  Train Median: {np.median(train_avg):.4f}")
    print(f"  Val Median:   {np.median(val_avg):.4f}")
    
    return train_errors, val_errors


def visualize_omics_embeddings(model):
    """Visualize the learned omics-type embeddings."""
    model.eval()
    
    embeddings = model.omics_embeddings.weight.detach().cpu().numpy()
    omics_names = model.omics_names
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Heatmap of embeddings
    im = axes[0].imshow(embeddings, aspect='auto', cmap='coolwarm')
    axes[0].set_yticks(range(len(omics_names)))
    axes[0].set_yticklabels([name.upper() for name in omics_names])
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].set_title("Omics Type Embeddings")
    plt.colorbar(im, ax=axes[0])
    
    # Pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    im2 = axes[1].imshow(sim_matrix, cmap='Blues', vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(omics_names)))
    axes[1].set_yticks(range(len(omics_names)))
    axes[1].set_xticklabels([name.upper() for name in omics_names])
    axes[1].set_yticklabels([name.upper() for name in omics_names])
    axes[1].set_title("Omics Embedding Similarity")
    
    # Add text annotations
    for i in range(len(omics_names)):
        for j in range(len(omics_names)):
            axes[1].text(j, i, f'{sim_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=10)
    
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.show()


def compare_latent_spaces(model, inputs):
    """Compare latent spaces from different omics types."""
    model.eval()
    with torch.no_grad():
        latents, _ = model(inputs)
    
    # Get latent representations
    z_gene = latents['gene'].cpu().numpy()
    z_meth = latents['meth'].cpu().numpy()
    z_cnv = latents['cnv'].cpu().numpy()
    
    # Compute correlations between latent spaces
    from scipy.stats import pearsonr
    
    print("Latent Space Correlations (mean across dimensions):")
    print("-" * 50)
    
    corrs = {
        'Gene-Meth': np.mean([pearsonr(z_gene[:, i], z_meth[:, i])[0] for i in range(z_gene.shape[1])]),
        'Gene-CNV': np.mean([pearsonr(z_gene[:, i], z_cnv[:, i])[0] for i in range(z_gene.shape[1])]),
        'Meth-CNV': np.mean([pearsonr(z_meth[:, i], z_cnv[:, i])[0] for i in range(z_meth.shape[1])])
    }
    
    for pair, corr in corrs.items():
        print(f"  {pair}: {corr:.3f}")
    
    return corrs


# ------------------ Run Diagnostics ------------------
print("="*70)
print("HYPERDNN DIAGNOSTICS")
print("="*70)

# 1. Check reconstruction errors
print("\n1. Reconstruction Error Analysis")
print("-"*50)
train_inputs_diag = {'gene': gene[:n_train], 'meth': meth[:n_train], 'cnv': cnv[:n_train]}
val_inputs_diag = {'gene': gene[n_train:], 'meth': meth[n_train:], 'cnv': cnv[n_train:]}
train_errs, val_errs = hyperdnn_reconstruction_error(hyperdnn, train_inputs_diag, val_inputs_diag, device)

# 2. Visualize omics embeddings
print("\n2. Omics Type Embeddings")
print("-"*50)
visualize_omics_embeddings(hyperdnn)

# 3. Compare latent spaces
print("\n3. Latent Space Comparison")
print("-"*50)
all_inputs_diag = {'gene': gene, 'meth': meth, 'cnv': cnv}
corrs = compare_latent_spaces(hyperdnn, all_inputs_diag)

# 4. Clustering sanity check on fused latents
print("\n4. Clustering Sanity Check")
print("-"*50)
from sklearn.cluster import KMeans
z_np = z_fused.cpu().numpy()
kmeans = KMeans(n_clusters=3, random_state=42).fit(z_np)
score = silhouette_score(z_np, kmeans.labels_)
print(f"Silhouette Score on fused latents: {score:.3f}")

from sklearn.cluster import KMeans

# number of clusters: if unknown, try different values & compare
n_clusters = 3  

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(z_fused.cpu().numpy())


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

X = z_fused.cpu().numpy()

sil_score = silhouette_score(X, cluster_labels)
ch_score  = calinski_harabasz_score(X, cluster_labels)
db_score  = davies_bouldin_score(X, cluster_labels)

print("Silhouette Score:", sil_score)              # higher is better ([-1, 1])
print("Calinski-Harabasz Score:", ch_score)        # higher is better
print("Davies-Bouldin Score:", db_score)           # lower is better



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(z_fused.cpu().numpy())

plt.scatter(z_2d[:,0], z_2d[:,1], c=cluster_labels, cmap="tab10")
plt.colorbar()
plt.title("t-SNE of Fused Latent Space")
plt.show()


# Load subtype labels
labels_df = pd.read_csv("../NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)

# Align labels with our data
common_samples = gene_df.index.intersection(labels_df.index)
print(f"Samples with labels: {len(common_samples)}")

# Filter data and labels to common samples
labels = labels_df.loc[common_samples].values.ravel()
z_fused_labeled = z_fused.cpu().numpy()

# If data was already aligned, we might need to reindex
if len(common_samples) < len(gene_df):
    # Refilter if needed
    gene_idx = gene_df.index.get_indexer(common_samples)
    z_fused_labeled = z_fused.cpu().numpy()[gene_idx]

print(f"Final data shape: {z_fused_labeled.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique classes: {np.unique(labels)}")
print(f"Class distribution: {np.bincount(labels.astype(int))}")

# Visualize HyperDNN embeddings with actual labels
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Get the fused embeddings and labels
embeddings = z_fused_labeled
true_labels = labels

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {true_labels.shape}")
print(f"Unique labels: {np.unique(true_labels)}")

# Create label names (assuming 0,1,2,3 are different cancer subtypes)
label_names = {0: 'Subtype 0', 1: 'Subtype 1', 2: 'Subtype 2', 3: 'Subtype 3'}
colored_labels = [label_names.get(int(label), f'Subtype {int(label)}') for label in true_labels]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('HyperDNN Embeddings Visualization with True Cancer Subtype Labels', fontsize=16, fontweight='bold')

# 1. PCA 2D visualization
print("Computing PCA...")
pca_2d = PCA(n_components=2, random_state=42)
embeddings_pca_2d = pca_2d.fit_transform(embeddings)

scatter1 = axes[0, 0].scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1], 
                             c=true_labels, cmap='tab10', alpha=0.7, s=50)
axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
axes[0, 0].set_title('PCA Visualization')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Subtype')

# 2. t-SNE 2D visualization
print("Computing t-SNE...")
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_tsne_2d = tsne_2d.fit_transform(embeddings)

scatter2 = axes[0, 1].scatter(embeddings_tsne_2d[:, 0], embeddings_tsne_2d[:, 1], 
                             c=true_labels, cmap='tab10', alpha=0.7, s=50)
axes[0, 1].set_xlabel('t-SNE 1')
axes[0, 1].set_ylabel('t-SNE 2')
axes[0, 1].set_title('t-SNE Visualization')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Subtype')

# 3. PCA 3D to 2D projections (PC1 vs PC3)
pca_3d = PCA(n_components=3, random_state=42)
embeddings_pca_3d = pca_3d.fit_transform(embeddings)

scatter3 = axes[0, 2].scatter(embeddings_pca_3d[:, 0], embeddings_pca_3d[:, 2], 
                             c=true_labels, cmap='tab10', alpha=0.7, s=50)
axes[0, 2].set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%} variance)')
axes[0, 2].set_ylabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%} variance)')
axes[0, 2].set_title('PCA: PC1 vs PC3')
axes[0, 2].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[0, 2], label='Subtype')

# 4. Cluster analysis comparison
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Perform K-means clustering
n_clusters = len(np.unique(true_labels))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
predicted_clusters = kmeans.fit_predict(embeddings)

# Calculate clustering metrics
ari_score = adjusted_rand_score(true_labels, predicted_clusters)
nmi_score = normalized_mutual_info_score(true_labels, predicted_clusters)

scatter4 = axes[1, 0].scatter(embeddings_pca_2d[:, 0], embeddings_pca_2d[:, 1], 
                             c=predicted_clusters, cmap='tab10', alpha=0.7, s=50)
axes[1, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
axes[1, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
axes[1, 0].set_title(f'K-means Clusters\nARI: {ari_score:.3f}, NMI: {nmi_score:.3f}')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter4, ax=axes[1, 0], label='Cluster')

# 5. Side-by-side comparison: True vs Predicted
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_labels, predicted_clusters)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted Cluster')
axes[1, 1].set_ylabel('True Subtype')
axes[1, 1].set_title('True Labels vs K-means Clusters')

# 6. Per-omics latent contribution analysis
# For HyperDNN with latent_dim=32 per omics, fused = 96 total (32*3)
latent_dim = 32
omic_contributions = {
    'Gene Expression': embeddings[:, :latent_dim],
    'Methylation': embeddings[:, latent_dim:2*latent_dim],
    'CNV': embeddings[:, 2*latent_dim:3*latent_dim]
}

omic_means = [np.mean(np.abs(contrib)) for contrib in omic_contributions.values()]
omic_stds = [np.std(np.abs(contrib)) for contrib in omic_contributions.values()]

bars = axes[1, 2].bar(omic_contributions.keys(), omic_means, 
                      yerr=omic_stds, capsize=5, alpha=0.7, 
                      color=['lightcoral', 'lightblue', 'lightgreen'])
axes[1, 2].set_ylabel('Mean Absolute Feature Value')
axes[1, 2].set_title('Average Contribution by Omic Type\n(HyperDNN Latent Space)')
axes[1, 2].set_xticklabels(omic_contributions.keys(), rotation=45)

# Add value labels on bars
for bar, mean_val in zip(bars, omic_means):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.05,
                    f'{mean_val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ====================================================================
# CENTRALIZED DATA SPLITTING FOR ALL ML CLASSIFIERS
# ====================================================================
# This cell creates train/test splits used by all models
# Each disease subtype is split 80/20 independently, then merged
# to ensure consistent comparison across different classifiers
# ====================================================================

from sklearn.model_selection import train_test_split
import numpy as np

# Prepare features and labels
X = z_fused_labeled  # Fused latent features
y = labels  # Cancer subtype labels

print("="*70)
print("STRATIFIED DATA SPLITTING BY DISEASE SUBTYPE")
print("="*70)
print(f"\n📊 Total samples: {len(X)}")
print(f"\n🔬 Disease subtype distribution:")

# Get unique disease subtypes
unique_subtypes = np.unique(y)
for subtype_idx in unique_subtypes:
    count = np.sum(y == subtype_idx)
    print(f"   - Subtype {subtype_idx}: {count} samples")

# ====================================================================
# Split each disease subtype 80/20, then merge
# ====================================================================
print(f"\n{'='*70}")
print("🔹 Train/Test Split - Stratified by Disease:")
print(f"{'='*70}")

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for subtype_idx in unique_subtypes:
    # Get samples for this subtype
    subtype_mask = (y == subtype_idx)
    X_subtype = X[subtype_mask]
    y_subtype = y[subtype_mask]
    
    # Split this subtype 80/20
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_subtype, y_subtype, test_size=0.2, random_state=42
    )
    
    X_train_list.append(X_train_sub)
    X_test_list.append(X_test_sub)
    y_train_list.append(y_train_sub)
    y_test_list.append(y_test_sub)
    
    print(f"   Subtype {subtype_idx}:")
    print(f"      Train: {len(X_train_sub)} samples ({len(X_train_sub)/len(X_subtype)*100:.1f}%)")
    print(f"      Test:  {len(X_test_sub)} samples ({len(X_test_sub)/len(X_subtype)*100:.1f}%)")

# Merge all subtypes
X_train = np.vstack(X_train_list)
X_test = np.vstack(X_test_list)
y_train = np.concatenate(y_train_list)
y_test = np.concatenate(y_test_list)

print(f"\n📦 Merged splits:")
print(f"   - Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Verify distribution in train and test
print(f"\n📊 Disease distribution in splits:")
for subtype_idx in unique_subtypes:
    train_count = np.sum(y_train == subtype_idx)
    test_count = np.sum(y_test == subtype_idx)
    print(f"   Subtype {subtype_idx}:")
    print(f"      Train: {train_count} ({train_count/len(y_train)*100:.1f}%)")
    print(f"      Test:  {test_count} ({test_count/len(y_test)*100:.1f}%)")

print(f"\n✅ Data splits created successfully with stratification by disease!")
print(f"   - All ML classifiers will use: X_train, X_test, y_train, y_test")
print("="*70)

# ====================================================================
# Using MLClassifier module for ML Classification
# X_train, X_test, y_train, y_test are already created from above
# ====================================================================

from ml_classifier import MLClassifier, train_classifiers

print(f"✅ Using centralized data splits:")
print(f"   Training set size: {X_train.shape[0]}")
print(f"   Test set size: {X_test.shape[0]}")
print(f"   Training class distribution: {np.bincount(y_train.astype(int))}")
print(f"   Test class distribution: {np.bincount(y_test.astype(int))}")

# Train and evaluate all ML classifiers using MLClassifier module
ml_classifier = MLClassifier(random_state=42, include_xgboost=True)
results = ml_classifier.train_and_evaluate(X_train, X_test, y_train, y_test, verbose=True)

# Print results summary
ml_classifier.print_results()

# Get results dataframe for later use
results_df = ml_classifier.get_results_dataframe()

# Get the classifiers dictionary for compatibility with later cells
classifiers = ml_classifier.classifiers

# Visualize ML classification results using MLClassifier module
ml_classifier.plot_results(figsize=(18, 12))

# ====================================================================
# ENSEMBLE MODEL TESTING - Using ensemble.py
# ====================================================================
# Import required modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import ensemble module
from ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    WeightedAverageEnsemble,
    BaggingEnsemble,
    BoostingEnsemble,
    create_default_classifiers,
    evaluate_ensemble,
    evaluate_multiple_ensembles,
    print_ensemble_comparison,
    plot_ensemble_comparison
)

print("="*70)
print("TRAINING & TESTING ENSEMBLE MODELS")
print("="*70)

# Create base classifiers
base_classifiers = create_default_classifiers(random_state=42)

# Dictionary to store all trained ensembles
ensembles = {}

# 1. Voting Ensemble (Soft)
print("\n🔹 Training Voting Ensemble (Soft)...")
voting_soft = VotingEnsemble(create_default_classifiers(random_state=42), voting='soft')
voting_soft.fit(X_train, y_train)
ensembles['Voting (Soft)'] = voting_soft

# 2. Voting Ensemble (Hard)
print("🔹 Training Voting Ensemble (Hard)...")
voting_hard = VotingEnsemble(create_default_classifiers(random_state=42), voting='hard')
voting_hard.fit(X_train, y_train)
ensembles['Voting (Hard)'] = voting_hard

# 3. Stacking Ensemble
print("🔹 Training Stacking Ensemble...")
stacking = StackingEnsemble(
    create_default_classifiers(random_state=42),
    meta_classifier=LogisticRegression(max_iter=1000)
)
stacking.fit(X_train, y_train)
ensembles['Stacking'] = stacking

# 4. Weighted Average Ensemble
print("🔹 Training Weighted Average Ensemble...")
weighted = WeightedAverageEnsemble(
    create_default_classifiers(random_state=42),
    weights='accuracy'
)
weighted.fit(X_train, y_train)
ensembles['Weighted Average'] = weighted

# 5. Bagging Ensemble
print("🔹 Training Bagging Ensemble...")
bagging = BaggingEnsemble(
    base_classifier=RandomForestClassifier(n_estimators=50, random_state=42),
    n_estimators=10,
    random_state=42
)
bagging.fit(X_train, y_train)
ensembles['Bagging'] = bagging

# 6. Boosting Ensemble (AdaBoost)
print("🔹 Training Boosting Ensemble (AdaBoost)...")
boosting = BoostingEnsemble(n_estimators=50, random_state=42)
boosting.fit(X_train, y_train)
ensembles['Boosting (AdaBoost)'] = boosting

print("\n✅ All ensemble models trained successfully!")
print(f"   Total ensembles: {len(ensembles)}")

# ====================================================================
# EVALUATE ALL ENSEMBLE MODELS
# ====================================================================
print("\n" + "="*70)
print("ENSEMBLE MODEL SCORES & EVALUATION")
print("="*70)

# Evaluate all ensembles using ensemble.py's evaluate_multiple_ensembles function
ensemble_results = evaluate_multiple_ensembles(ensembles, X_test, y_test)

# Print formatted results table
print("\n📊 ENSEMBLE PERFORMANCE COMPARISON:")
print("-"*80)
print_ensemble_comparison(ensemble_results)

# Display results as DataFrame
print("\n📋 Results DataFrame:")
print(ensemble_results.to_string(index=False))

# Get best ensemble
best_idx = ensemble_results['Accuracy'].idxmax()
best_ensemble = ensemble_results.loc[best_idx]

print("\n" + "="*70)
print("🏆 BEST ENSEMBLE MODEL:")
print("="*70)
print(f"   Model:         {best_ensemble['Model']}")
print(f"   Accuracy:      {best_ensemble['Accuracy']:.4f}")
print(f"   F1 (Macro):    {best_ensemble['F1 (Macro)']:.4f}")
print(f"   F1 (Micro):    {best_ensemble['F1 (Micro)']:.4f}")
print(f"   C-Index:       {best_ensemble['C-Index']:.4f}")

# ====================================================================
# VISUALIZATION
# ====================================================================
print("\n📈 Generating comparison plots...")
plot_ensemble_comparison(ensemble_results)

print("\n✅ Ensemble testing complete!")

# Cell 20: Prevent running cells below (choose one of the actions below)

# Option A: Quick abort (stops kernel/run-all)
# import sys
# sys.exit("Stopping execution here; cells below will not run.")

# Option B: Reusable stop flag + pre-run hook (preferred)
# STOP_FOLLOWING_CELLS = True

# def _stop_on_flag(*args, **kwargs):
#     if globals().get("STOP_FOLLOWING_CELLS"):
#         print("STOP_FOLLOWING_CELLS=True -- aborting further execution.")
#         raise SystemExit("Stopped by STOP_FOLLOWING_CELLS")

# get_ipython().events.register('pre_run_cell', _stop_on_flag)

# To resume running later:
# STOP_FOLLOWING_CELLS = False
# get_ipython().events.unregister('pre_run_cell', _stop_on_flag)

# # Detailed classification reports and confusion matrices
# import seaborn as sns
# from sklearn.metrics import classification_report

# for name in classifiers.keys():
#     print(f"\n{'='*80}")
#     print(f"DETAILED REPORT FOR MODEL: {name}")
#     print(f"{'='*80}\n")
    
#     print("Classification Report (per-class metrics):")
#     print("-"*80)
#     print(ml_classifier.get_classification_report(name))
    
#     cm = ml_classifier.get_confusion_matrix(name)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title(f'Confusion Matrix - {name}')
#     plt.show()
    
#     print(f"\nConfusion Matrix:")
#     print(cm)

# # Survival Analysis Visualization
# 
# Now let's perform survival analysis to understand the relationship between cancer subtypes (predicted by our HyperDNN model) and patient outcomes.


# Load and prepare survival data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
import warnings
warnings.filterwarnings('ignore')

# Load survival data
survival_df = pd.read_csv("../RawData/TCGA-SARC.survival.tsv", sep='\t')
print(f"Survival data shape: {survival_df.shape}")
print(f"Survival data columns: {survival_df.columns.tolist()}")
print(f"Sample survival data:")
print(survival_df.head())

# Check for missing values
print(f"\nMissing values:")
print(survival_df.isnull().sum())

# Basic statistics
print(f"\nSurvival statistics:")
print(f"OS.time range: {survival_df['OS.time'].min():.1f} - {survival_df['OS.time'].max():.1f} months")
print(f"Event rate (deaths): {survival_df['OS'].mean():.3f} ({survival_df['OS'].sum()} out of {len(survival_df)})")
print(f"Median follow-up time: {survival_df['OS.time'].median():.1f} months")

# Get predictions from the best model for survival analysis
best_model_name = results_df.iloc[0]['Model']
best_clf = classifiers[best_model_name]

# Get predictions for all samples (not just test set)
all_predictions = best_clf.predict(z_fused_labeled)
all_prediction_probs = best_clf.predict_proba(z_fused_labeled)

# Create a mapping of sample indices to predictions
# Assuming gene_df.index contains the sample IDs that match survival data
sample_ids = gene_df.index.tolist()
predictions_df = pd.DataFrame({
    'sample': sample_ids,
    'predicted_subtype': all_predictions,
    'subtype_0_prob': all_prediction_probs[:, 0],
    'subtype_1_prob': all_prediction_probs[:, 1] if all_prediction_probs.shape[1] > 1 else 0,
    'subtype_2_prob': all_prediction_probs[:, 2] if all_prediction_probs.shape[1] > 2 else 0,
    'subtype_3_prob': all_prediction_probs[:, 3] if all_prediction_probs.shape[1] > 3 else 0,
    'max_prob': np.max(all_prediction_probs, axis=1)  # Confidence score
})

print(f"Predictions shape: {predictions_df.shape}")
print(f"Predicted subtypes distribution:")
print(predictions_df['predicted_subtype'].value_counts().sort_index())
print(f"\nSample predictions:")
print(predictions_df.head())

# Merge survival data with predictions
# First, let's align the sample names properly
survival_df['sample_clean'] = survival_df['sample'].str.replace('-01A', '', regex=False)
predictions_df['sample_clean'] = predictions_df['sample'].str.replace('-01A', '', regex=False)

# Merge on cleaned sample names
survival_analysis_df = pd.merge(
    survival_df, 
    predictions_df, 
    left_on='sample_clean', 
    right_on='sample_clean', 
    how='inner'
)

print(f"Merged data shape: {survival_analysis_df.shape}")
print(f"Successfully matched {len(survival_analysis_df)} samples")

if len(survival_analysis_df) == 0:
    print("No matching samples found. Trying direct merge...")
    # Try direct merge without cleaning
    survival_analysis_df = pd.merge(
        survival_df, 
        predictions_df, 
        left_on='sample', 
        right_on='sample', 
        how='inner'
    )
    print(f"Direct merge result: {survival_analysis_df.shape}")

if len(survival_analysis_df) > 0:
    # Clean up column names - pandas creates sample_x and sample_y during merge
    if 'sample_x' in survival_analysis_df.columns:
        survival_analysis_df = survival_analysis_df.rename(columns={'sample_x': 'sample'})
        if 'sample_y' in survival_analysis_df.columns:
            survival_analysis_df = survival_analysis_df.drop(columns=['sample_y'])
    
    print(f"\nSubtype distribution in survival cohort:")
    print(survival_analysis_df['predicted_subtype'].value_counts().sort_index())
    print(f"\nSample of merged data:")
    print(survival_analysis_df[['sample', 'OS.time', 'OS', 'predicted_subtype', 'max_prob']].head())
else:
    print("Warning: No samples could be matched between survival and prediction data")
    print("Sample names in survival data:", survival_df['sample'].head().tolist())
    print("Sample names in prediction data:", predictions_df['sample'].head().tolist())

# Debug: Check the columns in survival_analysis_df
print("Columns in survival_analysis_df:")
print(survival_analysis_df.columns.tolist())
print("\nDataFrame info:")
print(survival_analysis_df.info())
print("\nFirst few rows:")
print(survival_analysis_df.head())

# Clean up column names for clarity
if 'sample_x' in survival_analysis_df.columns:
    survival_analysis_df = survival_analysis_df.rename(columns={'sample_x': 'sample'})
elif 'sample_y' in survival_analysis_df.columns:
    survival_analysis_df = survival_analysis_df.rename(columns={'sample_y': 'sample'})

# Also remove duplicate sample columns if they exist
columns_to_drop = []
if 'sample_y' in survival_analysis_df.columns and 'sample' in survival_analysis_df.columns:
    columns_to_drop.append('sample_y')
if 'sample_x' in survival_analysis_df.columns and 'sample' in survival_analysis_df.columns:
    columns_to_drop.append('sample_x')

if columns_to_drop:
    survival_analysis_df = survival_analysis_df.drop(columns=columns_to_drop)

print("Cleaned columns:", survival_analysis_df.columns.tolist())

# Comprehensive Survival Analysis Visualization
if len(survival_analysis_df) > 0:
    # Create a comprehensive survival analysis visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Survival Analysis: Cancer Subtypes Predicted by HyperDNN', 
                 fontsize=16, fontweight='bold')
    
    # Color palette for subtypes
    n_subtypes = len(survival_analysis_df['predicted_subtype'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, n_subtypes))
    subtype_colors = {subtype: colors[i] for i, subtype in enumerate(sorted(survival_analysis_df['predicted_subtype'].unique()))}
    
    # 1. Kaplan-Meier Survival Curves by Predicted Subtype
    kmf = KaplanMeierFitter()
    
    for subtype in sorted(survival_analysis_df['predicted_subtype'].unique()):
        subtype_data = survival_analysis_df[survival_analysis_df['predicted_subtype'] == subtype]
        kmf.fit(subtype_data['OS.time'], subtype_data['OS'], 
                label=f'Subtype {subtype} (n={len(subtype_data)})')
        kmf.plot_survival_function(ax=axes[0, 0], color=subtype_colors[subtype], 
                                 linewidth=2.5, ci_show=False)
    
    axes[0, 0].set_title('Kaplan-Meier Survival Curves by Predicted Subtype', fontweight='bold')
    axes[0, 0].set_xlabel('Time (months)')
    axes[0, 0].set_ylabel('Survival Probability')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best')
    
    # 2. Survival Statistics Summary
    axes[0, 1].axis('off')
    survival_stats = []
    
    for subtype in sorted(survival_analysis_df['predicted_subtype'].unique()):
        subtype_data = survival_analysis_df[survival_analysis_df['predicted_subtype'] == subtype]
        kmf.fit(subtype_data['OS.time'], subtype_data['OS'])
        
        median_survival = kmf.median_survival_time_
        event_rate = subtype_data['OS'].mean()
        n_patients = len(subtype_data)
        
        survival_stats.append({
            'Subtype': f'Subtype {subtype}',
            'N': n_patients,
            'Events': int(subtype_data['OS'].sum()),
            'Event Rate': f'{event_rate:.3f}',
            'Median Survival': f'{median_survival:.1f}' if not np.isnan(median_survival) else 'Not reached'
        })
    
    stats_df = pd.DataFrame(survival_stats)
    table = axes[0, 1].table(cellText=stats_df.values,
                            colLabels=stats_df.columns,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.15, 0.1, 0.15, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[0, 1].set_title('Survival Statistics by Subtype', fontweight='bold', pad=20)
    
    # 3. Log-rank Test Results
    try:
        # Perform pairwise log-rank tests
        subtypes = sorted(survival_analysis_df['predicted_subtype'].unique())
        if len(subtypes) > 1:
            # Overall log-rank test
            results = multivariate_logrank_test(
                survival_analysis_df['OS.time'],
                survival_analysis_df['predicted_subtype'],
                survival_analysis_df['OS']
            )
            
            axes[1, 0].axis('off')
            axes[1, 0].text(0.1, 0.8, f'Overall Log-rank Test:', fontweight='bold', fontsize=12)
            axes[1, 0].text(0.1, 0.7, f'Test Statistic: {results.test_statistic:.3f}', fontsize=11)
            axes[1, 0].text(0.1, 0.6, f'p-value: {results.p_value:.4f}', fontsize=11)
            significance = "Significant" if results.p_value < 0.05 else "Not Significant"
            axes[1, 0].text(0.1, 0.5, f'Result: {significance}', fontsize=11, 
                           color='red' if results.p_value < 0.05 else 'blue')
            axes[1, 0].set_title('Statistical Test Results', fontweight='bold')
            
            # Pairwise comparisons for first few subtypes
            pairwise_results = []
            for i in range(min(3, len(subtypes))):
                for j in range(i+1, min(3, len(subtypes))):
                    subtype_i_data = survival_analysis_df[survival_analysis_df['predicted_subtype'] == subtypes[i]]
                    subtype_j_data = survival_analysis_df[survival_analysis_df['predicted_subtype'] == subtypes[j]]
                    
                    if len(subtype_i_data) > 5 and len(subtype_j_data) > 5:  # Minimum sample size
                        lr_result = logrank_test(
                            subtype_i_data['OS.time'], subtype_j_data['OS.time'],
                            subtype_i_data['OS'], subtype_j_data['OS']
                        )
                        pairwise_results.append({
                            'Comparison': f'Subtype {subtypes[i]} vs {subtypes[j]}',
                            'Test Statistic': f'{lr_result.test_statistic:.3f}',
                            'p-value': f'{lr_result.p_value:.4f}',
                            'Significant': 'Yes' if lr_result.p_value < 0.05 else 'No'
                        })
            
            if pairwise_results:
                y_pos = 0.4
                axes[1, 0].text(0.1, y_pos, 'Pairwise Comparisons:', fontweight='bold', fontsize=12)
                for idx, result in enumerate(pairwise_results):
                    y_pos -= 0.08
                    axes[1, 0].text(0.1, y_pos, f"{result['Comparison']}: p={result['p-value']}", fontsize=10)
    
    except Exception as e:
        axes[1, 0].text(0.1, 0.5, f'Statistical test error: {str(e)}', fontsize=10)
        axes[1, 0].set_title('Statistical Test Results', fontweight='bold')
    
    # 4. Survival by Prediction Confidence
    # Create confidence groups based on max probability
    survival_analysis_df['confidence_group'] = pd.cut(
        survival_analysis_df['max_prob'], 
        bins=[0, 0.5, 0.75, 1.0], 
        labels=['Low (≤0.5)', 'Medium (0.5-0.75)', 'High (>0.75)']
    )
    
    for conf_group in survival_analysis_df['confidence_group'].dropna().unique():
        group_data = survival_analysis_df[survival_analysis_df['confidence_group'] == conf_group]
        if len(group_data) > 2:
            kmf.fit(group_data['OS.time'], group_data['OS'], label=f'{conf_group} (n={len(group_data)})')
            kmf.plot_survival_function(ax=axes[1, 1], linewidth=2.5, ci_show=False)
    
    axes[1, 1].set_title('Survival by Prediction Confidence', fontweight='bold')
    axes[1, 1].set_xlabel('Time (months)')
    axes[1, 1].set_ylabel('Survival Probability')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best')
    
    # 5. Hazard Ratios (Cox Proportional Hazards Model)
    try:
        # Prepare data for Cox regression
        cox_data = survival_analysis_df[['OS.time', 'OS', 'predicted_subtype']].copy()
        
        # Create dummy variables for subtypes (reference: subtype 0)
        subtype_dummies = pd.get_dummies(cox_data['predicted_subtype'], prefix='subtype')
        cox_data = pd.concat([cox_data, subtype_dummies], axis=1)
        
        # Remove reference category and original subtype column
        reference_col = f'subtype_{sorted(survival_analysis_df["predicted_subtype"].unique())[0]}'
        if reference_col in cox_data.columns:
            cox_data = cox_data.drop([reference_col, 'predicted_subtype'], axis=1)
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='OS.time', event_col='OS')
        
        # Plot hazard ratios
        hazard_ratios = cph.summary[['coef', 'exp(coef)', 'p']]
        hazard_ratios.columns = ['Log HR', 'Hazard Ratio', 'p-value']
        
        # Create hazard ratio plot
        hr_values = hazard_ratios['Hazard Ratio'].values
        hr_labels = [col.replace('subtype_', 'Subtype ') for col in hazard_ratios.index]
        
        bars = axes[2, 0].barh(range(len(hr_values)), hr_values, 
                              color=['red' if hr > 1 else 'blue' for hr in hr_values],
                              alpha=0.7)
        axes[2, 0].axvline(x=1, color='black', linestyle='--', alpha=0.8)
        axes[2, 0].set_yticks(range(len(hr_labels)))
        axes[2, 0].set_yticklabels(hr_labels)
        axes[2, 0].set_xlabel('Hazard Ratio')
        axes[2, 0].set_title('Hazard Ratios by Subtype\n(Reference: Subtype 0)', fontweight='bold')
        
        # Add HR values on bars
        for i, (bar, hr, p_val) in enumerate(zip(bars, hr_values, hazard_ratios['p-value'])):
            significance = '*' if p_val < 0.05 else ''
            axes[2, 0].text(hr + 0.05, bar.get_y() + bar.get_height()/2, 
                           f'{hr:.2f}{significance}', va='center', fontsize=9)
        
    except Exception as e:
        axes[2, 0].text(0.5, 0.5, f'Cox regression error: {str(e)}', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Hazard Ratios by Subtype', fontweight='bold')
    
    # 6. Survival Distribution by Subtype (Box Plot)
    survival_times_by_subtype = []
    subtype_labels = []
    
    for subtype in sorted(survival_analysis_df['predicted_subtype'].unique()):
        subtype_data = survival_analysis_df[survival_analysis_df['predicted_subtype'] == subtype]
        survival_times_by_subtype.append(subtype_data['OS.time'].values)
        subtype_labels.append(f'Subtype {subtype}')
    
    box_plot = axes[2, 1].boxplot(survival_times_by_subtype, labels=subtype_labels, 
                                 patch_artist=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], [subtype_colors[s] for s in sorted(survival_analysis_df['predicted_subtype'].unique())]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[2, 1].set_ylabel('Survival Time (months)')
    axes[2, 1].set_title('Distribution of Survival Times by Subtype', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SURVIVAL ANALYSIS SUMMARY (HyperDNN)")
    print("="*80)
    print(f"Total patients with survival data: {len(survival_analysis_df)}")
    print(f"Number of predicted subtypes: {len(survival_analysis_df['predicted_subtype'].unique())}")
    print(f"Overall event rate (deaths): {survival_analysis_df['OS'].mean():.3f}")
    print(f"Median follow-up time: {survival_analysis_df['OS.time'].median():.1f} months")
    
else:
    print("Cannot perform survival analysis - no matching samples between survival and prediction data")