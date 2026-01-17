import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import random
import sys
import os
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ==============================================================================

def load_data():
    print("Loading datasets (this may take a while)...")
    gene_df = pd.read_csv("../NewDatasets/processed_expression_4O.csv", index_col=0).T
    meth_df = pd.read_csv("../NewDatasets/processed_methylation_4O.csv", index_col=0).T
    cnv_df  = pd.read_csv("../NewDatasets/processed_cnv_4O.csv", index_col=0).T
    labels_df = pd.read_csv("../NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)
    phenotype_df = pd.read_csv("../NewDatasets/phenotype_data_clean_FXS_MOFA_3Omics.csv", index_col=0)
    return gene_df, meth_df, cnv_df, labels_df, phenotype_df

class RealDataset(Dataset):
    def __init__(self, gene_df, meth_df, cnv_df, labels_df, phenotype_df, var_percent=1.0):
        # --- Align Indices ---
        common_indices = gene_df.index.intersection(meth_df.index).intersection(cnv_df.index).intersection(labels_df.index).intersection(phenotype_df.index)
        
        self.gene_df = gene_df.loc[common_indices]
        self.meth_df = meth_df.loc[common_indices]
        self.cnv_df = cnv_df.loc[common_indices]
        self.labels_df = labels_df.loc[common_indices]
        self.phenotype_df = phenotype_df.loc[common_indices]

        # --- Variance Filtering ---
        if var_percent < 1.0:
            self.gene_df = self.filter_variance(self.gene_df, var_percent)
            self.meth_df = self.filter_variance(self.meth_df, var_percent)
            self.cnv_df = self.filter_variance(self.cnv_df, var_percent)

        # --- Standard Scaling ---
        scaler_g = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c = StandardScaler()
        
        gene_vals = scaler_g.fit_transform(self.gene_df.values)
        meth_vals = scaler_m.fit_transform(self.meth_df.values)
        cnv_vals = scaler_c.fit_transform(self.cnv_df.values)
        
        # --- Convert to Tensors ---
        self.rna = torch.tensor(gene_vals, dtype=torch.float32)
        self.meth = torch.tensor(meth_vals, dtype=torch.float32)
        self.cnv = torch.tensor(cnv_vals, dtype=torch.float32)
        
        # Labels
        self.labels = torch.tensor(self.labels_df.iloc[:,0].values, dtype=torch.long)
        self.num_classes = len(torch.unique(self.labels))

        # Mask (All ones for real data)
        self.mask = torch.ones(len(self.labels), 3)

    def filter_variance(self, df, percentile):
        variances = df.var()
        threshold = variances.quantile(1.0 - percentile)
        return df.loc[:, variances > threshold]

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.rna[idx], self.meth[idx], self.cnv[idx], self.mask[idx], self.labels[idx]

# ==============================================================================
# 2. BIO-PARSIMONIOUS NETWORK ARCHITECTURE (with Configurable Hyperparams)
# ==============================================================================

class ExplainableEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128, dropout=0.4, use_batch_norm=False):
        super().__init__()
        # The Gate: Learnable parameter deciding if a gene is useful
        self.feature_importance = nn.Parameter(torch.randn(input_dim) * 0.01)
        
        # Choose normalization type
        norm_layer = nn.BatchNorm1d if use_batch_norm else nn.LayerNorm
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            norm_layer(latent_dim)
        )

    def forward(self, x):
        # Apply the gate (Feature Selection)
        gate = torch.sigmoid(self.feature_importance)
        x_selected = x * gate
        return F.normalize(self.net(x_selected), dim=1)

class BioFMPN(nn.Module):
    def __init__(self, dims, num_classes, latent_dim=32, hidden_dim=128, dropout=0.4, 
                 attn_hidden=32, use_batch_norm=False):
        super().__init__()
        self.enc_rna = ExplainableEncoder(dims['rna'], latent_dim, hidden_dim, dropout, use_batch_norm)
        self.enc_meth = ExplainableEncoder(dims['meth'], latent_dim, hidden_dim, dropout, use_batch_norm)
        self.enc_cnv = ExplainableEncoder(dims['cnv'], latent_dim, hidden_dim, dropout, use_batch_norm)
       
        # Attention Gate for fusing modalities
        self.attn_gate = nn.Sequential(
            nn.Linear(latent_dim*3, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 3)
        )
        self.head = nn.Linear(latent_dim, num_classes, bias=False)

    def forward(self, rna, meth, cnv, mask):
        z_r = self.enc_rna(rna)
        z_m = self.enc_meth(meth)
        z_c = self.enc_cnv(cnv)
       
        # Zero out missing modalities for attention calculation
        z_cat = torch.cat([z_r*mask[:,0:1], z_m*mask[:,1:2], z_c*mask[:,2:3]], dim=1)
       
        # Calculate Attention Weights
        scores = self.attn_gate(z_cat)
        # Mask penalty (make missing modalities have -inf score)
        scores = scores + ((1.0 - mask) * -1e9)
        w = F.softmax(scores, dim=1)
       
        # Weighted Fusion
        z_fused = (w[:,0:1]*z_r) + (w[:,1:2]*z_m) + (w[:,2:3]*z_c)
        z_fused = F.normalize(z_fused, dim=1)
       
        return self.head(z_fused), w

# ==============================================================================
# 3. METRICS AND UTILITIES
# ==============================================================================

def calculate_metrics(y_true, logits):
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
   
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    try:
        auroc = roc_auc_score(y_true, probs, multi_class='ovr')
    except:
        auroc = 0.5
       
    return {'Acc': acc, 'BalAcc': bacc, 'F1': f1, 'AUROC': auroc}

# ==============================================================================
# 4. OPTUNA OBJECTIVE FUNCTION
# ==============================================================================

def create_objective(dataset, device, n_splits=3):
    """Creates an objective function for Optuna with the given dataset."""
    
    def objective(trial):
        # ==================================================
        # HYPERPARAMETER SEARCH SPACE
        # ==================================================
        
        # Architecture hyperparameters
        latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64, 128])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        attn_hidden = trial.suggest_categorical('attn_hidden', [16, 32, 64])
        
        # Regularization hyperparameters
        dropout = trial.suggest_float('dropout', 0.2, 0.7, step=0.1)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        l1_lambda = trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.05)
        
        # Optimizer hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Training hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # ==================================================
        # CROSS-VALIDATION TRAINING
        # ==================================================
        # Using StratifiedKFold for small dataset (n~205)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels.numpy())):
            
            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, train_idx), 
                batch_size=batch_size, 
                shuffle=True
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, val_idx), 
                batch_size=64, 
                shuffle=False
            )
            
            dims = {
                'rna': dataset.rna.shape[1], 
                'meth': dataset.meth.shape[1], 
                'cnv': dataset.cnv.shape[1]
            }
            
            model = BioFMPN(
                dims, 
                num_classes=dataset.num_classes,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                attn_hidden=attn_hidden,
                use_batch_norm=use_batch_norm
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            
            best_bacc = 0.0
            patience = 15
            counter = 0
            
            for epoch in range(100):
                # Training
                model.train()
                for rna, meth, cnv, mask, y in train_loader:
                    rna = rna.to(device)
                    meth = meth.to(device)
                    cnv = cnv.to(device)
                    mask = mask.to(device)
                    y = y.to(device)
                    
                    optimizer.zero_grad()
                    logits, w = model(rna, meth, cnv, mask)
                    
                    # L1 Sparsity Loss for feature selection
                    l1_reg = torch.sum(torch.sigmoid(model.enc_rna.feature_importance)) + \
                             torch.sum(torch.sigmoid(model.enc_meth.feature_importance)) + \
                             torch.sum(torch.sigmoid(model.enc_cnv.feature_importance))
                    
                    loss = criterion(logits, y) + l1_lambda * l1_reg
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Validation
                model.eval()
                val_logits, val_y = [], []
                with torch.no_grad():
                    for rna, meth, cnv, mask, y in val_loader:
                        rna = rna.to(device)
                        meth = meth.to(device)
                        cnv = cnv.to(device)
                        mask = mask.to(device)
                        y = y.to(device)
                        val_logits.append(model(rna, meth, cnv, mask)[0].cpu())
                        val_y.append(y.cpu())
                
                val_logits = torch.cat(val_logits).numpy()
                val_y = torch.cat(val_y).numpy()
                metrics = calculate_metrics(val_y, val_logits)
                
                if metrics['BalAcc'] > best_bacc:
                    best_bacc = metrics['BalAcc']
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        break
                
                # Pruning - report intermediate value for early stopping of bad trials
                trial.report(metrics['BalAcc'], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            fold_metrics.append(best_bacc)
        
        # Return mean balanced accuracy across folds
        return np.mean(fold_metrics)
    
    return objective

# ==============================================================================
# 5. OPTUNA OPTIMIZATION RUNNER
# ==============================================================================

def run_optuna_optimization(n_trials=100, var_percent=0.5):
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        n_trials: Number of optimization trials
        var_percent: Variance filtering percentage (keep top var_percent% high variance features)
    
    Returns:
        study: Optuna study object with results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Load data
    g_df, m_df, c_df, l_df, p_df = load_data()
    
    # Create dataset with variance filtering
    print(f"\nCreating dataset with top {var_percent*100}% high variance features...")
    dataset = RealDataset(g_df, m_df, c_df, l_df, p_df, var_percent=var_percent)
    print(f"Dataset created:")
    print(f"  - Samples: {len(dataset)}")
    print(f"  - RNA features: {dataset.rna.shape[1]}")
    print(f"  - Methylation features: {dataset.meth.shape[1]}")
    print(f"  - CNV features: {dataset.cnv.shape[1]}")
    print(f"  - Classes: {dataset.num_classes}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # Maximize balanced accuracy
        study_name='biofmpn_optimization',
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Create objective function
    objective = create_objective(dataset, device, n_splits=3)
    
    # Run optimization
    print(f"\n{'='*60}")
    print(f"Starting Optuna Optimization with {n_trials} trials")
    print(f"{'='*60}")
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        gc_after_trial=True  # Garbage collection after each trial (helps with memory)
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"  - Completed: {len(complete_trials)}")
    print(f"  - Pruned: {len(pruned_trials)}")
    
    print(f"\nBest Trial:")
    trial = study.best_trial
    
    print(f"  Value (Balanced Accuracy): {trial.value:.4f}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study

def train_with_best_params(study, var_percent=0.5, iterations=5):
    """
    Train the model with the best hyperparameters found by Optuna.
    
    Args:
        study: Optuna study object with optimization results
        var_percent: Variance filtering percentage
        iterations: Number of training runs for stability analysis
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get best hyperparameters
    best_params = study.best_trial.params
    
    print(f"\n{'='*60}")
    print("TRAINING WITH BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Load data
    g_df, m_df, c_df, l_df, p_df = load_data()
    dataset = RealDataset(g_df, m_df, c_df, l_df, p_df, var_percent=var_percent)
    
    all_metrics = []
    
    for i in range(iterations):
        torch.manual_seed(i + 3407)
        np.random.seed(i + 3407)
        
        # Stratified Split
        train_idx, val_idx = train_test_split(
            range(len(dataset)), 
            test_size=0.3, 
            stratify=dataset.labels,
            random_state=i + 3407
        )
        
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx), 
            batch_size=best_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx), 
            batch_size=64, 
            shuffle=False
        )
        
        dims = {
            'rna': dataset.rna.shape[1], 
            'meth': dataset.meth.shape[1], 
            'cnv': dataset.cnv.shape[1]
        }
        
        model = BioFMPN(
            dims, 
            num_classes=dataset.num_classes,
            latent_dim=best_params['latent_dim'],
            hidden_dim=best_params['hidden_dim'],
            dropout=best_params['dropout'],
            attn_hidden=best_params['attn_hidden'],
            use_batch_norm=best_params['use_batch_norm']
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=best_params['learning_rate'], 
            weight_decay=best_params['weight_decay']
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=best_params['label_smoothing'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        best_bacc = 0.0
        best_metrics = None
        patience = 15
        counter = 0
        
        for epoch in range(100):
            model.train()
            for rna, meth, cnv, mask, y in train_loader:
                rna, meth, cnv, mask, y = (
                    rna.to(device), meth.to(device), 
                    cnv.to(device), mask.to(device), y.to(device)
                )
                
                optimizer.zero_grad()
                logits, w = model(rna, meth, cnv, mask)
                
                l1_reg = (
                    torch.sum(torch.sigmoid(model.enc_rna.feature_importance)) +
                    torch.sum(torch.sigmoid(model.enc_meth.feature_importance)) +
                    torch.sum(torch.sigmoid(model.enc_cnv.feature_importance))
                )
                
                loss = criterion(logits, y) + best_params['l1_lambda'] * l1_reg
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_logits, val_y = [], []
            with torch.no_grad():
                for rna, meth, cnv, mask, y in val_loader:
                    rna, meth, cnv, mask, y = (
                        rna.to(device), meth.to(device), 
                        cnv.to(device), mask.to(device), y.to(device)
                    )
                    val_logits.append(model(rna, meth, cnv, mask)[0].cpu())
                    val_y.append(y.cpu())
            
            val_logits = torch.cat(val_logits).numpy()
            val_y = torch.cat(val_y).numpy()
            metrics = calculate_metrics(val_y, val_logits)
            
            if metrics['BalAcc'] > best_bacc:
                best_bacc = metrics['BalAcc']
                best_metrics = metrics
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
        
        all_metrics.append(best_metrics)
        print(f"Run {i+1}: BalAcc={best_metrics['BalAcc']:.3f}, F1={best_metrics['F1']:.3f}, AUROC={best_metrics['AUROC']:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS (with Best Hyperparameters)")
    print(f"{'='*60}")
    
    avg_bacc = np.mean([m['BalAcc'] for m in all_metrics])
    std_bacc = np.std([m['BalAcc'] for m in all_metrics])
    avg_f1 = np.mean([m['F1'] for m in all_metrics])
    std_f1 = np.std([m['F1'] for m in all_metrics])
    avg_auroc = np.mean([m['AUROC'] for m in all_metrics])
    std_auroc = np.std([m['AUROC'] for m in all_metrics])
    
    print(f"Balanced Accuracy: {avg_bacc:.3f} ± {std_bacc:.3f}")
    print(f"F1 Score:          {avg_f1:.3f} ± {std_f1:.3f}")
    print(f"AUROC:             {avg_auroc:.3f} ± {std_auroc:.3f}")
    
    return all_metrics

def save_best_params(study, filepath="best_hyperparams.txt"):
    """Save the best hyperparameters to a file."""
    with open(filepath, 'w') as f:
        f.write("Best Hyperparameters from Optuna Optimization\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best Balanced Accuracy: {study.best_trial.value:.4f}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\nBest hyperparameters saved to: {filepath}")

# ==============================================================================
# 6. QUICK OPTIMIZATION (For testing - fewer trials)
# ==============================================================================

def quick_optimization(n_trials=20, var_percent=0.5):
    """
    Quick optimization with fewer trials for testing purposes.
    Good for verifying the setup works before running full optimization.
    """
    print("="*60)
    print("QUICK OPTUNA OPTIMIZATION TEST")
    print("="*60)
    print(f"Number of trials: {n_trials}")
    print(f"Variance filter: Top {var_percent*100}% features")
    print("="*60)
    
    study = run_optuna_optimization(n_trials=n_trials, var_percent=var_percent)
    
    # Save results
    save_best_params(study)
    
    # Optional: Train with best params
    print("\nWould you like to train with best parameters? Running stability test...")
    metrics = train_with_best_params(study, var_percent=var_percent, iterations=3)
    
    return study, metrics

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization for BioFMPN')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--var_percent', type=float, default=0.5, help='Variance filtering percentage')
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer trials')
    parser.add_argument('--train_best', action='store_true', help='Train with best params after optimization')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test mode
        study, metrics = quick_optimization(n_trials=20, var_percent=args.var_percent)
    else:
        # Full optimization
        study = run_optuna_optimization(n_trials=args.n_trials, var_percent=args.var_percent)
        save_best_params(study)
        
        if args.train_best:
            metrics = train_with_best_params(study, var_percent=args.var_percent, iterations=5)
