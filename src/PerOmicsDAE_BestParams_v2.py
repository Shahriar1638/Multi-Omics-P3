# -*- coding: utf-8 -*-
"""
PerOmicsDAE with Best Hyperparameters - Notebook Version
=========================================================
This script is segmented for easy copy-paste into Jupyter notebooks.
Each segment starts with "# " which Jupyter recognizes as a cell boundary.

Uses Feature-Wise Gating and Anchor Loss architecture from Optuna optimization.
"""

#  [markdown]
# # PerOmicsDAE - Best Hyperparameters Evaluation
# This notebook evaluates the DAE-based multi-omics classification model using the best hyperparameters found by Optuna optimization.

#  [markdown]
# ## 1. Imports and Setup

# 
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# UMAP import
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP plots will be skipped.")

# Lifelines for C-index
try:
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Warning: lifelines not installed. C-index will not be computed.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

#  [markdown]
# ## 2. Best Hyperparameters (from Optuna)

# 
BEST_PARAMS = {
    'n_features': 2000,
    'latent_dim': 64,
    'hidden_dim': 256,
    'fusion_hidden_dim': 64,
    'lr_pre': 0.00021458006346062385,
    'lr_fine': 0.0006664627497719737,
    'dropout_rate': 0.16900036913444771,
    'dropout_encoder': 0.2704898398454909,
    'noise_level': 0.10406316876248747,
    'noise_type': 'uniform',
    'focal_gamma': 4.234547523000666,
    'focal_alpha_scale': 2.9971972638490785,
    'ntxent_weight': 4.917776272955244,
    'ntxent_temperature': 0.3159090275955854,
    'anchor_weight': 0.3705068361102422,
    'weight_decay': 1e-4,
}

SUBTYPES_OF_INTEREST = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]

print("Best Hyperparameters loaded!")
print(f"  Latent Dim: {BEST_PARAMS['latent_dim']}")
print(f"  Hidden Dim: {BEST_PARAMS['hidden_dim']}")
print(f"  Learning Rate (Pretrain): {BEST_PARAMS['lr_pre']:.6f}")
print(f"  Learning Rate (Fine-tune): {BEST_PARAMS['lr_fine']:.6f}")
print(f"  Noise Level: {BEST_PARAMS['noise_level']:.4f}")
print(f"  Noise Type: {BEST_PARAMS['noise_type']}")
print(f"  Anchor Weight: {BEST_PARAMS['anchor_weight']:.4f}")

#  [markdown]
# ## 3. Data Loading Function

# 
def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")

    pheno_path = "Data/phenotype_clean.csv"
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"{pheno_path} not found.")

    pheno = pd.read_csv(pheno_path, index_col=0)

    col_name = 'primary_diagnosis.diagnoses'
    if col_name not in pheno.columns:
        print(f"Warning: '{col_name}' not found. Available: {pheno.columns.tolist()}")
        return None

    mask = pheno[col_name].isin(SUBTYPES_OF_INTEREST)
    pheno = pheno[mask]
    print(f"  Phenotype Samples (filtered): {pheno.shape[0]}")

    def load_omic(path, name):
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {name}.")
            return None
        df = pd.read_csv(path, index_col=0)
        df = df.T  # samples x features
        return df

    rna = load_omic("Data/expression_log.csv", "RNA (Expression)")
    meth = load_omic("Data/methylation_mvalues.csv", "Methylation")
    cnv = load_omic("Data/cnv_log.csv", "CNV")

    if rna is None or meth is None or cnv is None:
        raise ValueError("One or more omics files missing.")

    common_samples = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    print(f"  Common Samples: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("No common samples found!")

    pheno = pheno.loc[common_samples]
    rna = rna.loc[common_samples]
    meth = meth.loc[common_samples]
    cnv = cnv.loc[common_samples]

    le = LabelEncoder()
    Y = le.fit_transform(pheno[col_name])
    print(f"  Classes: {le.classes_}")

    class_counts = np.bincount(Y)
    class_weights = len(Y) / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()
    print(f"  Class Counts: {class_counts}")
    print(f"  Class Weights (normalized): {class_weights}")

    # Survival Data
    T, E = None, None
    if 'days_to_death' in pheno.columns and 'vital_status' in pheno.columns:
        events = (pheno['vital_status'].isin(['Dead', 'Deceased'])).astype(int).values
        times = np.zeros(len(pheno))
        if 'days_to_death' in pheno.columns:
            mask_d = events == 1
            d_times = pd.to_numeric(pheno['days_to_death'], errors='coerce').fillna(0).values
            times[mask_d] = d_times[mask_d]
        col_fup = 'days_to_last_follow_up' if 'days_to_last_follow_up' in pheno.columns else 'days_to_last_followup'
        if col_fup in pheno.columns:
            mask_a = events == 0
            f_times = pd.to_numeric(pheno[col_fup], errors='coerce').fillna(0).values
            times[mask_a] = f_times[mask_a]
        T = times
        E = events

    return rna, meth, cnv, Y, T, E, le.classes_, class_weights

#  [markdown]
# ## 4. Model Definitions

# 
class NTXentLoss(nn.Module):
    """NT-Xent Loss for contrastive learning"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -9e15)
        target = torch.arange(batch_size, device=z.device)
        target = torch.cat([target + batch_size, target], dim=0)
        loss = F.cross_entropy(sim, target)
        return loss


class PerOmicCMAE(nn.Module):
    """Per-Omic Denoising Autoencoder with Contrastive Learning"""
    
    def __init__(self, input_dim, latent_dim=64, hidden_dim=256, dropout_encoder=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout_encoder),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), 
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, noise_level=0.0, noise_type='gaussian', encode_only=False):
        if self.training and noise_level > 0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(x) * noise_level
                x_corrupted = x + noise
            elif noise_type == 'uniform':
                noise = (torch.rand_like(x) - 0.5) * 2 * noise_level
                x_corrupted = x + noise
            elif noise_type == 'dropout':
                mask = torch.bernoulli(torch.ones_like(x) * (1 - noise_level))
                x_corrupted = x * mask
            else:
                x_corrupted = x
        else:
            x_corrupted = x

        z = self.encoder(x_corrupted)
        
        if encode_only:
            return None, None, z, x_corrupted
        
        rec = self.decoder(z)
        proj = self.projector(z)

        return rec, proj, z, x_corrupted


class GatedFeatureFusion(nn.Module):
    """Feature-Wise Gating for multi-omics fusion (SOTA upgrade from scalar attention)"""
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3, hidden_dim=64):
        super().__init__()
        # Feature-wise gates output (latent_dim), not (1)
        self.gate_rna = nn.Linear(latent_dim, latent_dim)
        self.gate_meth = nn.Linear(latent_dim, latent_dim)
        self.gate_clin = nn.Linear(latent_dim, latent_dim)
        
        # LayerNorm is critical for stability with high gamma
        self.ln_rna = nn.LayerNorm(latent_dim)
        self.ln_meth = nn.LayerNorm(latent_dim)
        self.ln_clin = nn.LayerNorm(latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.drop_rate = dropout_rate

    def forward(self, z_rna, z_meth, z_clin, apply_dropout=False):
        # Feature-wise gating (Sigmoid)
        g_rna = torch.sigmoid(self.gate_rna(z_rna))
        g_meth = torch.sigmoid(self.gate_meth(z_meth))
        g_clin = torch.sigmoid(self.gate_clin(z_clin))

        # Apply gates to normalized embeddings
        h_rna = g_rna * self.ln_rna(z_rna)
        h_meth = g_meth * self.ln_meth(z_meth)
        h_clin = g_clin * self.ln_clin(z_clin)

        # Dropout on the embeddings themselves (modality dropout)
        if apply_dropout and self.training:
            if torch.rand(1).item() < self.drop_rate: h_rna = torch.zeros_like(h_rna)
            if torch.rand(1).item() < self.drop_rate: h_meth = torch.zeros_like(h_meth)
            if torch.rand(1).item() < self.drop_rate: h_clin = torch.zeros_like(h_clin)

        z_fused = torch.cat([h_rna, h_meth, h_clin], dim=1)
        
        # Return logits, gates (concatenated), and fused embedding
        return self.classifier(z_fused), torch.cat([g_rna, g_meth, g_clin], dim=1), z_fused


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

print("Model classes defined!")

#  [markdown]
# ## 5. Load Data

# 
# Load Data
result = load_raw_aligned_data()
if result is None:
    raise ValueError("Failed to load data!")
    
rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names, class_weights = result

print(f"\nData loaded successfully!")
print(f"  Total samples: {len(Y)}")
print(f"  Number of classes: {len(class_names)}")
print(f"  Class names: {list(class_names)}")

#  [markdown]
# ## 6. Training and Evaluation Function

# 
def run_full_evaluation(rna_df, meth_df, cnv_df, Y, T, E, class_names, class_weights, params):
    """
    Run 5-fold cross-validation with the best hyperparameters
    Returns embeddings for all samples for visualization
    """
    
    n_features = params['n_features']
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Per-fold metrics
    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_micro': [],
        'precision': [], 'recall': [], 'c_index': []
    }
    
    # Store all embeddings and predictions
    all_embeddings = np.zeros((len(Y), params['latent_dim'] * 3))  # fused embeddings
    all_preds = np.zeros(len(Y), dtype=int)
    all_probs = np.zeros((len(Y), len(class_names)))
    
    # Extract hyperparameters
    latent_dim = params['latent_dim']
    hidden_dim = params['hidden_dim']
    fusion_hidden_dim = params['fusion_hidden_dim']
    lr_pre = params['lr_pre']
    lr_fine = params['lr_fine']
    dropout_rate = params['dropout_rate']
    dropout_encoder = params['dropout_encoder']
    noise_level = params['noise_level']
    noise_type = params['noise_type']
    focal_gamma = params['focal_gamma']
    focal_alpha_scale = params['focal_alpha_scale']
    weight_decay = params['weight_decay']
    ntxent_temp = params['ntxent_temperature']
    ntxent_weight = params['ntxent_weight']
    anchor_weight = params['anchor_weight']
    
    MAX_EPOCHS_PRE = 700
    MAX_EPOCHS_FINE = 1200
    PRE_PATIENCE = 20  # For pretraining early stopping
    FINE_PATIENCE = 30  # For fine-tuning early stopping

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # --- A. Data Processing ---
        tr_rna_raw, val_rna_raw = rna_df.iloc[train_idx], rna_df.iloc[val_idx]
        tr_meth_raw, val_meth_raw = meth_df.iloc[train_idx], meth_df.iloc[val_idx]
        tr_cnv_raw, val_cnv_raw = cnv_df.iloc[train_idx], cnv_df.iloc[val_idx]

        def simple_impute_data(train_data, val_data):
            imputer = SimpleImputer(strategy='median')
            tr_imputed = imputer.fit_transform(train_data.values)
            val_imputed = imputer.transform(val_data.values)
            return tr_imputed, val_imputed

        tr_rna_imp, val_rna_imp = simple_impute_data(tr_rna_raw, val_rna_raw)
        tr_meth_imp, val_meth_imp = simple_impute_data(tr_meth_raw, val_meth_raw)
        tr_cnv_imp, val_cnv_imp = simple_impute_data(tr_cnv_raw, val_cnv_raw)

        def get_top_k_indices(data, k):
            vars = np.var(data, axis=0)
            return np.argpartition(vars, -k)[-k:] if data.shape[1] > k else np.arange(data.shape[1])

        r_idx = get_top_k_indices(tr_rna_imp, n_features)
        m_idx = get_top_k_indices(tr_meth_imp, n_features)
        c_idx = get_top_k_indices(tr_cnv_imp, n_features)

        tr_rna_sel = tr_rna_imp[:, r_idx]; val_rna_sel = val_rna_imp[:, r_idx]
        tr_meth_sel = tr_meth_imp[:, m_idx]; val_meth_sel = val_meth_imp[:, m_idx]
        tr_cnv_sel = tr_cnv_imp[:, c_idx]; val_cnv_sel = val_cnv_imp[:, c_idx]

        sc_r = StandardScaler(); sc_m = StandardScaler(); sc_c = StandardScaler()
        tr_rna = sc_r.fit_transform(tr_rna_sel); val_rna = sc_r.transform(val_rna_sel)
        tr_meth = sc_m.fit_transform(tr_meth_sel); val_meth = sc_m.transform(val_meth_sel)
        tr_cnv = sc_c.fit_transform(tr_cnv_sel); val_cnv = sc_c.transform(val_cnv_sel)

        dims = (tr_rna.shape[1], tr_meth.shape[1], tr_cnv.shape[1])

        # Tensors
        t_tr_r = torch.FloatTensor(tr_rna).to(DEVICE)
        t_tr_m = torch.FloatTensor(tr_meth).to(DEVICE)
        t_tr_c = torch.FloatTensor(tr_cnv).to(DEVICE)
        t_tr_y = torch.LongTensor(Y[train_idx]).to(DEVICE)

        t_val_r = torch.FloatTensor(val_rna).to(DEVICE)
        t_val_m = torch.FloatTensor(val_meth).to(DEVICE)
        t_val_c = torch.FloatTensor(val_cnv).to(DEVICE)
        t_val_y = torch.LongTensor(Y[val_idx]).to(DEVICE)

        # --- B. Model Init ---
        cmae_r = PerOmicCMAE(dims[0], latent_dim, hidden_dim, dropout_encoder).to(DEVICE)
        cmae_m = PerOmicCMAE(dims[1], latent_dim, hidden_dim, dropout_encoder).to(DEVICE)
        cmae_c = PerOmicCMAE(dims[2], latent_dim, hidden_dim, dropout_encoder).to(DEVICE)

        opt_pre = optim.AdamW(
            list(cmae_r.parameters()) + list(cmae_m.parameters()) + list(cmae_c.parameters()),
            lr=lr_pre, weight_decay=weight_decay
        )

        # --- C. Pretraining with Early Stopping ---
        criterion_ntxent = NTXentLoss(temperature=ntxent_temp).to(DEVICE)
        best_pre_val_loss = float('inf')
        patience_counter_pre = 0

        for epoch in range(MAX_EPOCHS_PRE):
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            
            rec_r, proj_r, _, _ = cmae_r(t_tr_r, noise_level=noise_level, noise_type=noise_type)
            rec_m, proj_m, _, _ = cmae_m(t_tr_m, noise_level=noise_level, noise_type=noise_type)
            rec_c, proj_c, _, _ = cmae_c(t_tr_c, noise_level=noise_level, noise_type=noise_type)

            loss_recon = F.mse_loss(rec_r, t_tr_r) + F.mse_loss(rec_m, t_tr_m) + F.mse_loss(rec_c, t_tr_c)
            loss_nt = (criterion_ntxent(proj_r, proj_m) + 
                       criterion_ntxent(proj_r, proj_c) + 
                       criterion_ntxent(proj_m, proj_c)) / 3.0
            loss = loss_recon + ntxent_weight * loss_nt

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()

            # Validation check every 10 epochs
            if epoch % 10 == 0:
                cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
                with torch.no_grad():
                    rr, rp, _, _ = cmae_r(t_val_r)
                    mr, mp, _, _ = cmae_m(t_val_m)
                    cr, cp, _, _ = cmae_c(t_val_c)
                    val_loss = F.mse_loss(rr, t_val_r) + F.mse_loss(mr, t_val_m) + F.mse_loss(cr, t_val_c)
                    
                    if val_loss < best_pre_val_loss:
                        best_pre_val_loss = val_loss
                        patience_counter_pre = 0
                    else:
                        patience_counter_pre += 1
            
            # Early stopping for pretraining
            if patience_counter_pre > PRE_PATIENCE:
                print(f"    Pretrain early stop at epoch {epoch}")
                break

        # --- D. Fine-tuning with Anchor Loss and Early Stopping ---
        fusion = GatedFeatureFusion(
            latent_dim, num_classes=len(np.unique(Y)), 
            dropout_rate=dropout_rate, hidden_dim=fusion_hidden_dim
        ).to(DEVICE)

        alpha = torch.FloatTensor(class_weights * focal_alpha_scale).to(DEVICE)
        criterion_cls = FocalLoss(gamma=focal_gamma, alpha=alpha)

        opt_fine = optim.AdamW(
            list(cmae_r.encoder.parameters()) + list(cmae_m.encoder.parameters()) + 
            list(cmae_c.encoder.parameters()) + list(fusion.parameters()),
            lr=lr_fine, weight_decay=weight_decay
        )

        best_f1 = 0.0
        best_state = None
        best_encoder_states = None
        fine_patience_counter = 0

        for epoch in range(MAX_EPOCHS_FINE):
            cmae_r.train(); cmae_m.train(); cmae_c.train(); fusion.train()
            
            # Anchor Loss: run with encode_only=False to get reconstruction for regularization
            rec_r, _, zr, _ = cmae_r(t_tr_r, noise_level=0.0, encode_only=False)
            rec_m, _, zm, _ = cmae_m(t_tr_m, noise_level=0.0, encode_only=False)
            rec_c, _, zc, _ = cmae_c(t_tr_c, noise_level=0.0, encode_only=False)
            
            logits, _, _ = fusion(zr, zm, zc, apply_dropout=True)
            
            # Primary Loss: Classification
            loss_cls = criterion_cls(logits, t_tr_y)
            
            # Auxiliary Loss: Anchor (Reconstruction) to prevent forgetting manifold
            loss_anchor = F.mse_loss(rec_r, t_tr_r) + F.mse_loss(rec_m, t_tr_m) + F.mse_loss(rec_c, t_tr_c)
            
            total_loss = loss_cls + (anchor_weight * loss_anchor)

            opt_fine.zero_grad()
            total_loss.backward()
            opt_fine.step()

            # Validation every 5 epochs
            if epoch % 5 == 0:
                cmae_r.eval(); cmae_m.eval(); cmae_c.eval(); fusion.eval()
                with torch.no_grad():
                    _, _, zr_v, _ = cmae_r(t_val_r, encode_only=True)
                    _, _, zm_v, _ = cmae_m(t_val_m, encode_only=True)
                    _, _, zc_v, _ = cmae_c(t_val_c, encode_only=True)
                    logits_v, _, z_fused_v = fusion(zr_v, zm_v, zc_v)
                    preds = logits_v.argmax(dim=1).cpu().numpy()
                    targets = t_val_y.cpu().numpy()
                    
                    f1 = f1_score(targets, preds, average='macro')
                    if f1 > best_f1:
                        best_f1 = f1
                        fine_patience_counter = 0
                        best_state = fusion.state_dict()
                        best_encoder_states = {
                            'cmae_r': cmae_r.state_dict(),
                            'cmae_m': cmae_m.state_dict(),
                            'cmae_c': cmae_c.state_dict()
                        }
                    else:
                        fine_patience_counter += 1
            
            # Early stopping for fine-tuning
            if fine_patience_counter >= FINE_PATIENCE:
                print(f"    Fine-tune early stop at epoch {epoch}, best F1={best_f1:.4f}")
                break

        # Load best states
        if best_state:
            fusion.load_state_dict(best_state)
        if best_encoder_states:
            cmae_r.load_state_dict(best_encoder_states['cmae_r'])
            cmae_m.load_state_dict(best_encoder_states['cmae_m'])
            cmae_c.load_state_dict(best_encoder_states['cmae_c'])
            
        # --- E. Final Evaluation ---
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        fusion.eval()
        with torch.no_grad():
            _, _, zr_val, _ = cmae_r(t_val_r, noise_level=0.0, encode_only=True)
            _, _, zm_val, _ = cmae_m(t_val_m, noise_level=0.0, encode_only=True)
            _, _, zc_val, _ = cmae_c(t_val_c, noise_level=0.0, encode_only=True)
            logits, _, z_fused_val = fusion(zr_val, zm_val, zc_val)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = t_val_y.cpu().numpy()
            
        # Store embeddings and predictions
        all_embeddings[val_idx] = z_fused_val.cpu().numpy()
        all_preds[val_idx] = preds
        all_probs[val_idx] = probs.cpu().numpy()

        acc = accuracy_score(targets, preds)
        f1_mac = f1_score(targets, preds, average='macro')
        f1_mic = f1_score(targets, preds, average='micro')
        prec = precision_score(targets, preds, average='macro', zero_division=0)
        rec = recall_score(targets, preds, average='macro', zero_division=0)

        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        
        # C-index
        c_idx_fold = -1
        if HAS_LIFELINES and T is not None and E is not None:
            val_T = T[val_idx]
            val_E = E[val_idx]
            risk_scores = np.sum(probs.cpu().numpy() * np.arange(len(class_names)), axis=1)
            try:
                c_idx_fold = concordance_index(val_T, -risk_scores, val_E)
            except:
                c_idx_fold = -1
        fold_metrics['c_index'].append(c_idx_fold)
        
        print(f"    Fold {fold+1}: Acc={acc:.3f}, F1-macro={f1_mac:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")

    return fold_metrics, all_embeddings, all_preds, all_probs

#  [markdown]
# ## 7. Run Evaluation

# 
print("Starting evaluation with best hyperparameters...")
fold_metrics, all_embeddings, all_preds, all_probs = run_full_evaluation(
    rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names, class_weights, BEST_PARAMS
)

#  [markdown]
# ## 8. Classification Results

# 
print(f"\n{'='*60}")
print("CLASSIFICATION RESULTS (5-Fold CV)")
print(f"{'='*60}")

mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items() if v}
std_metrics = {k: np.std(v) for k, v in fold_metrics.items() if v}

print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10}")
print(f"  {'-'*35}")
for key in ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'c_index']:
    if key in mean_metrics and all(x != -1 for x in fold_metrics[key]):
        print(f"  {key:<15} {mean_metrics[key]:>10.4f} {std_metrics[key]:>10.4f}")

# Classification Report
print(f"\n{'='*60}")
print("DETAILED CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(Y, all_preds, target_names=class_names, zero_division=0))

#  [markdown]
# ## 9. Confusion Matrix

# 
# Confusion Matrix Visualization
cm = confusion_matrix(Y, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (DAE Best Params)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_dae.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved to 'confusion_matrix_dae.png'")

#  [markdown]
# ## 10. Clustering Analysis (K-Means)

# 
# K-Means Clustering
n_clusters = len(class_names)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_embeddings)

print(f"\n{'='*60}")
print("CLUSTERING ANALYSIS (K-Means)")
print(f"{'='*60}")
print(f"  Number of clusters: {n_clusters}")

#  [markdown]
# ## 11. Clustering Scores

# 
# Compute Clustering Scores
print(f"\n{'='*60}")
print("CLUSTERING SCORES")
print(f"{'='*60}")

# 1. Silhouette Score (internal: -1 to 1, higher is better)
sil_score = silhouette_score(all_embeddings, Y)
sil_score_kmeans = silhouette_score(all_embeddings, cluster_labels)
print(f"  Silhouette Score (True Labels):    {sil_score:.4f}")
print(f"  Silhouette Score (K-Means):        {sil_score_kmeans:.4f}")

# 2. NMI - Normalized Mutual Information (0 to 1, higher is better)
nmi_score = normalized_mutual_info_score(Y, cluster_labels)
print(f"  NMI (Normalized Mutual Info):      {nmi_score:.4f}")

# 3. ARI - Adjusted Rand Index (-1 to 1, higher is better)
ari_score = adjusted_rand_score(Y, cluster_labels)
print(f"  ARI (Adjusted Rand Index):         {ari_score:.4f}")

# 4. Calinski-Harabasz Index (higher is better, measures cluster separation)
ch_score = calinski_harabasz_score(all_embeddings, Y)
ch_score_kmeans = calinski_harabasz_score(all_embeddings, cluster_labels)
print(f"  Calinski-Harabasz (True Labels):   {ch_score:.4f}")
print(f"  Calinski-Harabasz (K-Means):       {ch_score_kmeans:.4f}")

# 5. Davies-Bouldin Index (lower is better, measures cluster compactness)
db_score = davies_bouldin_score(all_embeddings, Y)
db_score_kmeans = davies_bouldin_score(all_embeddings, cluster_labels)
print(f"  Davies-Bouldin (True Labels):      {db_score:.4f}")
print(f"  Davies-Bouldin (K-Means):          {db_score_kmeans:.4f}")

# Summary table
cluster_summary = pd.DataFrame({
    'Metric': ['Silhouette', 'NMI', 'ARI', 'Calinski-Harabasz', 'Davies-Bouldin'],
    'Score (True Labels)': [sil_score, nmi_score, ari_score, ch_score, db_score],
    'Score (K-Means)': [sil_score_kmeans, nmi_score, ari_score, ch_score_kmeans, db_score_kmeans],
    'Interpretation': ['Higher=Better', 'Higher=Better', 'Higher=Better', 'Higher=Better', 'Lower=Better']
})
print(f"\n{cluster_summary.to_string(index=False)}")

#  [markdown]
# ## 12. t-SNE Visualization

# 
# t-SNE Visualization
print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_tsne = tsne.fit_transform(all_embeddings)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: True Labels
scatter1 = axes[0].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                           c=Y, cmap='tab10', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
axes[0].set_title('t-SNE: True Labels', fontsize=14, fontweight='bold')
axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
legend1 = axes[0].legend(handles=scatter1.legend_elements()[0], labels=list(class_names), 
                         title="Class", loc='upper right', fontsize=8)
axes[0].add_artist(legend1)

# Plot 2: K-Means Clusters
scatter2 = axes[1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                           c=cluster_labels, cmap='Set1', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
axes[1].set_title('t-SNE: K-Means Clusters', fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
legend2 = axes[1].legend(handles=scatter2.legend_elements()[0], 
                         labels=[f'Cluster {i}' for i in range(n_clusters)], 
                         title="Cluster", loc='upper right', fontsize=8)
axes[1].add_artist(legend2)

plt.tight_layout()
plt.savefig('tsne_visualization_dae.png', dpi=300, bbox_inches='tight')
plt.show()

print("t-SNE visualization saved to 'tsne_visualization_dae.png'")

#  [markdown]
# ## 13. UMAP Visualization

# 
# UMAP Visualization
if HAS_UMAP:
    print("\nComputing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_umap = reducer.fit_transform(all_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: True Labels
    scatter1 = axes[0].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], 
                               c=Y, cmap='tab10', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    axes[0].set_title('UMAP: True Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP Dimension 1', fontsize=11)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=11)
    legend1 = axes[0].legend(handles=scatter1.legend_elements()[0], labels=list(class_names), 
                             title="Class", loc='upper right', fontsize=8)
    axes[0].add_artist(legend1)

    # Plot 2: K-Means Clusters
    scatter2 = axes[1].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], 
                               c=cluster_labels, cmap='Set1', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    axes[1].set_title('UMAP: K-Means Clusters', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('UMAP Dimension 1', fontsize=11)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=11)
    legend2 = axes[1].legend(handles=scatter2.legend_elements()[0], 
                             labels=[f'Cluster {i}' for i in range(n_clusters)], 
                             title="Cluster", loc='upper right', fontsize=8)
    axes[1].add_artist(legend2)

    plt.tight_layout()
    plt.savefig('umap_visualization_dae.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("UMAP visualization saved to 'umap_visualization_dae.png'")
else:
    print("UMAP not available. Install with: pip install umap-learn")

#  [markdown]
# ## 14. Summary

# 
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

print("\n### Classification Metrics (5-Fold CV):")
print(f"  - Accuracy:   {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
print(f"  - F1-Macro:   {mean_metrics['f1_macro']:.4f} ± {std_metrics['f1_macro']:.4f}")
print(f"  - F1-Micro:   {mean_metrics['f1_micro']:.4f} ± {std_metrics['f1_micro']:.4f}")
print(f"  - Precision:  {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
print(f"  - Recall:     {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")

if all(x != -1 for x in fold_metrics['c_index']):
    print(f"  - C-Index:    {mean_metrics['c_index']:.4f} ± {std_metrics['c_index']:.4f}")

print("\n### Clustering Metrics:")
print(f"  - Silhouette Score:      {sil_score:.4f}")
print(f"  - NMI:                   {nmi_score:.4f}")
print(f"  - ARI:                   {ari_score:.4f}")
print(f"  - Calinski-Harabasz:     {ch_score:.4f}")
print(f"  - Davies-Bouldin:        {db_score:.4f}")

print("\n>>> EVALUATION COMPLETE!")
