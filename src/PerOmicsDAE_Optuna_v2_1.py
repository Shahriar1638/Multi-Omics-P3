# -*- coding: utf-8 -*-
"""
PeromicsCMAE with Optuna Hyperparameter Optimization
=====================================================
Optimizes:
- Latent dimensions
- Focal Loss alpha (per-class weights) and gamma
- Noise parameters (noise_level, noise_type)
- Learning rates, dropout, epochs, patience
- Architecture choices

Designed for: 205 samples, 4 imbalanced classes, no upsampling
"""

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
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
import optuna
from optuna.trial import TrialState
import warnings
warnings.filterwarnings('ignore')

# Try importing Lifelines for C-index
try:
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Feature count
N_FEATURES = 2000

SUBTYPES_OF_INTEREST = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]


def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")

    # 1. Load Phenotype/Labels
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

    # 2. Load Omics
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

    # 3. Intersection
    common_samples = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    print(f"  Common Samples: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("No common samples found!")

    pheno = pheno.loc[common_samples]
    rna = rna.loc[common_samples]
    meth = meth.loc[common_samples]
    cnv = cnv.loc[common_samples]

    # 4. Prepare Labels
    le = LabelEncoder()
    Y = le.fit_transform(pheno[col_name])
    print(f"  Classes: {le.classes_}")

    # 5. Compute class weights for imbalanced data (will be used for focal loss alpha)
    class_counts = np.bincount(Y)
    class_weights = len(Y) / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()  # Normalize
    print(f"  Class Counts: {class_counts}")
    print(f"  Class Weights (normalized): {class_weights}")

    # 6. Survival Data
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



class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Used for contrastive learning to align latent spaces.
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of pairs (z_i, z_j).
        Inputs:
            z_i, z_j: [batch_size, proj_dim] - Projected representations
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate: [2*B, D]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix: [2*B, 2*B]
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Mask out self-similarity (diagonals)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -9e15)
        
        # Targets:
        # z_i[k] should match z_j[k] -> index k matches index k + B
        # z_j[k] should match z_i[k] -> index k + B matches index k
        target = torch.arange(batch_size, device=z.device)
        target = torch.cat([target + batch_size, target], dim=0)
        
        loss = F.cross_entropy(sim, target)
        return loss


class PerOmicCMAE(nn.Module):
    """Per-Omic Contrastive Masked Autoencoder with configurable architecture"""
    
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
        """
        Forward pass with configurable noise injection
        
        Args:
            noise_type: 'gaussian', 'uniform', 'dropout', 'salt_pepper'
            encode_only: If True, skip decoder and projector (for fine-tuning efficiency)
        """
        if self.training and noise_level > 0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(x) * noise_level
                x_corrupted = x + noise
            elif noise_type == 'uniform':
                noise = (torch.rand_like(x) - 0.5) * 2 * noise_level
                x_corrupted = x + noise
            elif noise_type == 'dropout':
                # Zero out random features with probability = noise_level
                mask = torch.bernoulli(torch.ones_like(x) * (1 - noise_level))
                x_corrupted = x * mask
            elif noise_type == 'salt_pepper':
                # Salt and pepper noise
                mask = torch.rand_like(x)
                x_corrupted = x.clone()
                x_corrupted[mask < noise_level / 2] = -1.0  # pepper
                x_corrupted[mask > 1 - noise_level / 2] = 1.0  # salt
            else:
                x_corrupted = x
        else:
            x_corrupted = x

        z = self.encoder(x_corrupted)
        
        # Conditional forward: skip decoder/projector during fine-tuning
        if encode_only:
            return None, None, z, x_corrupted
        
        rec = self.decoder(z)
        proj = self.projector(z)

        return rec, proj, z, x_corrupted


class GatedAttentionFusion(nn.Module):
    # <--- METHODOLOGICAL FIX 2: Dynamic Classes
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3, hidden_dim=64):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_clin = nn.Linear(latent_dim, 1) 
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),  # Changed input dim for Weighted Concatenation
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.drop_rate = dropout_rate

    def forward(self, z_rna, z_meth, z_clin, apply_dropout=False):
        if apply_dropout and self.training:
            if torch.rand(1).item() < self.drop_rate: z_rna = torch.zeros_like(z_rna)
            if torch.rand(1).item() < self.drop_rate: z_meth = torch.zeros_like(z_meth)
            if torch.rand(1).item() < self.drop_rate: z_clin = torch.zeros_like(z_clin)

        w_rna = torch.sigmoid(self.gate_rna(z_rna))
        w_meth = torch.sigmoid(self.gate_meth(z_meth))
        w_clin = torch.sigmoid(self.gate_clin(z_clin))

        # Weighted Concatenation
        # Concatenate weighted representations: [B, 3 * latent_dim]
        z_fused = torch.cat([w_rna * z_rna, w_meth * z_meth, w_clin * z_clin], dim=1)
        
        return self.classifier(z_fused), torch.cat([w_rna, w_meth, w_clin], dim=1), z_fused


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples
        alpha: Per-class weights. Can be:
               - None: no class weighting
               - float: single weight applied to positive class (binary)
               - Tensor: per-class weights
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross entropy (log softmax + nll)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Gather alpha for each sample based on target class
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


def run_cv_evaluation(params, n_features, rna_df, meth_df, cnv_df, Y, class_names, 
                      class_weights, verbose=True, trial=None):
    """
    Run 5-fold cross-validation with given parameters
    
    Returns mean F1-macro score (primary optimization target)
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_micro': [],
        'precision': [], 'recall': [], 'silhouette': [], 'nmi': [], 'ari': []
    }

    # Extract hyperparameters
    latent_dim = params.get('latent_dim', 64)
    hidden_dim = params.get('hidden_dim', 256)
    fusion_hidden_dim = params.get('fusion_hidden_dim', 64)
    lr_pre = params.get('lr_pre', 1e-3)
    lr_fine = params.get('lr_fine', 1e-3)
    dropout_rate = params.get('dropout_rate', 0.2)
    dropout_encoder = params.get('dropout_encoder', 0.0)
    noise_level = params.get('noise_level', 0.1)
    noise_type = params.get('noise_type', 'gaussian')
    focal_gamma = params.get('focal_gamma', 2.0)
    focal_alpha_scale = params.get('focal_alpha_scale', 1.0)  # Scale for class weights
    use_class_weights = params.get('use_class_weights', True)
    weight_decay = params.get('weight_decay', 1e-4)
    
    # Robust early stopping parameters
    # Relative gap: (val_loss - train_loss) / train_loss
    # E.g., max_rel_gap=0.5 means val_loss can be at most 50% higher than train_loss
    max_rel_gap_pre = params.get('max_rel_gap_pre', 0.5)  # 50% max relative gap for pretrain
    max_rel_gap_fine = params.get('max_rel_gap_fine', 0.3)  # 30% max relative gap for finetune
    overfit_patience = params.get('overfit_patience', 10)  # Epochs to tolerate overfitting before stopping
    no_improve_patience = params.get('no_improve_patience', 30)  # Epochs without val_loss improvement
    
    # Fixed max epochs (not tuned)
    MAX_EPOCHS_PRE = 500
    MAX_EPOCHS_FINE = 1000

    if verbose:
        print(f"    Hyperparams: latent={latent_dim}, hidden={hidden_dim}, "
              f"lr_pre={lr_pre:.5f}, lr_fine={lr_fine:.5f}, "
              f"dropout={dropout_rate:.2f}, noise={noise_level:.3f} ({noise_type}), "
              f"focal_gamma={focal_gamma:.2f}, max_rel_gap_pre={max_rel_gap_pre:.0%}, max_rel_gap_fine={max_rel_gap_fine:.0%}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        # --- A. Data Processing (Leak-Safe) ---
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

        # --- C. Pretraining with Robust Early Stopping + LR Scheduler ---
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        best_pre_val_loss = float('inf')
        best_cmae_states = None
        overfit_counter_pre = 0  # Counts consecutive epochs where gap > threshold
        no_improve_counter_pre = 0  # Counts epochs without improvement

        # Prepare NT-Xent Loss
        ntxent_temp = params.get('ntxent_temperature', 0.5)
        ntxent_weight = params.get('ntxent_weight', 1.0)
        criterion_ntxent = NTXentLoss(temperature=ntxent_temp).to(DEVICE)

        # LR Scheduler: reduce LR before early stopping
        scheduler_pre = ReduceLROnPlateau(opt_pre, mode='min', factor=0.5, 
                                          patience=overfit_patience // 2, min_lr=1e-6)
        lr_reductions_pre = 0  # Track how many times LR has been reduced
        min_lr_reductions = 2  # Require at least 2 LR reductions before stopping
        prev_lr_pre = lr_pre

        for epoch in range(MAX_EPOCHS_PRE):
            # Training step
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            rec_r1, proj_r1, _, _ = cmae_r(t_tr_r, noise_level=noise_level, noise_type=noise_type)
            rec_m1, proj_m1, _, _ = cmae_m(t_tr_m, noise_level=noise_level, noise_type=noise_type)
            rec_c1, proj_c1, _, _ = cmae_c(t_tr_c, noise_level=noise_level, noise_type=noise_type)

            # Reconstruction Loss (Train)
            loss_recon_tr = F.mse_loss(rec_r1, t_tr_r) + F.mse_loss(rec_m1, t_tr_m) + F.mse_loss(rec_c1, t_tr_c)
            
            # Contrastive Loss (Pairwise)
            loss_nt = (criterion_ntxent(proj_r1, proj_m1) + 
                       criterion_ntxent(proj_r1, proj_c1) + 
                       criterion_ntxent(proj_m1, proj_c1)) / 3.0
            
            train_loss = loss_recon_tr + ntxent_weight * loss_nt

            opt_pre.zero_grad()
            train_loss.backward()
            opt_pre.step()

            # Validation step
            cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
            with torch.no_grad():
                rec_r_val, proj_r_val, _, _ = cmae_r(t_val_r, noise_level=0.0)
                rec_m_val, proj_m_val, _, _ = cmae_m(t_val_m, noise_level=0.0)
                rec_c_val, proj_c_val, _, _ = cmae_c(t_val_c, noise_level=0.0)
                
                loss_recon_val = F.mse_loss(rec_r_val, t_val_r) + F.mse_loss(rec_m_val, t_val_m) + F.mse_loss(rec_c_val, t_val_c)
                loss_nt_val = (criterion_ntxent(proj_r_val, proj_m_val) + 
                               criterion_ntxent(proj_r_val, proj_c_val) + 
                               criterion_ntxent(proj_m_val, proj_c_val)) / 3.0
                val_loss = loss_recon_val + ntxent_weight * loss_nt_val

            train_loss_item = train_loss.item()
            val_loss_item = val_loss.item()
            
            # Step the scheduler with val_loss
            scheduler_pre.step(val_loss_item)
            
            # Check if LR was reduced
            current_lr = opt_pre.param_groups[0]['lr']
            if current_lr < prev_lr_pre:
                lr_reductions_pre += 1
                prev_lr_pre = current_lr
            
            # Calculate relative gap: how much larger is val_loss compared to train_loss?
            rel_gap = (val_loss_item - train_loss_item) / max(train_loss_item, 0.01)

            # Save best model based on val_loss
            if val_loss_item < best_pre_val_loss:
                best_pre_val_loss = val_loss_item
                best_cmae_states = {
                    'cmae_r': cmae_r.state_dict(),
                    'cmae_m': cmae_m.state_dict(),
                    'cmae_c': cmae_c.state_dict()
                }
                no_improve_counter_pre = 0
            else:
                no_improve_counter_pre += 1
            
            # Check for overfitting (relative gap exceeds threshold)
            if rel_gap > max_rel_gap_pre:
                overfit_counter_pre += 1
            else:
                overfit_counter_pre = 0  # Reset if gap is acceptable
            
            # Early stopping conditions (only if LR has been reduced enough times):
            # 1. Consistent overfitting for overfit_patience epochs
            # 2. No improvement for no_improve_patience epochs
            if lr_reductions_pre >= min_lr_reductions:
                if overfit_counter_pre >= overfit_patience:
                    break  # Stop: consistent overfitting detected
                if no_improve_counter_pre >= no_improve_patience:
                    break  # Stop: val_loss not improving

        # Load best pretrain states
        if best_cmae_states:
            cmae_r.load_state_dict(best_cmae_states['cmae_r'])
            cmae_m.load_state_dict(best_cmae_states['cmae_m'])
            cmae_c.load_state_dict(best_cmae_states['cmae_c'])

        # --- D. Joint Training (Encoders + Fusion trained together) + LR Scheduler ---
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        
        fusion = GatedAttentionFusion(
            latent_dim, num_classes=len(np.unique(Y)), 
            dropout_rate=dropout_rate, hidden_dim=fusion_hidden_dim
        ).to(DEVICE)

        # Prepare focal loss with class weights
        if use_class_weights:
            alpha = torch.FloatTensor(class_weights * focal_alpha_scale).to(DEVICE)
        else:
            alpha = None
            
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)

        # Include encoder parameters in fine-tuning optimizer to allow embedding refinement
        opt_fine = optim.AdamW(
            list(cmae_r.encoder.parameters()) + list(cmae_m.encoder.parameters()) + list(cmae_c.encoder.parameters()) + list(fusion.parameters()),
            lr=lr_fine, weight_decay=weight_decay
        )

        # LR Scheduler for fine-tuning
        scheduler_fine = ReduceLROnPlateau(opt_fine, mode='min', factor=0.5,
                                           patience=overfit_patience // 2, min_lr=1e-7)
        lr_reductions_fine = 0
        prev_lr_fine = lr_fine

        best_fine_val_loss = float('inf')
        best_state = None
        best_encoder_states = None
        overfit_counter_fine = 0  # Counts consecutive epochs where gap > threshold
        no_improve_counter_fine = 0  # Counts epochs without improvement

        for epoch in range(MAX_EPOCHS_FINE):
            # Training step - compute embeddings with gradients enabled
            # Use encode_only=True to skip decoder/projector (not used in fine-tuning)
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            fusion.train()
            
            _, _, zr_tr, _ = cmae_r(t_tr_r, noise_level=0.0, encode_only=True)
            _, _, zm_tr, _ = cmae_m(t_tr_m, noise_level=0.0, encode_only=True)
            _, _, zc_tr, _ = cmae_c(t_tr_c, noise_level=0.0, encode_only=True)
            
            logits, weights, _ = fusion(zr_tr, zm_tr, zc_tr, apply_dropout=True)
            train_loss_cls = criterion(logits, t_tr_y)
            opt_fine.zero_grad()
            train_loss_cls.backward()
            opt_fine.step()

            # Validation step
            cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
            fusion.eval()
            with torch.no_grad():
                _, _, zr_val, _ = cmae_r(t_val_r, noise_level=0.0, encode_only=True)
                _, _, zm_val, _ = cmae_m(t_val_m, noise_level=0.0, encode_only=True)
                _, _, zc_val, _ = cmae_c(t_val_c, noise_level=0.0, encode_only=True)
                v_logits, _, v_fused = fusion(zr_val, zm_val, zc_val, apply_dropout=False)
                val_loss_cls = criterion(v_logits, t_val_y)

            train_loss_item = train_loss_cls.item()
            val_loss_item = val_loss_cls.item()
            
            # Step the scheduler with val_loss
            scheduler_fine.step(val_loss_item)
            
            # Check if LR was reduced
            current_lr = opt_fine.param_groups[0]['lr']
            if current_lr < prev_lr_fine:
                lr_reductions_fine += 1
                prev_lr_fine = current_lr
            
            # Calculate relative gap
            rel_gap = (val_loss_item - train_loss_item) / max(train_loss_item, 0.01)

            # Save best model based on val_loss
            if val_loss_item < best_fine_val_loss:
                best_fine_val_loss = val_loss_item
                best_state = fusion.state_dict()
                best_encoder_states = {
                    'cmae_r': cmae_r.state_dict(),
                    'cmae_m': cmae_m.state_dict(),
                    'cmae_c': cmae_c.state_dict()
                }
                no_improve_counter_fine = 0
            else:
                no_improve_counter_fine += 1
            
            # Check for overfitting
            if rel_gap > max_rel_gap_fine:
                overfit_counter_fine += 1
            else:
                overfit_counter_fine = 0  # Reset if gap is acceptable
            
            # Early stopping conditions (only if LR has been reduced enough times):
            # 1. Consistent overfitting for overfit_patience epochs
            # 2. No improvement for no_improve_patience epochs
            if lr_reductions_fine >= min_lr_reductions:
                if overfit_counter_fine >= overfit_patience:
                    break  # Stop: consistent overfitting detected
                if no_improve_counter_fine >= no_improve_patience:
                    break  # Stop: val_loss not improving

        if best_state:
            fusion.load_state_dict(best_state)
        if best_encoder_states:
            cmae_r.load_state_dict(best_encoder_states['cmae_r'])
            cmae_m.load_state_dict(best_encoder_states['cmae_m'])
            cmae_c.load_state_dict(best_encoder_states['cmae_c'])
            
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        fusion.eval()
        with torch.no_grad():
            _, _, zr_val, _ = cmae_r(t_val_r, noise_level=0.0, encode_only=True)
            _, _, zm_val, _ = cmae_m(t_val_m, noise_level=0.0, encode_only=True)
            _, _, zc_val, _ = cmae_c(t_val_c, noise_level=0.0, encode_only=True)
            logits, _, z_fused_val = fusion(zr_val, zm_val, zc_val)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = t_val_y.cpu().numpy()
            z_emb = z_fused_val.cpu().numpy()

        acc = accuracy_score(targets, preds)
        f1_mac = f1_score(targets, preds, average='macro')
        f1_mic = f1_score(targets, preds, average='micro')
        prec = precision_score(targets, preds, average='macro', zero_division=0)
        rec = recall_score(targets, preds, average='macro', zero_division=0)

        sil = -1
        if len(np.unique(targets)) > 1:
            try:
                sil = silhouette_score(z_emb, targets)
            except:
                pass

        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['silhouette'].append(sil)

        # Clustering metrics
        kmeans = KMeans(n_clusters=len(np.unique(Y)), random_state=42, n_init=10).fit(z_emb)
        nmi = normalized_mutual_info_score(targets, kmeans.labels_)
        ari = adjusted_rand_score(targets, kmeans.labels_)
        fold_metrics['nmi'].append(nmi)
        fold_metrics['ari'].append(ari)

        # Report intermediate value (for Optuna pruning)
        if trial is not None:
            trial.report(np.mean(fold_metrics['f1_macro']), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if verbose:
            print(f"    Fold {fold+1}: Acc={acc:.3f}, F1-macro={f1_mac:.3f}, F1-micro={f1_mic:.3f}")

    return {k: np.mean(v) for k, v in fold_metrics.items() if v}


def objective(trial, rna_df, meth_df, cnv_df, Y, class_names, class_weights, n_features):
    """
    Optuna objective function
    
    Optimizes for best F1-macro score on imbalanced multi-class classification
    """
    
    # ============ HYPERPARAMETERS TO OPTIMIZE ============
    
    # Architecture
    latent_dim = trial.suggest_categorical('latent_dim', [32, 48, 64])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [16, 32, 48, 64])
    
    # Learning rates
    lr_pre = trial.suggest_float('lr_pre', 1e-4, 5e-3, log=True)
    lr_fine = trial.suggest_float('lr_fine', 1e-5, 1e-3, log=True)
    
    # Regularization
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.325)
    dropout_encoder = trial.suggest_float('dropout_encoder', 0.0, 0.325)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Noise parameters
    noise_level = trial.suggest_float('noise_level', 0.0, 0.3)
    noise_type = trial.suggest_categorical('noise_type', ['gaussian', 'uniform', 'dropout'])
    
    # Focal Loss parameters
    focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
    focal_alpha_scale = trial.suggest_float('focal_alpha_scale', 0.5, 2.0)
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    
    # Robust early stopping parameters (relative gap = percentage-based)
    # E.g., max_rel_gap=0.5 means val_loss can be at most 50% higher than train_loss
    max_rel_gap_pre = trial.suggest_float('max_rel_gap_pre', 0.2, 1.0)  # 20%-100% overfitting tolerance
    max_rel_gap_fine = trial.suggest_float('max_rel_gap_fine', 0.15, 0.6)  # 15%-60% for finetuning
    overfit_patience = trial.suggest_int('overfit_patience', 5, 20)  # Epochs to tolerate overfitting
    no_improve_patience = trial.suggest_int('no_improve_patience', 15, 50)  # No improvement patience
    
    params = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'fusion_hidden_dim': fusion_hidden_dim,
        'lr_pre': lr_pre,
        'lr_fine': lr_fine,
        'dropout_rate': dropout_rate,
        'dropout_encoder': dropout_encoder,
        'weight_decay': weight_decay,
        'noise_level': noise_level,
        'noise_type': noise_type,
        'focal_gamma': focal_gamma,
        'focal_alpha_scale': focal_alpha_scale,
        'use_class_weights': use_class_weights,
        'max_rel_gap_pre': max_rel_gap_pre,
        'max_rel_gap_fine': max_rel_gap_fine,
        'overfit_patience': overfit_patience,
        'no_improve_patience': no_improve_patience,
        'ntxent_temperature': trial.suggest_float('ntxent_temperature', 0.1, 1.0),
        'ntxent_weight': trial.suggest_float('ntxent_weight', 0.1, 10.0),
    }
    
    # Run cross-validation
    try:
        metrics = run_cv_evaluation(
            params, n_features, rna_df, meth_df, cnv_df, Y, class_names,
            class_weights, verbose=False, trial=trial
        )
        
        # Primary objective: F1-macro (best for imbalanced classification)
        f1_macro = metrics['f1_macro']
        
        # Store additional metrics as user attributes
        trial.set_user_attr('accuracy', metrics['accuracy'])
        trial.set_user_attr('f1_micro', metrics['f1_micro'])
        trial.set_user_attr('precision', metrics['precision'])
        trial.set_user_attr('recall', metrics['recall'])
        trial.set_user_attr('silhouette', metrics['silhouette'])
        trial.set_user_attr('nmi', metrics['nmi'])
        trial.set_user_attr('ari', metrics['ari'])
        
        return f1_macro
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0


def run_optuna_optimization(rna_df, meth_df, cnv_df, Y, class_names, class_weights, 
                            n_features, n_trials=100):
    """
    Run Optuna hyperparameter optimization
    """
    
    # Create study with MedianPruner (prunes unpromising trials early)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize F1-macro
        study_name='PeromicsCMAE_optimization',
        pruner=pruner
    )
    
    # Create objective with partial application
    def objective_fn(trial):
        return objective(trial, rna_df, meth_df, cnv_df, Y, class_names, class_weights, n_features)
    
    print(f"\n{'='*60}")
    print(f"STARTING OPTUNA OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"Samples: {len(Y)}, Classes: {len(np.unique(Y))}, Features: {n_features}")
    print(f"Class distribution: {np.bincount(Y)}")
    print(f"Objective: Maximize F1-macro (best for imbalanced classification)")
    print(f"{'='*60}\n")
    
    # Run optimization
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    return study


def print_study_results(study):
    """Print comprehensive study results"""
    
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    # Filter completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned: {len(pruned_trials)}")
    
    if not completed_trials:
        print("No completed trials!")
        return None
    
    # Best trial
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print("BEST TRIAL")
    print(f"{'='*60}")
    print(f"Trial Number: {best_trial.number}")
    print(f"F1-Macro (objective): {best_trial.value:.4f}")
    
    # Additional metrics
    print(f"\nAdditional Metrics:")
    print(f"  Accuracy: {best_trial.user_attrs.get('accuracy', 'N/A'):.4f}")
    print(f"  F1-Micro: {best_trial.user_attrs.get('f1_micro', 'N/A'):.4f}")
    print(f"  Precision: {best_trial.user_attrs.get('precision', 'N/A'):.4f}")
    print(f"  Recall: {best_trial.user_attrs.get('recall', 'N/A'):.4f}")
    print(f"  Silhouette: {best_trial.user_attrs.get('silhouette', 'N/A'):.4f}")
    print(f"  NMI: {best_trial.user_attrs.get('nmi', 'N/A'):.4f}")
    print(f"  ARI: {best_trial.user_attrs.get('ari', 'N/A'):.4f}")
    
    # Best hyperparameters
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Export best params as dictionary
    print(f"\n{'='*60}")
    print("BEST_PARAMS (copy-paste ready)")
    print(f"{'='*60}")
    print("BEST_PARAMS = {")
    for key, value in best_trial.params.items():
        if isinstance(value, str):
            print(f"    '{key}': '{value}',")
        elif isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                print(f"    '{key}': {value:.6e},")
            else:
                print(f"    '{key}': {value:.6f},")
        else:
            print(f"    '{key}': {value},")
    print("}")
    
    return best_trial.params


def run_final_evaluation(best_params, rna_df, meth_df, cnv_df, Y, class_names, 
                         class_weights, n_features):
    """Run final detailed evaluation with best parameters and save model weights for all 5 folds"""
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION WITH BEST PARAMETERS (and model saving)")
    print(f"{'='*60}")
    
    # Create directory for saving models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_micro': [],
        'precision': [], 'recall': [], 'silhouette': [], 'nmi': [], 'ari': []
    }
    
    # Extract hyperparameters
    latent_dim = best_params.get('latent_dim', 64)
    hidden_dim = best_params.get('hidden_dim', 256)
    fusion_hidden_dim = best_params.get('fusion_hidden_dim', 64)
    lr_pre = best_params.get('lr_pre', 1e-3)
    lr_fine = best_params.get('lr_fine', 1e-3)
    dropout_rate = best_params.get('dropout_rate', 0.2)
    dropout_encoder = best_params.get('dropout_encoder', 0.0)
    noise_level = best_params.get('noise_level', 0.1)
    noise_type = best_params.get('noise_type', 'gaussian')
    focal_gamma = best_params.get('focal_gamma', 2.0)
    focal_alpha_scale = best_params.get('focal_alpha_scale', 1.0)
    use_class_weights = best_params.get('use_class_weights', True)
    weight_decay = best_params.get('weight_decay', 1e-4)
    max_rel_gap_pre = best_params.get('max_rel_gap_pre', 0.5)
    max_rel_gap_fine = best_params.get('max_rel_gap_fine', 0.3)
    overfit_patience = best_params.get('overfit_patience', 10)
    no_improve_patience = best_params.get('no_improve_patience', 30)
    ntxent_temp = best_params.get('ntxent_temperature', 0.5)
    ntxent_weight = best_params.get('ntxent_weight', 1.0)
    
    MAX_EPOCHS_PRE = 500
    MAX_EPOCHS_FINE = 1000
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # --- A. Data Processing (Leak-Safe) ---
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
        
        # --- C. Pretraining with LR Scheduler ---
        criterion_ntxent = NTXentLoss(temperature=ntxent_temp).to(DEVICE)
        best_pre_val_loss = float('inf')
        best_cmae_states = None
        overfit_counter_pre = 0
        no_improve_counter_pre = 0
        
        # LR Scheduler for pretraining
        scheduler_pre = ReduceLROnPlateau(opt_pre, mode='min', factor=0.5,
                                          patience=overfit_patience // 2, min_lr=1e-6)
        lr_reductions_pre = 0
        min_lr_reductions = 2
        prev_lr_pre = lr_pre
        
        for epoch in range(MAX_EPOCHS_PRE):
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            rec_r1, proj_r1, _, _ = cmae_r(t_tr_r, noise_level=noise_level, noise_type=noise_type)
            rec_m1, proj_m1, _, _ = cmae_m(t_tr_m, noise_level=noise_level, noise_type=noise_type)
            rec_c1, proj_c1, _, _ = cmae_c(t_tr_c, noise_level=noise_level, noise_type=noise_type)
            
            loss_recon_tr = F.mse_loss(rec_r1, t_tr_r) + F.mse_loss(rec_m1, t_tr_m) + F.mse_loss(rec_c1, t_tr_c)
            loss_nt = (criterion_ntxent(proj_r1, proj_m1) + 
                       criterion_ntxent(proj_r1, proj_c1) + 
                       criterion_ntxent(proj_m1, proj_c1)) / 3.0
            train_loss = loss_recon_tr + ntxent_weight * loss_nt
            
            opt_pre.zero_grad()
            train_loss.backward()
            opt_pre.step()
            
            cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
            with torch.no_grad():
                rec_r_val, proj_r_val, _, _ = cmae_r(t_val_r, noise_level=0.0)
                rec_m_val, proj_m_val, _, _ = cmae_m(t_val_m, noise_level=0.0)
                rec_c_val, proj_c_val, _, _ = cmae_c(t_val_c, noise_level=0.0)
                
                loss_recon_val = F.mse_loss(rec_r_val, t_val_r) + F.mse_loss(rec_m_val, t_val_m) + F.mse_loss(rec_c_val, t_val_c)
                loss_nt_val = (criterion_ntxent(proj_r_val, proj_m_val) + 
                               criterion_ntxent(proj_r_val, proj_c_val) + 
                               criterion_ntxent(proj_m_val, proj_c_val)) / 3.0
                val_loss = loss_recon_val + ntxent_weight * loss_nt_val
            
            train_loss_item = train_loss.item()
            val_loss_item = val_loss.item()
            
            # Step the scheduler
            scheduler_pre.step(val_loss_item)
            
            # Check if LR was reduced
            current_lr = opt_pre.param_groups[0]['lr']
            if current_lr < prev_lr_pre:
                lr_reductions_pre += 1
                prev_lr_pre = current_lr
            
            rel_gap = (val_loss_item - train_loss_item) / max(train_loss_item, 0.01)
            
            if val_loss_item < best_pre_val_loss:
                best_pre_val_loss = val_loss_item
                best_cmae_states = {
                    'cmae_r': cmae_r.state_dict(),
                    'cmae_m': cmae_m.state_dict(),
                    'cmae_c': cmae_c.state_dict()
                }
                no_improve_counter_pre = 0
            else:
                no_improve_counter_pre += 1
            
            if rel_gap > max_rel_gap_pre:
                overfit_counter_pre += 1
            else:
                overfit_counter_pre = 0
            
            if lr_reductions_pre >= min_lr_reductions:
                if overfit_counter_pre >= overfit_patience or no_improve_counter_pre >= no_improve_patience:
                    break
        
        if best_cmae_states:
            cmae_r.load_state_dict(best_cmae_states['cmae_r'])
            cmae_m.load_state_dict(best_cmae_states['cmae_m'])
            cmae_c.load_state_dict(best_cmae_states['cmae_c'])
        
        # --- D. Joint Training with LR Scheduler ---
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        
        fusion = GatedAttentionFusion(
            latent_dim, num_classes=len(np.unique(Y)), 
            dropout_rate=dropout_rate, hidden_dim=fusion_hidden_dim
        ).to(DEVICE)
        
        if use_class_weights:
            alpha = torch.FloatTensor(class_weights * focal_alpha_scale).to(DEVICE)
        else:
            alpha = None
        
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
        
        # Only train encoder + fusion (skip decoder/projector)
        opt_fine = optim.AdamW(
            list(cmae_r.encoder.parameters()) + list(cmae_m.encoder.parameters()) + list(cmae_c.encoder.parameters()) + list(fusion.parameters()),
            lr=lr_fine, weight_decay=weight_decay
        )
        
        # LR Scheduler for fine-tuning
        scheduler_fine = ReduceLROnPlateau(opt_fine, mode='min', factor=0.5,
                                           patience=overfit_patience // 2, min_lr=1e-7)
        lr_reductions_fine = 0
        prev_lr_fine = lr_fine
        
        best_fine_val_loss = float('inf')
        best_state = None
        best_encoder_states = None
        overfit_counter_fine = 0
        no_improve_counter_fine = 0
        
        for epoch in range(MAX_EPOCHS_FINE):
            cmae_r.train(); cmae_m.train(); cmae_c.train()
            fusion.train()
            
            # Use encode_only=True for efficiency
            _, _, zr_tr, _ = cmae_r(t_tr_r, noise_level=0.0, encode_only=True)
            _, _, zm_tr, _ = cmae_m(t_tr_m, noise_level=0.0, encode_only=True)
            _, _, zc_tr, _ = cmae_c(t_tr_c, noise_level=0.0, encode_only=True)
            
            logits, weights, _ = fusion(zr_tr, zm_tr, zc_tr, apply_dropout=True)
            train_loss_cls = criterion(logits, t_tr_y)
            opt_fine.zero_grad()
            train_loss_cls.backward()
            opt_fine.step()
            
            cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
            fusion.eval()
            with torch.no_grad():
                _, _, zr_val, _ = cmae_r(t_val_r, noise_level=0.0, encode_only=True)
                _, _, zm_val, _ = cmae_m(t_val_m, noise_level=0.0, encode_only=True)
                _, _, zc_val, _ = cmae_c(t_val_c, noise_level=0.0, encode_only=True)
                v_logits, _, v_fused = fusion(zr_val, zm_val, zc_val, apply_dropout=False)
                val_loss_cls = criterion(v_logits, t_val_y)
            
            train_loss_item = train_loss_cls.item()
            val_loss_item = val_loss_cls.item()
            
            # Step scheduler
            scheduler_fine.step(val_loss_item)
            
            # Check for LR reduction
            current_lr = opt_fine.param_groups[0]['lr']
            if current_lr < prev_lr_fine:
                lr_reductions_fine += 1
                prev_lr_fine = current_lr
            
            rel_gap = (val_loss_item - train_loss_item) / max(train_loss_item, 0.01)
            
            if val_loss_item < best_fine_val_loss:
                best_fine_val_loss = val_loss_item
                best_state = fusion.state_dict()
                best_encoder_states = {
                    'cmae_r': cmae_r.state_dict(),
                    'cmae_m': cmae_m.state_dict(),
                    'cmae_c': cmae_c.state_dict()
                }
                no_improve_counter_fine = 0
            else:
                no_improve_counter_fine += 1
            
            if rel_gap > max_rel_gap_fine:
                overfit_counter_fine += 1
            else:
                overfit_counter_fine = 0
            
            if lr_reductions_fine >= min_lr_reductions:
                if overfit_counter_fine >= overfit_patience or no_improve_counter_fine >= no_improve_patience:
                    break
        
        if best_state:
            fusion.load_state_dict(best_state)
        if best_encoder_states:
            cmae_r.load_state_dict(best_encoder_states['cmae_r'])
            cmae_m.load_state_dict(best_encoder_states['cmae_m'])
            cmae_c.load_state_dict(best_encoder_states['cmae_c'])
        
        # --- E. Save model weights for this fold ---
        fold_model_path = os.path.join(save_dir, f'fold_{fold + 1}_models.pt')
        torch.save({
            'fold': fold + 1,
            'cmae_r': cmae_r.state_dict(),
            'cmae_m': cmae_m.state_dict(),
            'cmae_c': cmae_c.state_dict(),
            'fusion': fusion.state_dict(),
            'hyperparams': best_params,
            'dims': dims,
            'feature_indices': {'rna': r_idx, 'meth': m_idx, 'cnv': c_idx}
        }, fold_model_path)
        print(f"    Model weights saved to: {fold_model_path}")
        
        # --- F. Evaluation ---
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        fusion.eval()
        with torch.no_grad():
            _, _, zr_val, _ = cmae_r(t_val_r, noise_level=0.0, encode_only=True)
            _, _, zm_val, _ = cmae_m(t_val_m, noise_level=0.0, encode_only=True)
            _, _, zc_val, _ = cmae_c(t_val_c, noise_level=0.0, encode_only=True)
            logits, _, z_fused_val = fusion(zr_val, zm_val, zc_val)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = t_val_y.cpu().numpy()
            z_emb = z_fused_val.cpu().numpy()
        
        acc = accuracy_score(targets, preds)
        f1_mac = f1_score(targets, preds, average='macro')
        f1_mic = f1_score(targets, preds, average='micro')
        prec = precision_score(targets, preds, average='macro', zero_division=0)
        rec = recall_score(targets, preds, average='macro', zero_division=0)
        
        sil = -1
        if len(np.unique(targets)) > 1:
            try:
                sil = silhouette_score(z_emb, targets)
            except:
                pass
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['silhouette'].append(sil)
        
        kmeans = KMeans(n_clusters=len(np.unique(Y)), random_state=42, n_init=10).fit(z_emb)
        nmi = normalized_mutual_info_score(targets, kmeans.labels_)
        ari = adjusted_rand_score(targets, kmeans.labels_)
        fold_metrics['nmi'].append(nmi)
        fold_metrics['ari'].append(ari)
        
        print(f"    Fold {fold+1}: Acc={acc:.3f}, F1-macro={f1_mac:.3f}, F1-micro={f1_mic:.3f}")
    
    # --- G. Save per-fold hyperparameters and scores to CSV ---
    # 1. Save hyperparameters for all folds (same hyperparams for all folds)
    hyperparams_rows = []
    for fold_num in range(1, 6):
        row = {'fold': fold_num}
        row.update(best_params)
        hyperparams_rows.append(row)
    hyperparams_df = pd.DataFrame(hyperparams_rows)
    hyperparams_df.to_csv(os.path.join(save_dir, 'fold_hyperparams.csv'), index=False)
    print(f"\nPer-fold hyperparameters saved to: {save_dir}/fold_hyperparams.csv")
    
    # 2. Save per-fold scores
    scores_rows = []
    for fold_num in range(5):
        scores_rows.append({
            'fold': fold_num + 1,
            'accuracy': fold_metrics['accuracy'][fold_num],
            'f1_macro': fold_metrics['f1_macro'][fold_num],
            'f1_micro': fold_metrics['f1_micro'][fold_num],
            'precision': fold_metrics['precision'][fold_num],
            'recall': fold_metrics['recall'][fold_num],
            'silhouette': fold_metrics['silhouette'][fold_num],
            'nmi': fold_metrics['nmi'][fold_num],
            'ari': fold_metrics['ari'][fold_num]
        })
    scores_df = pd.DataFrame(scores_rows)
    scores_df.to_csv(os.path.join(save_dir, 'fold_scores.csv'), index=False)
    print(f"Per-fold scores saved to: {save_dir}/fold_scores.csv")
    
    metrics = {k: np.mean(v) for k, v in fold_metrics.items() if v}
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nAll 5 fold models saved to: {save_dir}/")
    
    return metrics, fold_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PeromicsCMAE with Optuna Optimization')
    parser.add_argument('--n_trials', type=int, default=100, 
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--n_features', type=int, default=2000,
                        help='Number of features to select per omic (default: 2000)')
    parser.add_argument('--skip_final', action='store_true',
                        help='Skip final evaluation after optimization')
    args = parser.parse_args()
    
    # Load Data
    result = load_raw_aligned_data()
    if result is None:
        print("Failed to load data!")
        sys.exit(1)
        
    rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names, class_weights = result
    
    # Run Optuna optimization
    study = run_optuna_optimization(
        rna_df, meth_df, cnv_df, Y, class_names, class_weights,
        n_features=args.n_features, n_trials=args.n_trials
    )
    
    # Print results and get best params
    best_params = print_study_results(study)
    
    if best_params and not args.skip_final:
        # Run final evaluation with best params
        final_metrics, fold_metrics = run_final_evaluation(
            best_params, rna_df, meth_df, cnv_df, Y, class_names,
            class_weights, args.n_features
        )
        
        # Save best hyperparameters to JSON file
        import json
        hyperparams_to_save = {
            'n_features': args.n_features,
            'best_f1_macro': study.best_trial.value,
            **best_params
        }
        with open('best_hyperparams.json', 'w') as f:
            json.dump(hyperparams_to_save, f, indent=4)
        print(f"\nBest hyperparameters saved to 'best_hyperparams.json'")
        
        # Save aggregated best scores to CSV file
        scores_df = pd.DataFrame([{
            'n_features': args.n_features,
            'best_trial_number': study.best_trial.number,
            'f1_macro': final_metrics['f1_macro'],
            'f1_micro': final_metrics['f1_micro'],
            'accuracy': final_metrics['accuracy'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'silhouette': final_metrics['silhouette'],
            'nmi': final_metrics['nmi'],
            'ari': final_metrics['ari']
        }])
        scores_df.to_csv('best_scores.csv', index=False)
        print(f"Aggregated best scores saved to 'best_scores.csv'")
    
    print("\n>>> OPTIMIZATION COMPLETE!")
