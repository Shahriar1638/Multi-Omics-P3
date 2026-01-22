# -*- coding: utf-8 -*-
"""
PerOmicDAE with Boruta Feature Selection
=========================================
Uses Boruta to select best features from top 5000 variance-selected features.
Boruta feature counts: 300 / 500 / 800 (configurable)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Boruta import
try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except ImportError:
    HAS_BORUTA = False
    print("Warning: boruta not installed. Install with: pip install boruta")

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

# ================== CONFIGURATION ==================
# Boruta final feature count options: 300, 500, or 800
BORUTA_N_FEATURES = 500  # Change this to 300, 500, or 800
VARIANCE_PRESELECT = 5000  # Pre-select top 5000 by variance before Boruta

BEST_PARAMS = {
    'n_features': BORUTA_N_FEATURES,  # Will be set by Boruta
    'latent_dim': 64,
    'hidden_dim': 128,
    'fusion_hidden_dim': 64,
    'lr_pre': 0.0002146758585108322,
    'lr_fine': 0.00011360532119650048,
    'dropout_rate': 0.11015537480141437,
    'dropout_encoder': 0.22590162867072883,
    'weight_decay': 1.9336124732004638e-05,
    'noise_level': 0.2184491078143125,
    'noise_type': 'uniform',
    'focal_gamma': 4.050282882242696,
    'focal_alpha_scale': 1.8214490747770962,
    'use_class_weights': False,
    'max_rel_gap_pre': 0.33104531983955027,
    'max_rel_gap_fine': 0.29846272422979525,
    'overfit_patience': 10,
    'no_improve_patience': 46,
    'ntxent_temperature': 0.2989172299699971,
    'ntxent_weight': 7.766899511244319
}

SUBTYPES_OF_INTEREST = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]

print(f"Configuration:")
print(f"  Variance Pre-selection: {VARIANCE_PRESELECT} features")
print(f"  Boruta Final Features: {BORUTA_N_FEATURES} features per omic")


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


# ================== MODEL DEFINITIONS ==================
class NTXentLoss(nn.Module):
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


class GatedAttentionFusion(nn.Module):
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3, hidden_dim=64):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_clin = nn.Linear(latent_dim, 1) 
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
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

        z_fused = torch.cat([w_rna * z_rna, w_meth * z_meth, w_clin * z_clin], dim=1)
        return self.classifier(z_fused), torch.cat([w_rna, w_meth, w_clin], dim=1), z_fused


class FocalLoss(nn.Module):
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


# ================== BORUTA FEATURE SELECTION ==================
def boruta_feature_selection(X_train, y_train, n_features_to_select, omic_name="omic"):
    """
    Apply Boruta feature selection to select top features
    
    Args:
        X_train: Training data (samples x features)
        y_train: Training labels
        n_features_to_select: Target number of features
        omic_name: Name for logging
    
    Returns:
        selected_indices: Indices of selected features
    """
    if not HAS_BORUTA:
        print(f"  Boruta not available. Using variance-based selection for {omic_name}")
        vars = np.var(X_train, axis=0)
        return np.argpartition(vars, -n_features_to_select)[-n_features_to_select:]
    
    print(f"  Running Boruta for {omic_name} ({X_train.shape[1]} -> {n_features_to_select} features)...")
    
    # Random Forest estimator for Boruta
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    # Boruta selector
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=50,  # Max iterations
        random_state=42,
        verbose=0
    )
    
    # Fit Boruta
    boruta.fit(X_train, y_train)
    
    # Get selected features
    selected_mask = boruta.support_
    tentative_mask = boruta.support_weak_
    
    # Combine confirmed and tentative features
    all_selected = selected_mask | tentative_mask
    selected_indices = np.where(all_selected)[0]
    
    print(f"    Boruta selected: {len(selected_indices)} features (confirmed: {selected_mask.sum()}, tentative: {tentative_mask.sum()})")
    
    # If Boruta selected more than needed, use feature ranking to trim
    if len(selected_indices) > n_features_to_select:
        # Use ranking to select top n_features_to_select
        ranking = boruta.ranking_
        top_indices = np.argsort(ranking)[:n_features_to_select]
        selected_indices = top_indices
        print(f"    Trimmed to top {n_features_to_select} by ranking")
    
    # If Boruta selected fewer than needed, supplement with variance-based selection
    elif len(selected_indices) < n_features_to_select:
        remaining_needed = n_features_to_select - len(selected_indices)
        # Get indices not selected by Boruta
        unselected = np.setdiff1d(np.arange(X_train.shape[1]), selected_indices)
        # Select by variance from remaining
        vars = np.var(X_train[:, unselected], axis=0)
        additional = unselected[np.argpartition(vars, -remaining_needed)[-remaining_needed:]]
        selected_indices = np.concatenate([selected_indices, additional])
        print(f"    Supplemented with {remaining_needed} variance-selected features")
    
    return selected_indices


# ================== TRAINING AND EVALUATION ==================
def run_full_evaluation(rna_df, meth_df, cnv_df, Y, T, E, class_names, class_weights, params):
    """Run 5-fold cross-validation with Boruta feature selection"""
    
    n_boruta_features = params['n_features']
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_micro': [],
        'precision': [], 'recall': [], 'c_index': []
    }
    
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
    use_class_weights = params['use_class_weights']
    weight_decay = params['weight_decay']
    max_rel_gap_pre = params['max_rel_gap_pre']
    max_rel_gap_fine = params['max_rel_gap_fine']
    overfit_patience = params['overfit_patience']
    no_improve_patience = params['no_improve_patience']
    ntxent_temp = params['ntxent_temperature']
    ntxent_weight = params['ntxent_weight']
    
    MAX_EPOCHS_PRE = 700
    MAX_EPOCHS_FINE = 1200
    min_lr_reductions = 2

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'='*60}")
        
        # --- A. Data Processing with Boruta ---
        tr_rna_raw, val_rna_raw = rna_df.iloc[train_idx], rna_df.iloc[val_idx]
        tr_meth_raw, val_meth_raw = meth_df.iloc[train_idx], meth_df.iloc[val_idx]
        tr_cnv_raw, val_cnv_raw = cnv_df.iloc[train_idx], cnv_df.iloc[val_idx]
        y_train = Y[train_idx]

        def impute_and_preselect(train_data, val_data, n_preselect):
            """Impute and pre-select top features by variance"""
            imputer = SimpleImputer(strategy='median')
            tr_imp = imputer.fit_transform(train_data.values)
            val_imp = imputer.transform(val_data.values)
            
            # Pre-select by variance
            if tr_imp.shape[1] > n_preselect:
                vars = np.var(tr_imp, axis=0)
                idx = np.argpartition(vars, -n_preselect)[-n_preselect:]
                tr_imp = tr_imp[:, idx]
                val_imp = val_imp[:, idx]
            
            return tr_imp, val_imp

        # Step 1: Impute and pre-select top 5000 by variance
        print(f"\nStep 1: Variance pre-selection ({VARIANCE_PRESELECT} features)")
        tr_rna_pre, val_rna_pre = impute_and_preselect(tr_rna_raw, val_rna_raw, VARIANCE_PRESELECT)
        tr_meth_pre, val_meth_pre = impute_and_preselect(tr_meth_raw, val_meth_raw, VARIANCE_PRESELECT)
        tr_cnv_pre, val_cnv_pre = impute_and_preselect(tr_cnv_raw, val_cnv_raw, VARIANCE_PRESELECT)

        # Step 2: Apply Boruta to select final features
        print(f"\nStep 2: Boruta feature selection ({n_boruta_features} features)")
        r_idx = boruta_feature_selection(tr_rna_pre, y_train, n_boruta_features, "RNA")
        m_idx = boruta_feature_selection(tr_meth_pre, y_train, n_boruta_features, "Methylation")
        c_idx = boruta_feature_selection(tr_cnv_pre, y_train, n_boruta_features, "CNV")

        # Apply Boruta indices
        tr_rna_sel = tr_rna_pre[:, r_idx]; val_rna_sel = val_rna_pre[:, r_idx]
        tr_meth_sel = tr_meth_pre[:, m_idx]; val_meth_sel = val_meth_pre[:, m_idx]
        tr_cnv_sel = tr_cnv_pre[:, c_idx]; val_cnv_sel = val_cnv_pre[:, c_idx]

        # Step 3: Scale
        sc_r = StandardScaler(); sc_m = StandardScaler(); sc_c = StandardScaler()
        tr_rna = sc_r.fit_transform(tr_rna_sel); val_rna = sc_r.transform(val_rna_sel)
        tr_meth = sc_m.fit_transform(tr_meth_sel); val_meth = sc_m.transform(val_meth_sel)
        tr_cnv = sc_c.fit_transform(tr_cnv_sel); val_cnv = sc_c.transform(val_cnv_sel)

        dims = (tr_rna.shape[1], tr_meth.shape[1], tr_cnv.shape[1])
        print(f"\nFinal dimensions: RNA={dims[0]}, Meth={dims[1]}, CNV={dims[2]}")

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

        # --- C. Pretraining ---
        print(f"\nPretraining...")
        criterion_ntxent = NTXentLoss(temperature=ntxent_temp).to(DEVICE)
        best_pre_val_loss = float('inf')
        best_cmae_states = None
        overfit_counter_pre = 0
        no_improve_counter_pre = 0
        
        scheduler_pre = ReduceLROnPlateau(opt_pre, mode='min', factor=0.5, 
                                          patience=overfit_patience // 2, min_lr=1e-6)
        lr_reductions_pre = 0
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

            val_loss_item = val_loss.item()
            train_loss_item = train_loss.item()
            
            scheduler_pre.step(val_loss_item)
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

        # --- D. Fine-tuning ---
        print(f"Fine-tuning...")
        fusion = GatedAttentionFusion(
            latent_dim, num_classes=len(np.unique(Y)), 
            dropout_rate=dropout_rate, hidden_dim=fusion_hidden_dim
        ).to(DEVICE)

        if use_class_weights:
            alpha = torch.FloatTensor(class_weights * focal_alpha_scale).to(DEVICE)
        else:
            alpha = None
            
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)

        opt_fine = optim.AdamW(
            list(cmae_r.encoder.parameters()) + list(cmae_m.encoder.parameters()) + 
            list(cmae_c.encoder.parameters()) + list(fusion.parameters()),
            lr=lr_fine, weight_decay=weight_decay
        )

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
            
            scheduler_fine.step(val_loss_item)
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
            
        # --- E. Evaluation ---
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
        
        print(f"\nFold {fold+1} Results: Acc={acc:.3f}, F1-macro={f1_mac:.3f}, F1-micro={f1_mic:.3f}, "
              f"Precision={prec:.3f}, Recall={rec:.3f}")

    return fold_metrics, all_preds, all_probs


# ================== MAIN ==================
if __name__ == "__main__":
    # Load Data
    result = load_raw_aligned_data()
    if result is None:
        print("Failed to load data!")
        sys.exit(1)
        
    rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names, class_weights = result
    
    print(f"\nData loaded successfully!")
    print(f"  Total samples: {len(Y)}")
    print(f"  Number of classes: {len(class_names)}")
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"RUNNING EVALUATION WITH BORUTA FEATURE SELECTION")
    print(f"  Variance Pre-select: {VARIANCE_PRESELECT}")
    print(f"  Boruta Features: {BORUTA_N_FEATURES}")
    print(f"{'='*60}")
    
    fold_metrics, all_preds, all_probs = run_full_evaluation(
        rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names, class_weights, BEST_PARAMS
    )
    
    # Print Results
    print(f"\n{'='*60}")
    print("FINAL CLASSIFICATION RESULTS (5-Fold CV)")
    print(f"{'='*60}")
    
    mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items() if v}
    std_metrics = {k: np.std(v) for k, v in fold_metrics.items() if v}
    
    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*35}")
    for key in ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'c_index']:
        if key in mean_metrics and all(x != -1 for x in fold_metrics[key]):
            print(f"  {key:<15} {mean_metrics[key]:>10.4f} {std_metrics[key]:>10.4f}")
    
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(Y, all_preds, target_names=class_names, zero_division=0))
    
    print("\n>>> EVALUATION COMPLETE!")
