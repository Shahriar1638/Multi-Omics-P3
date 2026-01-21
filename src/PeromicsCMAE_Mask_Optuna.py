# -*- coding: utf-8 -*-
"""
PeromicsCMAE with MASKING Mechanism - Optuna Hyperparameter Optimization
=========================================================================
This version uses MASKING (like MAE - Masked Autoencoder) instead of DAE noise.

Optimizes:
- Latent dimensions & architecture
- Focal Loss alpha (per-class weights) and gamma
- Masking parameters (mask_ratio, mask_strategy)
- Learning rates, dropout, epochs, patience

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
N_FEATURES = 1200

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


class PerOmicCMAE(nn.Module):
    """
    Per-Omic Contrastive Masked Autoencoder with configurable masking strategies
    
    Unlike DAE (which adds noise), this uses MASKING - setting features to zero
    and learning to reconstruct them (similar to BERT/MAE approach).
    """
    
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

    def forward(self, x, mask_ratio=0.0, mask_strategy='random'):
        """
        Forward pass with configurable masking
        
        mask_ratio: fraction of features to mask (0.0 to 1.0)
        mask_strategy: 
            - 'random': random uniform masking
            - 'block': mask contiguous blocks of features
            - 'structured': mask features based on variance (mask low variance first)
            - 'inverse_structured': mask high variance features first
        """
        if mask_ratio > 0 and self.training:
            batch_size, num_features = x.shape
            
            if mask_strategy == 'random':
                # Standard random masking (like original MAE)
                mask = (torch.rand_like(x) > mask_ratio).float()
                
            elif mask_strategy == 'block':
                # Block masking - mask contiguous regions
                mask = torch.ones_like(x)
                block_size = int(num_features * mask_ratio)
                for i in range(batch_size):
                    start_idx = torch.randint(0, num_features - block_size + 1, (1,)).item()
                    mask[i, start_idx:start_idx + block_size] = 0
                    
            elif mask_strategy == 'structured':
                # Mask based on variance (mask low variance features preferentially)
                feature_var = x.var(dim=0)
                sorted_indices = torch.argsort(feature_var)  # Low to high variance
                num_mask = int(num_features * mask_ratio)
                mask = torch.ones_like(x)
                mask[:, sorted_indices[:num_mask]] = 0
                
            elif mask_strategy == 'inverse_structured':
                # Mask high variance features (harder reconstruction)
                feature_var = x.var(dim=0)
                sorted_indices = torch.argsort(feature_var, descending=True)  # High to low
                num_mask = int(num_features * mask_ratio)
                mask = torch.ones_like(x)
                mask[:, sorted_indices[:num_mask]] = 0
                
            elif mask_strategy == 'hybrid':
                # Combination: 50% random, 50% structured
                random_mask = (torch.rand_like(x) > mask_ratio * 0.5).float()
                feature_var = x.var(dim=0)
                sorted_indices = torch.argsort(feature_var)
                num_mask = int(num_features * mask_ratio * 0.5)
                struct_mask = torch.ones_like(x)
                struct_mask[:, sorted_indices[:num_mask]] = 0
                mask = random_mask * struct_mask
                
            else:
                mask = (torch.rand_like(x) > mask_ratio).float()
            
            x_masked = x * mask
        else:
            mask = torch.ones_like(x)
            x_masked = x

        z = self.encoder(x_masked)
        rec = self.decoder(z)

        return rec, z, mask


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion with configurable architecture"""
    
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3, hidden_dim=64):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_cnv = nn.Linear(latent_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.drop_rate = dropout_rate

    def forward(self, z_rna, z_meth, z_cnv, apply_dropout=False):
        if apply_dropout and self.training:
            if torch.rand(1).item() < self.drop_rate:
                z_rna = torch.zeros_like(z_rna)
            if torch.rand(1).item() < self.drop_rate:
                z_meth = torch.zeros_like(z_meth)
            if torch.rand(1).item() < self.drop_rate:
                z_cnv = torch.zeros_like(z_cnv)

        w_rna = torch.sigmoid(self.gate_rna(z_rna))
        w_meth = torch.sigmoid(self.gate_meth(z_meth))
        w_cnv = torch.sigmoid(self.gate_cnv(z_cnv))

        z_fused = (w_rna * z_rna + w_meth * z_meth + w_cnv * z_cnv) / (w_rna + w_meth + w_cnv + 1e-8)
        return self.classifier(z_fused), torch.cat([w_rna, w_meth, w_cnv], dim=1), z_fused


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples
        alpha: Per-class weights. Can be:
               - None: no class weighting
               - float: single weight applied to all
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
                focal_loss = self.alpha * focal_loss
            elif isinstance(self.alpha, (torch.Tensor, list, np.ndarray)):
                if not isinstance(self.alpha, torch.Tensor):
                    self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Gather alpha for each sample based on target class
                alpha_t = self.alpha.gather(0, targets.view(-1))
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
    
    # Masking parameters (key difference from DAE version)
    mask_ratio = params.get('mask_ratio', 0.15)
    mask_strategy = params.get('mask_strategy', 'random')
    
    epochs_fine = params.get('epochs_fine', 500)
    patience = params.get('patience', 20)
    focal_gamma = params.get('focal_gamma', 2.0)
    focal_alpha_scale = params.get('focal_alpha_scale', 1.0)
    use_class_weights = params.get('use_class_weights', True)
    weight_decay = params.get('weight_decay', 1e-4)
    pre_patience = params.get('pre_patience', 15)
    epochs_pre = params.get('epochs_pre', 100)
    
    # Reconstruction loss type
    recon_loss_type = params.get('recon_loss_type', 'mse')  # 'mse' or 'masked_mse'

    if verbose:
        print(f"    Hyperparams: latent={latent_dim}, hidden={hidden_dim}, "
              f"lr_pre={lr_pre:.5f}, lr_fine={lr_fine:.5f}, "
              f"dropout={dropout_rate:.2f}, mask_ratio={mask_ratio:.3f} ({mask_strategy}), "
              f"focal_gamma={focal_gamma:.2f}, focal_alpha_scale={focal_alpha_scale:.2f}")

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

        # --- C. Pretraining with Early Stopping (Using Masking) ---
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        best_pre_loss = float('inf')
        pre_patience_counter = 0
        best_cmae_states = None

        for epoch in range(epochs_pre):
            # Forward pass with masking
            rec_r1, z_r, mask_r = cmae_r(t_tr_r, mask_ratio=mask_ratio, mask_strategy=mask_strategy)
            rec_m1, z_m, mask_m = cmae_m(t_tr_m, mask_ratio=mask_ratio, mask_strategy=mask_strategy)
            rec_c1, z_c, mask_c = cmae_c(t_tr_c, mask_ratio=mask_ratio, mask_strategy=mask_strategy)

            if recon_loss_type == 'masked_mse':
                # Only compute loss on masked (reconstructed) positions
                # This is more similar to original MAE objective
                inv_mask_r = 1 - mask_r
                inv_mask_m = 1 - mask_m
                inv_mask_c = 1 - mask_c
                
                loss_r = ((rec_r1 - t_tr_r) ** 2 * inv_mask_r).sum() / (inv_mask_r.sum() + 1e-8)
                loss_m = ((rec_m1 - t_tr_m) ** 2 * inv_mask_m).sum() / (inv_mask_m.sum() + 1e-8)
                loss_c = ((rec_c1 - t_tr_c) ** 2 * inv_mask_c).sum() / (inv_mask_c.sum() + 1e-8)
                loss = loss_r + loss_m + loss_c
            else:
                # Standard MSE on full reconstruction
                loss = F.mse_loss(rec_r1, t_tr_r) + F.mse_loss(rec_m1, t_tr_m) + F.mse_loss(rec_c1, t_tr_c)

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()

            current_loss = loss.item()

            if current_loss < best_pre_loss:
                best_pre_loss = current_loss
                pre_patience_counter = 0
                best_cmae_states = {
                    'cmae_r': cmae_r.state_dict(),
                    'cmae_m': cmae_m.state_dict(),
                    'cmae_c': cmae_c.state_dict()
                }
            else:
                pre_patience_counter += 1
                if pre_patience_counter >= pre_patience:
                    break

        # Load best pretrain states
        if best_cmae_states:
            cmae_r.load_state_dict(best_cmae_states['cmae_r'])
            cmae_m.load_state_dict(best_cmae_states['cmae_m'])
            cmae_c.load_state_dict(best_cmae_states['cmae_c'])

        # --- D. Fine-tuning ---
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        
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

        opt_fine = optim.AdamW(fusion.parameters(), lr=lr_fine, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        with torch.no_grad():
            # No masking during inference
            _, zr_tr, _ = cmae_r(t_tr_r, mask_ratio=0.0)
            _, zm_tr, _ = cmae_m(t_tr_m, mask_ratio=0.0)
            _, zc_tr, _ = cmae_c(t_tr_c, mask_ratio=0.0)
            _, zr_val, _ = cmae_r(t_val_r, mask_ratio=0.0)
            _, zm_val, _ = cmae_m(t_val_m, mask_ratio=0.0)
            _, zc_val, _ = cmae_c(t_val_c, mask_ratio=0.0)

        for epoch in range(epochs_fine):
            fusion.train()
            logits, weights, _ = fusion(zr_tr, zm_tr, zc_tr, apply_dropout=True)
            loss_cls = criterion(logits, t_tr_y)
            opt_fine.zero_grad()
            loss_cls.backward()
            opt_fine.step()

            fusion.eval()
            with torch.no_grad():
                v_logits, _, v_fused = fusion(zr_val, zm_val, zc_val, apply_dropout=False)
                v_loss = criterion(v_logits, t_val_y)

            tr_loss_item = loss_cls.item()
            val_loss_item = v_loss.item()
            gap = max(0, val_loss_item - tr_loss_item)
            current_score = val_loss_item + 0.5 * gap

            if current_score < best_val_loss:
                best_val_loss = current_score
                best_state = fusion.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            fusion.load_state_dict(best_state)
            
        fusion.eval()
        with torch.no_grad():
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
    Optuna objective function for MASKING-based model
    
    Optimizes for best F1-macro score on imbalanced multi-class classification
    """
    
    # ============ HYPERPARAMETERS TO OPTIMIZE ============
    
    # Architecture
    latent_dim = trial.suggest_categorical('latent_dim', [32, 48, 64, 96, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 384, 512])
    fusion_hidden_dim = trial.suggest_categorical('fusion_hidden_dim', [32, 48, 64, 96])
    
    # Learning rates
    lr_pre = trial.suggest_float('lr_pre', 1e-4, 5e-3, log=True)
    lr_fine = trial.suggest_float('lr_fine', 1e-5, 1e-3, log=True)
    
    # Regularization
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    dropout_encoder = trial.suggest_float('dropout_encoder', 0.0, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # ===== MASKING PARAMETERS (Key difference from DAE) =====
    mask_ratio = trial.suggest_float('mask_ratio', 0.05, 0.5)  # How much to mask
    mask_strategy = trial.suggest_categorical('mask_strategy', 
        ['random', 'block', 'structured', 'inverse_structured', 'hybrid'])
    recon_loss_type = trial.suggest_categorical('recon_loss_type', ['mse', 'masked_mse'])
    
    # Focal Loss parameters
    focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
    focal_alpha_scale = trial.suggest_float('focal_alpha_scale', 0.5, 2.0)
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    
    # Training dynamics
    epochs_fine = trial.suggest_int('epochs_fine', 200, 1000, step=100)
    epochs_pre = trial.suggest_int('epochs_pre', 50, 150, step=25)
    patience = trial.suggest_int('patience', 10, 40, step=5)
    pre_patience = trial.suggest_int('pre_patience', 10, 30, step=5)
    
    params = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'fusion_hidden_dim': fusion_hidden_dim,
        'lr_pre': lr_pre,
        'lr_fine': lr_fine,
        'dropout_rate': dropout_rate,
        'dropout_encoder': dropout_encoder,
        'weight_decay': weight_decay,
        'mask_ratio': mask_ratio,
        'mask_strategy': mask_strategy,
        'recon_loss_type': recon_loss_type,
        'focal_gamma': focal_gamma,
        'focal_alpha_scale': focal_alpha_scale,
        'use_class_weights': use_class_weights,
        'epochs_fine': epochs_fine,
        'epochs_pre': epochs_pre,
        'patience': patience,
        'pre_patience': pre_patience,
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
    Run Optuna hyperparameter optimization for MASKING model
    """
    
    # Create study with MedianPruner (prunes unpromising trials early)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize F1-macro
        study_name='PeromicsCMAE_Mask_optimization',
        pruner=pruner
    )
    
    # Create objective with partial application
    def objective_fn(trial):
        return objective(trial, rna_df, meth_df, cnv_df, Y, class_names, class_weights, n_features)
    
    print(f"\n{'='*60}")
    print(f"STARTING OPTUNA OPTIMIZATION - MASKING MODEL ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"Samples: {len(Y)}, Classes: {len(np.unique(Y))}, Features: {n_features}")
    print(f"Class distribution: {np.bincount(Y)}")
    print(f"Objective: Maximize F1-macro (best for imbalanced classification)")
    print(f"Method: MASKING (like MAE) - NOT noise injection (DAE)")
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
    print("OPTUNA OPTIMIZATION RESULTS (MASKING MODEL)")
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
    print("BEST HYPERPARAMETERS (MASKING MODEL)")
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
    """Run final detailed evaluation with best parameters"""
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION WITH BEST PARAMETERS (MASKING MODEL)")
    print(f"{'='*60}")
    
    metrics = run_cv_evaluation(
        best_params, n_features, rna_df, meth_df, cnv_df, Y, class_names,
        class_weights, verbose=True, trial=None
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PeromicsCMAE MASKING Model with Optuna Optimization')
    parser.add_argument('--n_trials', type=int, default=100, 
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--n_features', type=int, default=1200,
                        help='Number of features to select per omic (default: 1200)')
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
        final_metrics = run_final_evaluation(
            best_params, rna_df, meth_df, cnv_df, Y, class_names,
            class_weights, args.n_features
        )
        
        # Save results
        results_df = pd.DataFrame([{
            'n_features': args.n_features,
            **final_metrics,
            **{f'param_{k}': v for k, v in best_params.items()}
        }])
        results_df.to_csv('PerOmicsCMAE_Mask_optuna_best_results.csv', index=False)
        print(f"\nResults saved to 'PerOmicsCMAE_Mask_optuna_best_results.csv'")
    
    # Save study results
    try:
        study_df = study.trials_dataframe()
        study_df.to_csv('optuna_mask_study_history.csv', index=False)
        print(f"Study history saved to 'optuna_mask_study_history.csv'")
    except Exception as e:
        print(f"Could not save study history: {e}")
    
    print("\n>>> MASKING MODEL OPTIMIZATION COMPLETE!")
