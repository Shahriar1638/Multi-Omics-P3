
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os
import copy

# Try importing CuPy for GPU mRMR
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not found. Using CPU fallback (slower).")

# Suppress warnings
warnings.filterwarnings("ignore")

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ==========================================
# 1. Helper Functions (GPU mRMR, etc.)
# ==========================================

def f_classif_gpu(X, y):
    """Compute ANOVA F-value on GPU."""
    n_samples, n_features = X.shape
    classes = cp.unique(y)
    n_classes = len(classes)
    
    mean_total = cp.mean(X, axis=0)
    var_total = cp.var(X, axis=0)
    ss_total = var_total * n_samples
    
    ss_between = cp.zeros(n_features, dtype=cp.float32)
    
    for c in classes:
        mask = (y == c)
        n_c = cp.sum(mask)
        mean_c = cp.mean(X[mask], axis=0)
        ss_between += n_c * (mean_c - mean_total)**2
        
    ss_within = ss_total - ss_between
    
    df_between = n_classes - 1
    df_within = n_samples - n_classes
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    f_stat = cp.divide(ms_between, ms_within)
    f_stat = cp.nan_to_num(f_stat, nan=0.0)
    return f_stat

def correlation_gpu(X, y):
    """Compute Pearson correlation on GPU."""
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    
    numerator = cp.dot(y_centered, X_centered)
    X_ss = cp.sum(X_centered**2, axis=0)
    y_ss = cp.sum(y_centered**2)
    
    denominator = cp.sqrt(X_ss * y_ss)
    corr = numerator / denominator
    return cp.abs(corr)

def mrmr_gpu_impl(X, y, K):
    """GPU accelerated mRMR."""
    X_gpu = cp.asarray(X, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.int32)
    
    n_samples, n_features = X_gpu.shape
    
    # 1. Relevance
    relevance = f_classif_gpu(X_gpu, y_gpu)
    
    selected_indices = []
    candidate_mask = cp.ones(n_features, dtype=bool)
    redundancy = cp.zeros(n_features, dtype=cp.float32)
    
    for k in range(K):
        if k == 0:
            scores = relevance
        else:
            scores = relevance - (redundancy / k)
            
        current_scores = scores.copy()
        current_scores[~candidate_mask] = -np.inf 
        
        best_idx = int(cp.argmax(current_scores))
        selected_indices.append(best_idx)
        candidate_mask[best_idx] = False
        
        if k < K - 1:
            last_selected_feature = X_gpu[:, best_idx]
            new_corrs = correlation_gpu(X_gpu, last_selected_feature)
            redundancy += new_corrs
            
    return selected_indices

def variance_filter(train_vals, val_vals, top_k):
    if train_vals.shape[1] <= top_k:
        return train_vals, val_vals, np.arange(train_vals.shape[1])
    vars = np.nanvar(train_vals, axis=0)
    top_idx = np.argpartition(vars, -top_k)[-top_k:]
    top_idx = np.sort(top_idx)
    return train_vals[:, top_idx], val_vals[:, top_idx], top_idx

def mrmr_filter(train_vals, val_vals, train_y, top_k):
    if train_vals.shape[1] <= top_k:
        return train_vals, val_vals, np.arange(train_vals.shape[1])
        
    if HAS_CUPY:
        try:
            selected_indices = mrmr_gpu_impl(train_vals, train_y, top_k)
            if hasattr(selected_indices, 'get'): selected_indices = selected_indices.get()
            selected_indices = list(selected_indices)
            return train_vals[:, selected_indices], val_vals[:, selected_indices], selected_indices
        except Exception as e:
            print(f"GPU mRMR Error: {e}")
            
    # Fallback to Variance Filter if GPU fails or not available
    print("Fallback to Variance Filter for Feature Selection")
    return variance_filter(train_vals, val_vals, top_k)

def load_data():
    pheno = pd.read_csv("Data/phenotype_clean.csv", index_col=0)
    # Filter Subtypes
    SUBTYPES = ['Leiomyosarcoma, NOS', 'Dedifferentiated liposarcoma', 'Undifferentiated sarcoma', 'Fibromyxosarcoma']
    pheno = pheno[pheno['primary_diagnosis.diagnoses'].isin(SUBTYPES)]
    
    def load_omic(path):
        if not os.path.exists(path): return None
        return pd.read_csv(path, index_col=0).T
        
    rna = load_omic("Data/expression_log.csv")
    meth = load_omic("Data/methylation_mvalues.csv")
    cnv = load_omic("Data/cnv_log.csv")
    
    common = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    print(f"Common Samples: {len(common)}")
    
    pheno, rna, meth, cnv = pheno.loc[common], rna.loc[common], meth.loc[common], cnv.loc[common]
    
    le = LabelEncoder()
    Y = le.fit_transform(pheno['primary_diagnosis.diagnoses'])
    
    return rna, meth, cnv, Y, le.classes_

def prepare_fold(t_idx, v_idx, rna, meth, cnv, Y):
    # Split
    tr_r, val_r = rna.iloc[t_idx].values, rna.iloc[v_idx].values
    tr_m, val_m = meth.iloc[t_idx].values, meth.iloc[v_idx].values
    tr_c, val_c = cnv.iloc[t_idx].values, cnv.iloc[v_idx].values
    y_tr = Y[t_idx]
    
    # 1. Variance Filter (Pre-filter)
    tr_r, val_r, _ = variance_filter(tr_r, val_r, 6000)
    tr_m, val_m, _ = variance_filter(tr_m, val_m, 8000)
    tr_c, val_c, _ = variance_filter(tr_c, val_c, 5000)
    
    # 2. Impute
    imp = KNNImputer(n_neighbors=12)
    tr_r, val_r = imp.fit_transform(tr_r), imp.transform(val_r)
    tr_m, val_m = imp.fit_transform(tr_m), imp.transform(val_m)
    tr_c, val_c = imp.fit_transform(tr_c), imp.transform(val_c)
    
    # 3. mRMR
    tr_r, val_r, _ = mrmr_filter(tr_r, val_r, y_tr, 600)
    tr_m, val_m, _ = mrmr_filter(tr_m, val_m, y_tr, 800)
    tr_c, val_c, _ = mrmr_filter(tr_c, val_c, y_tr, 400)
    
    # 4. Scale
    sc = StandardScaler()
    tr_r, val_r = sc.fit_transform(tr_r), sc.transform(val_r)
    tr_m, val_m = sc.fit_transform(tr_m), sc.transform(val_m)
    tr_c, val_c = sc.fit_transform(tr_c), sc.transform(val_c)
    
    return (
        torch.FloatTensor(tr_r).to(DEVICE), torch.FloatTensor(val_r).to(DEVICE),
        torch.FloatTensor(tr_m).to(DEVICE), torch.FloatTensor(val_m).to(DEVICE),
        torch.FloatTensor(tr_c).to(DEVICE), torch.FloatTensor(val_c).to(DEVICE),
        torch.LongTensor(y_tr).to(DEVICE), torch.LongTensor(Y[v_idx]).to(DEVICE)
    )

# ==========================================
# 2. Model Classes
# ==========================================

class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class PerOmicEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.noise = GaussianNoise(std=0.05)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        x = self.noise(x)
        return self.encoder(x)

class SimpleFusionNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.concat_dim = latent_dim * 3 
        
        self.proj_head = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim) 
        )
        
    def forward(self, zr, zm, zc):
        z_fused = torch.cat([zr, zm, zc], dim=1) 
        proj_feat = self.proj_head(z_fused)
        return z_fused, proj_feat



class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) 
        sim_matrix = torch.div(sim_matrix, self.temperature)
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        labels = labels.contiguous().view(-1, 1)
        mask_labels = torch.eq(labels, labels.T).float().to(features.device)
        mask_eye = torch.eye(labels.shape[0]).to(features.device)
        mask_labels = mask_labels - mask_eye 
        
        exp_sim = torch.exp(sim_matrix)
        denom = torch.sum(exp_sim * (1 - mask_eye), dim=1)
        log_prob = sim_matrix - torch.log(denom + 1e-6).unsqueeze(1)
        mean_log_prob_pos = (mask_labels * log_prob).sum(1) / (mask_labels.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        return loss

# ==========================================
# 3. Optuna Objective
# ==========================================

def objective(trial, data):
    rna, meth, cnv, Y, classes = data

    # Hyperparameters
    # 1. Hidden Dim (Must be > fusion_dim and > latent_dim). 
    # Since min fusion is 32, hidden must be > 32. So minimal hidden is 64.
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128])
    
    # 2. Conditional Sampling for Latent & Fusion
    # Use distinct parameter names to avoid "CategoricalDistribution does not support dynamic value space" error.
    if hidden_dim == 64:
        latent_dim = trial.suggest_categorical('latent_dim_h64', [16, 24, 32, 48])
        fusion_dim = trial.suggest_categorical('fusion_dim_h64', [32])
    else: # 128
        latent_dim = trial.suggest_categorical('latent_dim_h128', [16, 24, 32, 48, 64])
        fusion_dim = trial.suggest_categorical('fusion_dim_h128', [32, 64, 96])

    dropout = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    temperature = trial.suggest_categorical('temperature', [0.05, 0.07, 0.1, 0.2])
    
    # SVM Hyperparameter
    svm_c = trial.suggest_float('svm_c', 0.1, 10.0, log=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    f1_scores = []
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(rna, Y)):
        # Prep Data
        (tr_r, val_r, tr_m, val_m, tr_c, val_c, y_tr, y_val) = prepare_fold(t_idx, v_idx, rna, meth, cnv, Y)
        
        # Models
        dim_r, dim_m, dim_c = tr_r.shape[1], tr_m.shape[1], tr_c.shape[1]
        enc_r = PerOmicEncoder(dim_r, latent_dim, hidden_dim, dropout).to(DEVICE)
        enc_m = PerOmicEncoder(dim_m, latent_dim, hidden_dim, dropout).to(DEVICE)
        enc_c = PerOmicEncoder(dim_c, latent_dim, hidden_dim, dropout).to(DEVICE)
        
        # SimpleFusionNet no longer needs num_classes
        fusion = SimpleFusionNet(latent_dim, fusion_dim, dropout).to(DEVICE)
        
        # Optim
        params = list(enc_r.parameters()) + list(enc_m.parameters()) + list(enc_c.parameters()) + list(fusion.parameters())
        opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        
        con_crit = SupervisedContrastiveLoss(temperature=temperature)
        
        # Train Loop
        EPOCHS = 500
        patience = 20
        cur_patience = 0
        best_val_loss = float('inf')
        
        # State saving
        best_state = None
        
        for ep in range(EPOCHS):
            enc_r.train(); enc_m.train(); enc_c.train(); fusion.train()
            
            # Forward
            zr = enc_r(tr_r)
            zm = enc_m(tr_m)
            zc = enc_c(tr_c)
            # No logits, only embeddings and projection
            z_fused, proj = fusion(zr, zm, zc)
            
            # Loss: Only SupCon
            loss = con_crit(proj, y_tr)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Val
            enc_r.eval(); enc_m.eval(); enc_c.eval(); fusion.eval()
            with torch.no_grad():
                vzr = enc_r(val_r)
                vzm = enc_m(val_m)
                vzc = enc_c(val_c)
                vz_fused, vproj = fusion(vzr, vzm, vzc)
                
                # Val Loss: Only SupCon
                val_total = con_crit(vproj, y_val)
            
            scheduler.step(val_total)
            
            if val_total < best_val_loss:
                best_val_loss = val_total
                cur_patience = 0
                # Deepcopy best state
                best_state = {
                    'enc_r': copy.deepcopy(enc_r.state_dict()),
                    'enc_m': copy.deepcopy(enc_m.state_dict()),
                    'enc_c': copy.deepcopy(enc_c.state_dict()),
                    'fusion': copy.deepcopy(fusion.state_dict())
                }
            else:
                cur_patience += 1
                
            if cur_patience >= patience:
                break
                
        # Restore best state for evaluation
        if best_state is not None:
            enc_r.load_state_dict(best_state['enc_r'])
            enc_m.load_state_dict(best_state['enc_m'])
            enc_c.load_state_dict(best_state['enc_c'])
            fusion.load_state_dict(best_state['fusion'])
            
        # Eval SVM
        enc_r.eval(); enc_m.eval(); enc_c.eval(); fusion.eval()
        with torch.no_grad():
            zr = enc_r(tr_r); zm = enc_m(tr_m); zc = enc_c(tr_c)
            tr_emb, _ = fusion(zr, zm, zc)
            
            vzr = enc_r(val_r); vzm = enc_m(val_m); vzc = enc_c(val_c)
            val_emb, _ = fusion(vzr, vzm, vzc)
            
            X_train = tr_emb.cpu().numpy()
            X_val = val_emb.cpu().numpy()
            y_train = y_tr.cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            
        svm = SVC(kernel='rbf', C=svm_c, class_weight='balanced', probability=True, random_state=SEED)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_val)
        f1 = f1_score(y_val_np, preds, average='macro')
        f1_scores.append(f1)
        
        # Report intermediate result to optuna
        trial.report(f1, fold)
        
        if trial.should_prune():
           raise optuna.exceptions.TrialPruned()

    return np.mean(f1_scores)

if __name__ == "__main__":
    # Load data ONCE
    print("Loading data...")
    print("Current Working Directory:", os.getcwd())
    data = load_data()
    
    # Pass data to objective using lambda
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data), n_trials=50, show_progress_bar=True)
    
    print("\nBest Params:")
    print(study.best_params)
    print(f"Best F1: {study.best_value}")
    
    # Consolidate keys for CSV
    best_params = study.best_params.copy()
    
    # Normalize latent_dim keys
    if 'latent_dim_h64' in best_params:
        best_params['latent_dim'] = best_params.pop('latent_dim_h64')
    elif 'latent_dim_h128' in best_params:
        best_params['latent_dim'] = best_params.pop('latent_dim_h128')
        
    # Normalize fusion_dim keys
    if 'fusion_dim_h64' in best_params:
        best_params['fusion_dim'] = best_params.pop('fusion_dim_h64')
    elif 'fusion_dim_h128' in best_params:
        best_params['fusion_dim'] = best_params.pop('fusion_dim_h128')
    
    # Save to CSV
    df = pd.DataFrame([best_params])
    df['best_f1_macro'] = study.best_value
    df.to_csv("best_hybrid_params_svm.csv", index=False)
    print("Saved best params to 'best_hybrid_params_svm.csv'")
