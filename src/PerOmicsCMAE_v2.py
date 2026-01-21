# -*- coding: utf-8 -*-
import sys
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer 
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap

# --- OPTUNA IMPORT ---
try:
    import optuna
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# --- REPRODUCIBILITY SETUP ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

N_FEATURES = 1200  
SUBTYPES_OF_INTEREST = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]

# --- DATA LOADING ---
def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")
    pheno_path = "Data/phenotype_clean.csv"
    if not os.path.exists(pheno_path): raise FileNotFoundError(f"{pheno_path} not found.")

    pheno = pd.read_csv(pheno_path, index_col=0)
    col_name = 'primary_diagnosis.diagnoses'
    if col_name not in pheno.columns: return None

    mask = pheno[col_name].isin(SUBTYPES_OF_INTEREST)
    pheno = pheno[mask]
    
    def load_omic(path):
        if not os.path.exists(path): return None
        return pd.read_csv(path, index_col=0).T

    rna = load_omic("Data/expression_log.csv")
    meth = load_omic("Data/methylation_mvalues.csv")
    cnv = load_omic("Data/cnv_log.csv")

    if rna is None or meth is None or cnv is None: raise ValueError("Omics missing.")

    common = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    pheno = pheno.loc[common]
    rna = rna.loc[common]; meth = meth.loc[common]; cnv = cnv.loc[common]

    le = LabelEncoder()
    Y = le.fit_transform(pheno[col_name])
    
    return rna, meth, cnv, Y, le.classes_

# --- ARCHITECTURES ---
class MOCSS_CMAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        # 3-Layer Encoder/Decoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.encoder_specific = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder_shared = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, input_dim)
        )
        self.decoder_specific = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, input_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, noise_level=0.0):
        if self.training and noise_level > 0:
            x_corr = x + (torch.randn_like(x) * noise_level)
        else:
            x_corr = x
        z_c = self.encoder_shared(x_corr)
        z_s = self.encoder_specific(x_corr)
        rec_c = self.decoder_shared(z_c)
        rec_s = self.decoder_specific(z_s)
        proj_c = self.projector(z_c)
        return rec_c, rec_s, proj_c, z_c, z_s

class MOCSS_Gated_Classifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_cnv = nn.Linear(latent_dim, 1)
        
        # Input = Shared + Spec_R + Spec_M + Spec_C
        input_dim = latent_dim * 4 
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, z_shared_avg, zs_r, zs_m, zs_c):
        g_r = torch.sigmoid(self.gate_rna(zs_r))
        g_m = torch.sigmoid(self.gate_meth(zs_m))
        g_c = torch.sigmoid(self.gate_cnv(zs_c))
        zs_r_gated = zs_r * g_r
        zs_m_gated = zs_m * g_m
        zs_c_gated = zs_c * g_c
        z_fused = torch.cat([z_shared_avg, zs_r_gated, zs_m_gated, zs_c_gated], dim=1)
        return self.classifier(z_fused), z_fused

# --- LOSSES ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.4):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z_all = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z_all, z_all.t()) / self.temperature
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim.masked_fill_(mask, -9e15)
        return self.criterion(sim, labels) / (2 * batch_size)

def orthogonality_loss(zc, zs):
    zc_n = F.normalize(zc, dim=1)
    zs_n = F.normalize(zs, dim=1)
    interaction = torch.mm(zc_n.t(), zs_n) 
    return torch.norm(interaction, p='fro')**2 / (zc.shape[1]**2)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = ce_loss * at
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- EVALUATION LOOP (With Optuna Integration) ---
def run_cv_evaluation(params, n_features, rna_df, meth_df, cnv_df, Y, class_names):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track multiple metrics for composite score
    metrics = {
        'accuracy': [], 
        'f1_macro': [],
        'silhouette': []
    }
    
    # Extract Optuna Params
    latent_dim = params['latent_dim']
    lr_pre = params['lr_pre']
    lr_fine = params['lr_fine']
    dropout = params['dropout_rate']
    noise = params['noise_level']
    patience = params['patience']
    w_ortho = params['ortho_weight']
    w_cont = params['contrast_weight']
    temp = params['temperature']
    epochs_pre = params.get('epochs_pre', 100) # Tunable pretrain epochs
    
    num_classes = len(class_names)

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        # --- Preprocessing ---
        tr_rna_raw, val_rna_raw = rna_df.iloc[train_idx], rna_df.iloc[val_idx]
        tr_meth_raw, val_meth_raw = meth_df.iloc[train_idx], meth_df.iloc[val_idx]
        tr_cnv_raw, val_cnv_raw = cnv_df.iloc[train_idx], cnv_df.iloc[val_idx]

        def process_pipeline(tr_df, val_df, k):
            vars = np.nanvar(tr_df.values, axis=0)
            vars = np.nan_to_num(vars, nan=-1)
            k = min(k, tr_df.shape[1])
            idx = np.argpartition(vars, -k)[-k:]
            cols = tr_df.columns[idx]
            
            sc = StandardScaler()
            tr_sc = sc.fit_transform(tr_df[cols])
            val_sc = sc.transform(val_df[cols])
            
            imp = KNNImputer(n_neighbors=5)
            tr_fin = imp.fit_transform(tr_sc)
            val_fin = imp.transform(val_sc)
            return tr_fin, val_fin, tr_fin.shape[1]

        tr_rna, val_rna, dim_r = process_pipeline(tr_rna_raw, val_rna_raw, n_features)
        tr_meth, val_meth, dim_m = process_pipeline(tr_meth_raw, val_meth_raw, n_features)
        tr_cnv, val_cnv, dim_c = process_pipeline(tr_cnv_raw, val_cnv_raw, n_features)

        t_tr_r = torch.FloatTensor(tr_rna).to(DEVICE); t_val_r = torch.FloatTensor(val_rna).to(DEVICE)
        t_tr_m = torch.FloatTensor(tr_meth).to(DEVICE); t_val_m = torch.FloatTensor(val_meth).to(DEVICE)
        t_tr_c = torch.FloatTensor(tr_cnv).to(DEVICE); t_val_c = torch.FloatTensor(val_cnv).to(DEVICE)
        t_tr_y = torch.LongTensor(Y[train_idx]).to(DEVICE); t_val_y = torch.LongTensor(Y[val_idx]).to(DEVICE)

        # --- Models ---
        cmae_r = MOCSS_CMAE(dim_r, latent_dim).to(DEVICE)
        cmae_m = MOCSS_CMAE(dim_m, latent_dim).to(DEVICE)
        cmae_c = MOCSS_CMAE(dim_c, latent_dim).to(DEVICE)
        
        opt_pre = optim.AdamW(list(cmae_r.parameters()) + list(cmae_m.parameters()) + list(cmae_c.parameters()), lr=lr_pre)
        nt_xent = NTXentLoss(temperature=temp).to(DEVICE)

        # --- Pretraining ---
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        best_pre_loss = float('inf'); pat_counter = 0; best_states = None
        
        for epoch in range(epochs_pre): 
            rc_r, rs_r, p_r, zc_r, zs_r = cmae_r(t_tr_r, noise_level=noise)
            rc_m, rs_m, p_m, zc_m, zs_m = cmae_m(t_tr_m, noise_level=noise)
            rc_c, rs_c, p_c, zc_c, zs_c = cmae_c(t_tr_c, noise_level=noise)
            
            l_rec = (F.mse_loss(rc_r, t_tr_r) + F.mse_loss(rs_r, t_tr_r)) + \
                    (F.mse_loss(rc_m, t_tr_m) + F.mse_loss(rs_m, t_tr_m)) + \
                    (F.mse_loss(rc_c, t_tr_c) + F.mse_loss(rs_c, t_tr_c))
            l_cont = nt_xent(p_r, p_m) + nt_xent(p_r, p_c) + nt_xent(p_m, p_c)
            l_ortho = orthogonality_loss(zc_r, zs_r) + orthogonality_loss(zc_m, zs_m) + orthogonality_loss(zc_c, zs_c)
            
            loss = l_rec + (w_cont * l_cont) + (w_ortho * l_ortho)
            
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            
            if loss.item() < best_pre_loss:
                best_pre_loss = loss.item(); pat_counter = 0
                best_states = {'r': cmae_r.state_dict(), 'm': cmae_m.state_dict(), 'c': cmae_c.state_dict()}
            else:
                pat_counter += 1
                if pat_counter >= 15: break 
        
        if best_states:
            cmae_r.load_state_dict(best_states['r'])
            cmae_m.load_state_dict(best_states['m'])
            cmae_c.load_state_dict(best_states['c'])

        # --- Latents ---
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        with torch.no_grad():
            _, _, _, zc_r_tr, zs_r_tr = cmae_r(t_tr_r); _, _, _, zc_m_tr, zs_m_tr = cmae_m(t_tr_m); _, _, _, zc_c_tr, zs_c_tr = cmae_c(t_tr_c)
            _, _, _, zc_r_val, zs_r_val = cmae_r(t_val_r); _, _, _, zc_m_val, zs_m_val = cmae_m(t_val_m); _, _, _, zc_c_val, zs_c_val = cmae_c(t_val_c)
            
            z_shared_tr = (zc_r_tr + zc_m_tr + zc_c_tr) / 3.0
            z_shared_val = (zc_r_val + zc_m_val + zc_c_val) / 3.0

        # --- Fine-tuning ---
        classifier = MOCSS_Gated_Classifier(latent_dim, num_classes=num_classes, dropout_rate=dropout).to(DEVICE)
        
        # Class Balance
        cnts = np.bincount(Y[train_idx]); cnts = np.maximum(cnts, 1)
        wts = 1.0 / cnts; wts = wts / wts.sum() * num_classes
        criterion = FocalLoss(gamma=2.0, alpha=torch.FloatTensor(wts).to(DEVICE)).to(DEVICE)
        opt_fine = optim.AdamW(classifier.parameters(), lr=lr_fine)
        
        # Scheduler (FIX 5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_fine, mode='min', factor=0.5, patience=5)

        best_val_score = float('inf'); pat_counter = 0; best_state = None

        for epoch in range(params['epochs_fine']):
            classifier.train()
            logits, _ = classifier(z_shared_tr, zs_r_tr, zs_m_tr, zs_c_tr)
            loss_cls = criterion(logits, t_tr_y)
            opt_fine.zero_grad(); loss_cls.backward(); opt_fine.step()

            classifier.eval()
            with torch.no_grad():
                v_logits, _ = classifier(z_shared_val, zs_r_val, zs_m_val, zs_c_val)
                v_loss = criterion(v_logits, t_val_y)
            
            # Step Scheduler
            scheduler.step(v_loss)
            
            # Gap Logic Score
            score = v_loss.item() + 0.5 * max(0, v_loss.item() - loss_cls.item())

            if score < best_val_score:
                best_val_score = score
                best_state = classifier.state_dict()
                pat_counter = 0
            else:
                pat_counter += 1
                if pat_counter >= patience: break
        
        # Final Metrics for this Fold
        if best_state: classifier.load_state_dict(best_state)
        classifier.eval()
        with torch.no_grad():
            logits, z_emb_val = classifier(z_shared_val, zs_r_val, zs_m_val, zs_c_val)
            preds = logits.argmax(dim=1).cpu().numpy()
            targets = t_val_y.cpu().numpy()
            z_emb = z_emb_val.cpu().numpy()

        metrics['accuracy'].append(accuracy_score(targets, preds))
        metrics['f1_macro'].append(f1_score(targets, preds, average='macro'))
        try: 
            metrics['silhouette'].append(silhouette_score(z_emb, targets))
        except: 
            metrics['silhouette'].append(0)

    # --- FIX 2: COMPOSITE METRIC ---
    mean_f1 = np.mean(metrics['f1_macro'])
    mean_acc = np.mean(metrics['accuracy'])
    mean_sil = np.mean(metrics['silhouette'])
    
    # 50% F1, 30% Accuracy, 20% Clustering Quality
    composite_score = (0.5 * mean_f1) + (0.3 * mean_acc) + (0.2 * mean_sil)
    return composite_score

# --- OPTUNA OBJECTIVE ---
def objective(trial):
    # FIX 4: Error Handling to prevent study crash
    try:
        # Sample Hyperparameters
        params = {
            'latent_dim': trial.suggest_categorical('latent_dim', [32, 64, 128]),
            'lr_pre': trial.suggest_float('lr_pre', 1e-4, 1e-2, log=True),
            'lr_fine': trial.suggest_float('lr_fine', 1e-5, 1e-3, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'noise_level': trial.suggest_float('noise_level', 0.05, 0.2),
            'epochs_pre': trial.suggest_int('epochs_pre', 50, 200, step=50), # FIX 5: Tunable pretrain
            'epochs_fine': 500, 
            'patience': 10,
            'ortho_weight': trial.suggest_float('ortho_weight', 0.01, 1.0, log=True),
            'contrast_weight': trial.suggest_float('contrast_weight', 0.5, 10.0, log=True),
            'temperature': trial.suggest_float('temperature', 0.1, 1.0)
        }
        
        return run_cv_evaluation(params, N_FEATURES, rna_g, meth_g, cnv_g, Y_g, classes_g)
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf') # Return worst possible score on fail

if __name__ == "__main__":
    # 1. Load Data Globally
    print("Loading data for Optuna...")
    rna_g, meth_g, cnv_g, Y_g, classes_g = load_raw_aligned_data()
    
    # 2. Setup Study with Persistence (FIX 3)
    study = optuna.create_study(
        direction="maximize", 
        study_name="mocss_gated_optuna",
        storage="sqlite:///mocss_optuna.db", # Persist results
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("\n" + "="*60)
    print("STARTING HYPERPARAMETER OPTIMIZATION (Composite Score)")
    print("="*60)
    
    # 3. Optimize (Run 30 trials)
    study.optimize(objective, n_trials=30) 
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best Composite Score: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Save results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv")
    print("\nResults saved to optuna_results.csv")