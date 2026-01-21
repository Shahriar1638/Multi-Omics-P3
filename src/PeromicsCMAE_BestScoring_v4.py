# -*- coding: utf-8 -*-
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

# Try importing Lifelines for C-index
try:
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    # print("Warning: lifelines not found. C-index will be skipped.")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Best hyperparameters from Optuna optimization (sample: 2000)
BEST_PARAMS = {
    'latent_dim': 64,
    'lr_pre': 0.001,
    'lr_fine': 0.00011,
    'dropout_rate': 0.2,
    'mask_ratio': 0.1035,
    'epochs_fine': 2000,
    'patience': 15
}

N_FEATURES = 1200  # Feature count for which params were optimized

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
        df = df.T # samples x features
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

    # 5. Survival Data
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

    return rna, meth, cnv, Y, T, E, le.classes_

class PerOmicCMAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, mask_ratio=0.0):
        if mask_ratio > 0 and self.training:
            mask = (torch.rand_like(x) > mask_ratio).float()
            x_masked = x * mask
        else:
            mask = torch.ones_like(x)
            x_masked = x
        z = self.encoder(x_masked)
        return self.decoder(z), z, mask

class GatedAttentionFusion(nn.Module):
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.gate_rna = nn.Linear(latent_dim, 1)
        self.gate_meth = nn.Linear(latent_dim, 1)
        self.gate_clin = nn.Linear(latent_dim, 1) # reusing name 'clin' for 'cnv'
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
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

        z_fused = (w_rna * z_rna + w_meth * z_meth + w_clin * z_clin) / (w_rna + w_meth + w_clin + 1e-8)
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
        # (1-pt)^gamma * CE
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # If alpha is scalar/list, broadcasting might be needed or gathering
            # For simplicity, assuming no alpha or handled externally for now unless requested
            # A common way using weight in cross_entropy handles alpha part of it, 
            # but standard Focal Loss definition separates them.
            # Here we follow robust implementation:
            pass 

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss







def run_cv_evaluation(params, n_features, rna_df, meth_df, cnv_df, Y, class_names):
    """
    params: dict of hyperparameters
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_micro': [],
        'precision': [], 'recall': [], 'silhouette': [], 'nmi': [], 'ari': []
    }

    # Hyperparams
    latent_dim = params.get('latent_dim', 128)
    lr_pre = params.get('lr_pre', 1e-3)
    lr_fine = params.get('lr_fine', 1e-3)
    dropout_rate = params.get('dropout_rate', 0.2)
    mask_ratio = params.get('mask_ratio', 0.5)
    epochs_fine = params.get('epochs_fine', 2000)
    patience = params.get('patience', 15)

    print(f"    Params: {params}")


    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        # --- A. Data Lease Safe Processing ---
        tr_rna_raw, val_rna_raw = rna_df.iloc[train_idx], rna_df.iloc[val_idx]
        tr_meth_raw, val_meth_raw = meth_df.iloc[train_idx], meth_df.iloc[val_idx]
        tr_cnv_raw, val_cnv_raw = cnv_df.iloc[train_idx], cnv_df.iloc[val_idx]

        # Impute using SimpleImputer (median strategy - fast and handles large matrices)
        def simple_impute_data(train_data, val_data):
            """Apply SimpleImputer: fit on train, transform both train and val"""
            imputer = SimpleImputer(strategy='median')
            # Fit on training data only (leakage-safe)
            tr_imputed = imputer.fit_transform(train_data.values)
            val_imputed = imputer.transform(val_data.values)
            return tr_imputed, val_imputed
        
        tr_rna_imp, val_rna_imp = simple_impute_data(tr_rna_raw, val_rna_raw)
        tr_meth_imp, val_meth_imp = simple_impute_data(tr_meth_raw, val_meth_raw)
        tr_cnv_imp, val_cnv_imp = simple_impute_data(tr_cnv_raw, val_cnv_raw)

        # Variance Filter
        def get_top_k_indices(data, k):
            vars = np.var(data, axis=0)
            return np.argpartition(vars, -k)[-k:] if data.shape[1] > k else np.arange(data.shape[1])

        r_idx = get_top_k_indices(tr_rna_imp, n_features)
        m_idx = get_top_k_indices(tr_meth_imp, n_features)
        c_idx = get_top_k_indices(tr_cnv_imp, n_features)

        tr_rna_sel = tr_rna_imp[:, r_idx]; val_rna_sel = val_rna_imp[:, r_idx]
        tr_meth_sel = tr_meth_imp[:, m_idx]; val_meth_sel = val_meth_imp[:, m_idx]
        tr_cnv_sel = tr_cnv_imp[:, c_idx]; val_cnv_sel = val_cnv_imp[:, c_idx]

        # Scale
        sc_r = StandardScaler(); sc_m = StandardScaler(); sc_c = StandardScaler()
        tr_rna = sc_r.fit_transform(tr_rna_sel); val_rna = sc_r.transform(val_rna_sel)
        tr_meth = sc_m.fit_transform(tr_meth_sel); val_meth = sc_m.transform(val_meth_sel)
        tr_cnv = sc_c.fit_transform(tr_cnv_sel); val_cnv = sc_c.transform(val_cnv_sel)

        dims = (tr_rna.shape[1], tr_meth.shape[1], tr_cnv.shape[1])

        # Tensor
        t_tr_r = torch.FloatTensor(tr_rna).to(DEVICE)
        t_tr_m = torch.FloatTensor(tr_meth).to(DEVICE)
        t_tr_c = torch.FloatTensor(tr_cnv).to(DEVICE)
        t_tr_y = torch.LongTensor(Y[train_idx]).to(DEVICE)

        t_val_r = torch.FloatTensor(val_rna).to(DEVICE)
        t_val_m = torch.FloatTensor(val_meth).to(DEVICE)
        t_val_c = torch.FloatTensor(val_cnv).to(DEVICE)
        t_val_y = torch.LongTensor(Y[val_idx]).to(DEVICE)

        # --- B. Model Init ---
        cmae_r = PerOmicCMAE(dims[0], latent_dim).to(DEVICE)
        cmae_m = PerOmicCMAE(dims[1], latent_dim).to(DEVICE)
        cmae_c = PerOmicCMAE(dims[2], latent_dim).to(DEVICE)
        # loss_fn Removed: Using direct MSE sum
        opt_pre = optim.AdamW(list(cmae_r.parameters())+list(cmae_m.parameters())+list(cmae_c.parameters()), lr=lr_pre)

        # Pretraining with early stopping
        cmae_r.train(); cmae_m.train(); cmae_c.train()
        best_pre_loss = float('inf')
        pre_patience_counter = 0
        pre_patience = 10  # Early stopping patience for pretraining
        best_cmae_states = None
        
        # Memory Bank Removed
        
        print(f"\n  [Fold {fold+1}] PRETRAINING (max 100 epochs)")
        for epoch in range(100):
            # 1. Forward Pass
            rec_r1, _, _ = cmae_r(t_tr_r, mask_ratio=mask_ratio)
            rec_m1, _, _ = cmae_m(t_tr_m, mask_ratio=mask_ratio)
            rec_c1, _, _ = cmae_c(t_tr_c, mask_ratio=mask_ratio)
            
            # Simple sum of MSE losses
            loss = F.mse_loss(rec_r1, t_tr_r) + F.mse_loss(rec_m1, t_tr_m) + F.mse_loss(rec_c1, t_tr_c)
            
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            
            # Early stopping check
            current_loss = loss.item()
            
            # Print epoch progress every 10 epochs
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"    Pretrain Ep {epoch+1:03d} | Loss: {current_loss:.4f}")
            
            # Early stopping check
            current_loss = loss.item()
            
            # Print epoch progress
            status = "*" if current_loss < best_pre_loss else ""
            print(f"    Pretrain Ep {epoch+1:03d} | Loss: {current_loss:.4f} {status}")
            
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
                    print(f"    >> Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  [Fold {fold+1}] Pretraining complete. Best Loss: {best_pre_loss:.4f}")
        
        # Load best pretrain states
        if best_cmae_states:
            cmae_r.load_state_dict(best_cmae_states['cmae_r'])
            cmae_m.load_state_dict(best_cmae_states['cmae_m'])
            cmae_c.load_state_dict(best_cmae_states['cmae_c'])

        # Fine-tuning
        cmae_r.eval(); cmae_m.eval(); cmae_c.eval()
        fusion = GatedAttentionFusion(latent_dim, num_classes=4, dropout_rate=dropout_rate).to(DEVICE)
        
        # Use Focal Loss instead of Cross Entropy
        # We instantiate it here. You can adjust gamma if needed (default 2.0).
        criterion = FocalLoss(gamma=2.0).to(DEVICE)
        
        opt_fine = optim.AdamW(fusion.parameters(), lr=lr_fine)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        with torch.no_grad():
            _, zr_tr, _ = cmae_r(t_tr_r); _, zm_tr, _ = cmae_m(t_tr_m); _, zc_tr, _ = cmae_c(t_tr_c)
            _, zr_val, _ = cmae_r(t_val_r); _, zm_val, _ = cmae_m(t_val_m); _, zc_val, _ = cmae_c(t_val_c)

        print(f"\n  [Fold {fold+1}] FINE-TUNING (max {epochs_fine} epochs)")
        for epoch in range(epochs_fine):
            fusion.train()
            logits, weights, _ = fusion(zr_tr, zm_tr, zc_tr, apply_dropout=True)
            loss_cls = criterion(logits, t_tr_y)
            opt_fine.zero_grad(); loss_cls.backward(); opt_fine.step()

            fusion.eval()
            with torch.no_grad():
                v_logits, _, v_fused = fusion(zr_val, zm_val, zc_val, apply_dropout=False)
                v_loss = criterion(v_logits, t_val_y)
                
                # Calculate accuracies
                train_acc = (logits.argmax(1) == t_tr_y).float().mean().item()
                val_acc = (v_logits.argmax(1) == t_val_y).float().mean().item()
            
            # --- New Early Stopping Logic ---
            # Score minimizes val_loss but penalizes large gaps between train/val loss
            tr_loss_item = loss_cls.item()
            val_loss_item = v_loss.item()
            gap = max(0, val_loss_item - tr_loss_item) # only penalize if val > train
            
            # Score formula: val_loss + 0.5 * gap
            # This encourages low val loss AND low overfitting
            current_score = val_loss_item + 0.5 * gap
            
            # Print epoch progress every 10 epochs
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"    Finetune Ep {epoch+1:03d} | Train Loss: {tr_loss_item:.4f} (Acc: {train_acc:.2f}) | Val Loss: {val_loss_item:.4f} (Acc: {val_acc:.2f})")

            if current_score < best_val_loss:
                best_val_loss = current_score
                best_state = fusion.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"    >> Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  [Fold {fold+1}] Fine-tuning complete. Best Val Loss: {best_val_loss:.4f}")

        if best_state: fusion.load_state_dict(best_state)
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
            except: pass

        fold_metrics['f1_macro'].append(f1_mac)

        # Full metrics
        kmeans = KMeans(n_clusters=len(np.unique(Y)), random_state=42, n_init=10).fit(z_emb)
        nmi = normalized_mutual_info_score(targets, kmeans.labels_)
        ari = adjusted_rand_score(targets, kmeans.labels_)

        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['silhouette'].append(sil)
        fold_metrics['nmi'].append(nmi)
        fold_metrics['ari'].append(ari)

        if fold == 4: # Viz for last fold of best model
            try:
                reducer = umap.UMAP(random_state=42)
                z_umap = reducer.fit_transform(z_emb)
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=targets, cmap='viridis', s=50, alpha=0.8)
                plt.title(f'UMAP Projection (Features={n_features})\nAcc={acc:.3f}, F1={f1_mac:.3f}')
                plt.colorbar(scatter, ticks=range(len(class_names)), label='Class')
                plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
                plt.tight_layout()
                plt.savefig(f"cluster_viz_{n_features}_fold5.png")
                plt.close()
            except Exception as ex: print(f"Viz error: {ex}")

    return {k: np.mean(v) for k, v in fold_metrics.items() if v}



if __name__ == "__main__":
    # Load Data Once
    rna_df, meth_df, cnv_df, Y, T_surv, E_surv, class_names = load_raw_aligned_data()

    print("\n" + "="*60)
    print(f"RUNNING EVALUATION WITH BEST HYPERPARAMETERS (Features: {N_FEATURES})")
    print("="*60)
    print(f"Parameters: {BEST_PARAMS}")
    all_results = []
    
    print(f"\n{'='*60}")
    print(f">>> EVALUATING ...")
    print(f"{'='*60}")
    
    res = run_cv_evaluation(BEST_PARAMS, N_FEATURES, rna_df, meth_df, cnv_df, Y, class_names)
    res['n_features'] = N_FEATURES
    all_results.append(res)

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    df_res = pd.DataFrame(all_results)
    if not df_res.empty:
        cols = ['n_features', 'f1_macro', 'f1_micro', 'precision', 'recall', 'silhouette', 'nmi', 'ari', 'accuracy']
        cols = [c for c in cols if c in df_res.columns]
        print(df_res[cols].round(4).to_string(index=False))
        df_res.to_csv("PerOmicsCMAE_results.csv", index=False)
        print("\nFull results saved to 'PerOmicsCMAE_results.csv'")