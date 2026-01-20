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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import optuna

# ==========================================
# 0. CONFIGURATION & DEVICE
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

FEATURE_COUNTS = [500, 1000, 2000, 3000]
SUBTYPES_OF_INTEREST = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]
N_TRIALS = 20

def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")
    
    # 1. Load Phenotype/Labels
    pheno_path = "Data/phenotype_clean.csv"
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"{pheno_path} not found.")
    
    pheno = pd.read_csv(pheno_path, index_col=0)
    
    col_name = 'primary_diagnosis.diagnoses'
    if col_name not in pheno.columns:
        print(f"Warning: '{col_name}' not found.")
        return None
        
    mask = pheno[col_name].isin(SUBTYPES_OF_INTEREST)
    pheno = pheno[mask]
    print(f"  Phenotype Samples (filtered): {pheno.shape[0]}")

    # 2. Load Omics
    def load_omic(path, name):
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
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

    return rna, meth, cnv, Y, le.classes_

# 
# ==========================================
# 1. INTEGRATED ARCHITECTURE
# ==========================================

class OmicsEncoder(nn.Module):
    """Modular encoder per modality for transparency."""
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.net = nn.Sequential(*layers)
        self.latent = nn.Linear(prev, latent_dim)

    def forward(self, x):
        return self.latent(self.net(x))

class HyperDNN(nn.Module):
    """Supervised Multi-Task Omics Model."""
    def __init__(self, input_dims, num_classes, latent_dim, encoder_hidden_dims, dropout):
        super().__init__()
        self.omics_names = list(input_dims.keys())
        # Independent Encoders for Explainability
        self.encoders = nn.ModuleDict({
            n: OmicsEncoder(input_dims[n], encoder_hidden_dims, latent_dim, dropout) for n in self.omics_names
        })
        # Simple Linear Decoders for Reconstruction
        self.decoders = nn.ModuleDict({
            n: nn.Linear(latent_dim, input_dims[n]) for n in self.omics_names
        })
        # Integrated Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * len(self.omics_names), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs):
        z_dict = {n: self.encoders[n](inputs[n]) for n in self.omics_names}
        recons = {n: self.decoders[n](z_dict[n]) for n in self.omics_names}
        fused = torch.cat([z_dict[n] for n in self.omics_names], dim=1)
        return recons, self.classifier(fused), z_dict

# ==========================================
# 2. CROSS-VALIDATION PIPELINE
# ==========================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. CROSS-VALIDATION PIPELINE (Fixed)
# ==========================================

def run_cv_evaluation(params, n_features, rna_df, meth_df, cnv_df, Y, class_names, is_optuna=True):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = {'accuracy': [], 'f1_macro': [], 'f1_micro': [], 'precision': [], 'recall': []}
    
    # Hyperparams extraction
    latent_dim = params['latent_dim']
    # Removing unused params for standard HyperDNN (if they are in params, they are just ignored here)
    # hyper_hidden = params['hyper_hidden_dim'] 
    # embed_dim = params['embedding_dim']
    # rank = params['rank']
    
    enc_layers = params['n_encoder_layers']
    # Dynamic hidden dims
    enc_hidden = []
    current_dim = 256
    for _ in range(enc_layers):
        enc_hidden.append(current_dim)
        current_dim = current_dim // 2
    
    lr_ae = params['lr_ae']
    lr_clf = params['lr_clf']
    dropout = params['dropout']
    epochs_ae = 50 
    epochs_clf = 200

    num_classes_total = len(class_names)

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        # Data Splitting
        tr_r, val_r = rna_df.iloc[train_idx], rna_df.iloc[val_idx]
        tr_m, val_m = meth_df.iloc[train_idx], meth_df.iloc[val_idx]
        tr_c, val_c = cnv_df.iloc[train_idx], cnv_df.iloc[val_idx]
        
        # Imputation
        imp = SimpleImputer(strategy='median')
        tr_r_imp = imp.fit_transform(tr_r); val_r_imp = imp.transform(val_r)
        tr_m_imp = imp.fit_transform(tr_m); val_m_imp = imp.transform(val_m)
        tr_c_imp = imp.fit_transform(tr_c); val_c_imp = imp.transform(val_c)
        
        # Variance Filter
        def get_top_k(d, k):
            v = np.var(d, axis=0) # Variance calculated on TRAIN only
            return np.argpartition(v, -k)[-k:] if d.shape[1] > k else np.arange(d.shape[1])
            
        r_idx = get_top_k(tr_r_imp, n_features)
        m_idx = get_top_k(tr_m_imp, n_features)
        c_idx = get_top_k(tr_c_imp, n_features)
        
        tr_r_sel = tr_r_imp[:, r_idx]; val_r_sel = val_r_imp[:, r_idx]
        tr_m_sel = tr_m_imp[:, m_idx]; val_m_sel = val_m_imp[:, m_idx]
        tr_c_sel = tr_c_imp[:, c_idx]; val_c_sel = val_c_imp[:, c_idx]
        
        # Scaling
        sc = StandardScaler()
        tr_r_fin = sc.fit_transform(tr_r_sel); val_r_fin = sc.transform(val_r_sel)
        tr_m_fin = sc.fit_transform(tr_m_sel); val_m_fin = sc.transform(val_m_sel)
        tr_c_fin = sc.fit_transform(tr_c_sel); val_c_fin = sc.transform(val_c_sel)
        
        # Tensors
        inputs_tr = {
            'RNA': torch.FloatTensor(tr_r_fin).to(DEVICE),
            'Meth': torch.FloatTensor(tr_m_fin).to(DEVICE),
            'CNV': torch.FloatTensor(tr_c_fin).to(DEVICE)
        }
        inputs_val = {
            'RNA': torch.FloatTensor(val_r_fin).to(DEVICE),
            'Meth': torch.FloatTensor(val_m_fin).to(DEVICE),
            'CNV': torch.FloatTensor(val_c_fin).to(DEVICE)
        }
        y_tr = torch.LongTensor(Y[train_idx]).to(DEVICE)
        y_val = torch.LongTensor(Y[val_idx]).to(DEVICE)
        
        # --- VERIFICATION CHECKS ---
        if fold == 0:
            print(f"\n[Fold {fold}] Verification:")
            print(f"  Train/Val Split: {len(train_idx)}/{len(val_idx)}")
            print(f"  Class Dist Train: {np.bincount(Y[train_idx])}")
            print(f"  Class Dist Val:   {np.bincount(Y[val_idx])}")
            print(f"  Data Leakage Check (Mean of Scaled Train RNA): {tr_r_fin.mean():.4f} (Should be ~0)")
            print(f"  Data Leakage Check (Mean of Scaled Val RNA):   {val_r_fin.mean():.4f} (Should NOT be exactly 0)")
        # ---------------------------
        
        input_dims = {'RNA': tr_r_fin.shape[1], 'Meth': tr_m_fin.shape[1], 'CNV': tr_c_fin.shape[1]}
        
        # --- A. Train HyperDNN (Autoencoder) ---
        # Fixed instantiation to match class definition
        model_ae = HyperDNN(
            input_dims=input_dims,
            num_classes=num_classes_total, # Added missing arg
            latent_dim=latent_dim,
            encoder_hidden_dims=enc_hidden,
            dropout=dropout
            # Removed: hyper_hidden_dim, embedding_dim, rank (not in __init__)
        ).to(DEVICE)
        
        opt_ae = optim.AdamW(model_ae.parameters(), lr=lr_ae)
        crit_mse = nn.MSELoss()
        
        model_ae.train()
        for e in range(epochs_ae):
            opt_ae.zero_grad()
            recons, _, _ = model_ae(inputs_tr) # Correct unpacking: recons is first
            loss = 0
            for k in inputs_tr:
                loss += crit_mse(recons[k], inputs_tr[k])
            loss.backward()
            opt_ae.step()
            
        # --- B. Train Classifier (on Latents) ---
        model_ae.eval()
        with torch.no_grad():
            _, _, lat_tr = model_ae(inputs_tr) # Correct unpacking: z_dict is third
            _, _, lat_val = model_ae(inputs_val)
            
            # Fuse = Concat
            z_tr = torch.cat([lat_tr[k] for k in ['RNA', 'Meth', 'CNV']], dim=1)
            z_val = torch.cat([lat_val[k] for k in ['RNA', 'Meth', 'CNV']], dim=1)
            
        clf = SimpleClassifier(z_tr.shape[1], 128, num_classes_total, dropout).to(DEVICE)
        opt_clf = optim.AdamW(clf.parameters(), lr=lr_clf, weight_decay=params.get('weight_decay', 1e-4))
        
        # Calculate weights based on training labels only (to avoid leakage)
        unique_classes = np.unique(Y[train_idx])
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=unique_classes, 
            y=Y[train_idx]
        )
        
        weight_tensor = torch.ones(num_classes_total).to(DEVICE)
        
        for cls_idx, w in zip(unique_classes, weights):
            if cls_idx < num_classes_total:
                weight_tensor[cls_idx] = w
                
        crit_cls = nn.CrossEntropyLoss(weight=weight_tensor)
        
        best_acc = 0
        best_state = None
        
        for e in range(epochs_clf):
            clf.train()
            opt_clf.zero_grad()
            out = clf(z_tr)
            loss = crit_cls(out, y_tr)
            loss.backward()
            opt_clf.step()
            
            # Val (for early stop capability)
            clf.eval()
            with torch.no_grad():
                out_v = clf(z_val)
                acc_v = accuracy_score(y_val.cpu(), out_v.argmax(1).cpu())
                if acc_v > best_acc:
                    best_acc = acc_v
                    best_state = clf.state_dict()
        
        if best_state: clf.load_state_dict(best_state)
        
        # Tests
        clf.eval()
        with torch.no_grad():
            preds = clf(z_val).argmax(1).cpu().numpy()
            targets = y_val.cpu().numpy()
            
        fold_metrics['f1_macro'].append(f1_score(targets, preds, average='macro'))
        if not is_optuna:
            fold_metrics['f1_micro'].append(f1_score(targets, preds, average='micro'))
            fold_metrics['accuracy'].append(accuracy_score(targets, preds))
            fold_metrics['precision'].append(precision_score(targets, preds, average='macro', zero_division=0))
            fold_metrics['recall'].append(recall_score(targets, preds, average='macro', zero_division=0))
            
    if is_optuna:
        return np.mean(fold_metrics['f1_macro'])
    else:
        return {k: np.mean(v) for k, v in fold_metrics.items()}

# 
# ==========================================
# 4. OPTIMIZATION
# ==========================================
if __name__ == "__main__":
    if os.path.exists("Data/expression_log.csv"):
        rna_df, meth_df, cnv_df, Y, class_names = load_raw_aligned_data()
        
        param_file = "HyperDNN_best_params.txt"
        with open(param_file, 'w') as f:
            f.write("Features | F1_Macro | Params\n")
            
        all_final_results = []
        
        print("\n" + "="*50)
        print("STARTING OPTUNA OPTIMIZATION FOR HYPERDNN")
        print("="*50)
        
        for n_feat in FEATURE_COUNTS:
            print(f"\n>>> Feature Count: {n_feat}")
            
            def obj(trial):
                params = {
                    # Latent space: keep it tight for small N
                    'latent_dim': trial.suggest_categorical('latent_dim', [16, 32, 48]), 
                    
                    # HyperNetwork parameters: low rank = better generalization
                    'hyper_hidden_dim': trial.suggest_categorical('hyper_hidden_dim', [32, 64]),
                    'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 32]),
                    'rank': trial.suggest_categorical('rank', [4, 8, 16]),
                    
                    # Layers: Don't go too deep
                    'n_encoder_layers': trial.suggest_int('n_encoder_layers', 1, 2),
                    
                    # Learning Rates: Separate them to allow classifier focus
                    'lr_ae': trial.suggest_float('lr_ae', 1e-4, 1e-3, log=True),
                    'lr_clf': trial.suggest_float('lr_clf', 1e-3, 1e-2, log=True),
                    
                    # Regularization: Crucial for N=205
                    'dropout': trial.suggest_float('dropout', 0.3, 0.5),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
                }
                return run_cv_evaluation(params, n_feat, rna_df, meth_df, cnv_df, Y, class_names, is_optuna=True)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=N_TRIALS)
            
            print(f"  Best F1: {study.best_value:.4f}")
            with open(param_file, 'a') as f:
                f.write(f"{n_feat} | {study.best_value:.4f} | {study.best_params}\n")
                
            # Final Eval
            res = run_cv_evaluation(study.best_params, n_feat, rna_df, meth_df, cnv_df, Y, class_names, is_optuna=False)
            res['n_features'] = n_feat
            all_final_results.append(res)
            
        # Summary
        df_res = pd.DataFrame(all_final_results)
        if not df_res.empty:
            cols = ['n_features', 'f1_macro', 'f1_micro', 'precision', 'recall', 'accuracy']
            print("\n" + "="*60)
            print("FINAL RESULTS SUMMARY")
            print("="*60)
            print(df_res[cols].round(4).to_string(index=False))
            df_res.to_csv("HyperDNN_results.csv", index=False)
    else:
        print("Data files not found in Data/ directory.")