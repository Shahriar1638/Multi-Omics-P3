
import sys
import os
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- BEST PARAMETERS (from optuna_best_params_supervised.csv) ---
BEST_PARAMS = {
    'latent_dim': 24,
    'hidden_dim': 128,
    'dropout_encoder': 0.3284628624179063,
    'dropout_rate': 0.35317785638083593,
    'lr_fine': 0.0006541890601631066,
    'weight_decay': 0.0007680199499155277,
    'noise_level': 0.29857678286552536,
    'focal_gamma': 4.234806377417181,
    'alpha_scale': 1.9402234817234842
}
# Enforce Funnel Constraint
BEST_PARAMS['fusion_hidden_dim'] = BEST_PARAMS['latent_dim'] 

class PerOmicCMAE(nn.Module):
    """Per-Omic Contrastive Masked Autoencoder (Encoder Only used for Supervised)"""

    def __init__(self, input_dim, latent_dim=64, hidden_dim=256, dropout_encoder=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_encoder),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, noise_level=0.0, noise_type='gaussian', encode_only=True):
        # Noise Injection / Masking during training for robustness
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
        return None, None, z, x_corrupted


class GatedAttentionFusion(nn.Module):
    """Gated Attention Fusion for multi-omics"""
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
            # Modality Dropout (regularization)
            if torch.rand(1).item() < self.drop_rate: z_rna = torch.zeros_like(z_rna)
            if torch.rand(1).item() < self.drop_rate: z_meth = torch.zeros_like(z_meth)
            if torch.rand(1).item() < self.drop_rate: z_clin = torch.zeros_like(z_clin)

        w_rna = torch.sigmoid(self.gate_rna(z_rna))
        w_meth = torch.sigmoid(self.gate_meth(z_meth))
        w_clin = torch.sigmoid(self.gate_clin(z_clin))

        z_fused = torch.cat([w_rna * z_rna, w_meth * z_meth, w_clin * z_clin], dim=1)

        return self.classifier(z_fused), torch.cat([w_rna, w_meth, w_clin], dim=1), z_fused


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

def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")
    pheno_path = "../Data/phenotype_clean.csv" # Adjusted path assuming run from src
    if not os.path.exists(pheno_path):
        # Fallback to current dir if run from root
        pheno_path = "Data/phenotype_clean.csv"
    
    if not os.path.exists(pheno_path):
         raise FileNotFoundError(f"{pheno_path} not found.")

    pheno = pd.read_csv(pheno_path, index_col=0)
    SUBTYPES_OF_INTEREST = [
        'Leiomyosarcoma, NOS', 'Dedifferentiated liposarcoma',
        'Undifferentiated sarcoma', 'Fibromyxosarcoma'
    ]
    col_name = 'primary_diagnosis.diagnoses'
    mask = pheno[col_name].isin(SUBTYPES_OF_INTEREST)
    pheno = pheno[mask]

    # Helper to find data relative to script location
    data_dir = os.path.dirname(pheno_path)

    def load_omic(filename, name):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        df = df.T  # samples x features
        return df

    rna = load_omic("expression_log.csv", "RNA")
    meth = load_omic("methylation_mvalues.csv", "Methylation")
    cnv = load_omic("cnv_log.csv", "CNV")

    common_samples = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    print(f"  Common Samples: {len(common_samples)}")

    pheno = pheno.loc[common_samples]
    rna = rna.loc[common_samples]
    meth = meth.loc[common_samples]
    cnv = cnv.loc[common_samples]

    le = LabelEncoder()
    Y = le.fit_transform(pheno[col_name])

    # Calculate Class Weights
    class_counts = np.bincount(Y)
    class_weights = len(Y) / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum()

    return rna, meth, cnv, Y, le.classes_, class_weights

def prepare_fold_data(train_df, val_df, max_features=5000):
    tr_vals = train_df.values
    val_vals = val_df.values

    # 1. Variance Filter (Fit on Train only)
    if tr_vals.shape[1] > max_features:
        variances = np.nanvar(tr_vals, axis=0)
        top_indices = np.argpartition(variances, -max_features)[-max_features:]
        tr_vals = tr_vals[:, top_indices]
        val_vals = val_vals[:, top_indices]

    # 2. KNN Imputation (Fit on Train)
    imputer = KNNImputer(n_neighbors=5)
    tr_imp = imputer.fit_transform(tr_vals)
    val_imp = imputer.transform(val_vals)

    # 3. Scaling (Fit on Train)
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr_imp)
    val_scaled = scaler.transform(val_imp)

    return (
        torch.FloatTensor(tr_scaled).to(DEVICE),
        torch.FloatTensor(val_scaled).to(DEVICE),
        tr_scaled.shape[1]
    )

def run_final_evaluation(rna_df, meth_df, cnv_df, Y, class_names, class_weights, params):
    print(f"\n{'='*40}")
    print("STARTING FINAL EVALUATION (Best Params)")
    print(f"{'='*40}")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(rna_df, Y)):
        print(f"\n--- Fold {fold+1} ---")

        t_r_tr, t_r_val, dim_r = prepare_fold_data(rna_df.iloc[train_idx], rna_df.iloc[val_idx])
        t_m_tr, t_m_val, dim_m = prepare_fold_data(meth_df.iloc[train_idx], meth_df.iloc[val_idx])
        t_c_tr, t_c_val, dim_c = prepare_fold_data(cnv_df.iloc[train_idx], cnv_df.iloc[val_idx])

        t_y_tr = torch.LongTensor(Y[train_idx]).to(DEVICE)
        t_y_val = torch.LongTensor(Y[val_idx]).to(DEVICE)

        encoder_r = PerOmicCMAE(dim_r, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        encoder_m = PerOmicCMAE(dim_m, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        encoder_c = PerOmicCMAE(dim_c, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        # Enforce Funnel: Fusion hidden dim = Latent dim
        fusion = GatedAttentionFusion(params['latent_dim'], len(class_names), params['dropout_rate'], params['fusion_hidden_dim']).to(DEVICE)

        optimizer = optim.AdamW(
            list(encoder_r.parameters()) + list(encoder_m.parameters()) +
            list(encoder_c.parameters()) + list(fusion.parameters()),
            lr=params['lr_fine'], weight_decay=params['weight_decay']
        )

        alpha = torch.FloatTensor(class_weights * params['alpha_scale']).to(DEVICE)
        criterion = FocalLoss(gamma=params['focal_gamma'], alpha=alpha)

        # Mode changed to 'max' to track F1-Macro
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_f1 = 0.0
        best_model_states = None
        patience = 20
        cur_pat = 0
        warmup_epochs = 10

        for epoch in range(800):
            # --- WARMUP LOGIC ---
            if epoch < warmup_epochs:
                warmup_lr = params['lr_fine'] * ((epoch + 1) / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            encoder_r.train(); encoder_m.train(); encoder_c.train(); fusion.train()
            _, _, zr, _ = encoder_r(t_r_tr, noise_level=params['noise_level'], noise_type='uniform')
            _, _, zm, _ = encoder_m(t_m_tr, noise_level=params['noise_level'], noise_type='uniform')
            _, _, zc, _ = encoder_c(t_c_tr, noise_level=params['noise_level'], noise_type='uniform')

            logits, _, _ = fusion(zr, zm, zc, apply_dropout=True)
            loss = criterion(logits, t_y_tr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- VALIDATION (Tracking F1) ---
            encoder_r.eval(); encoder_m.eval(); encoder_c.eval(); fusion.eval()
            with torch.no_grad():
                _, _, zr_v, _ = encoder_r(t_r_val, noise_level=0.0)
                _, _, zm_v, _ = encoder_m(t_m_val, noise_level=0.0)
                _, _, zc_v, _ = encoder_c(t_c_val, noise_level=0.0)
                logits_v, _, _ = fusion(zr_v, zm_v, zc_v)

                preds_v = logits_v.argmax(dim=1).cpu().numpy()
                targets_v = t_y_val.cpu().numpy()
                val_f1 = f1_score(targets_v, preds_v, average='macro')

            # Only step the plateau scheduler after warmup finishes
            if epoch >= warmup_epochs:
                scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                cur_pat = 0
                best_model_states = {
                    'enc_r': copy.deepcopy(encoder_r.state_dict()),
                    'enc_m': copy.deepcopy(encoder_m.state_dict()),
                    'enc_c': copy.deepcopy(encoder_c.state_dict()),
                    'fusion': copy.deepcopy(fusion.state_dict()),
                    'preds': preds_v,
                    'targets': targets_v
                }
            else:
                cur_pat += 1

            if cur_pat >= patience and epoch >= warmup_epochs:
                break

        # --- RESTORE BEST WEIGHTS ---
        # If training failed to improve, best_model_states might be None? 
        # With patience 20 and warmup 10, it's virtually impossible not to have at least one eval.
        if best_model_states is None:
             print("Warning: Model did not improve. Using last state.")
             final_preds = preds_v
             final_targets = targets_v
        else:
            encoder_r.load_state_dict(best_model_states['enc_r'])
            encoder_m.load_state_dict(best_model_states['enc_m'])
            encoder_c.load_state_dict(best_model_states['enc_c'])
            fusion.load_state_dict(best_model_states['fusion'])
            final_targets = best_model_states['targets']
            final_preds = best_model_states['preds']

        f1_macro = f1_score(final_targets, final_preds, average='macro')
        f1_micro = f1_score(final_targets, final_preds, average='micro')
        prec = precision_score(final_targets, final_preds, average='macro')
        rec = recall_score(final_targets, final_preds, average='macro')
        acc = accuracy_score(final_targets, final_preds)

        print(f"   Fold {fold+1} Result: F1-Macro={f1_macro:.4f}, Acc={acc:.4f}")
        
        # Classification Report for this fold
        print("\nClassification Report (Fold {}):".format(fold+1))
        print(classification_report(final_targets, final_preds, target_names=class_names))

        fold_metrics.append({
            'Fold': fold + 1,
            'F1_Macro': f1_macro,
            'F1_Micro': f1_micro,
            'Precision_Macro': prec,
            'Recall_Macro': rec,
            'Accuracy': acc
        })

    # Aggregate Metrics
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print(f"\n{'='*40}")
    print("FINAL AVERAGED RESULTS (5-Fold CV)")
    print(f"{'='*40}")
    print(f"F1-Macro:       {mean_metrics['F1_Macro']:.4f} (+/- {std_metrics['F1_Macro']:.4f})")
    print(f"F1-Micro:       {mean_metrics['F1_Micro']:.4f} (+/- {std_metrics['F1_Micro']:.4f})")
    print(f"Precision:      {mean_metrics['Precision_Macro']:.4f} (+/- {std_metrics['Precision_Macro']:.4f})")
    print(f"Recall:         {mean_metrics['Recall_Macro']:.4f} (+/- {std_metrics['Recall_Macro']:.4f})")
    print(f"Accuracy:       {mean_metrics['Accuracy']:.4f} (+/- {std_metrics['Accuracy']:.4f})")

    metrics_df.to_csv("GatedFusionCMLP_BestParams_Results.csv", index=False)
    print("\nDetailed metrics saved to 'GatedFusionCMLP_BestParams_Results.csv'")

if __name__ == "__main__":
    try:
        rna_df, meth_df, cnv_df, Y, class_names, class_weights = load_raw_aligned_data()
        run_final_evaluation(rna_df, meth_df, cnv_df, Y, class_names, class_weights, BEST_PARAMS)
    except Exception as e:
        print(f"An error occurred: {e}")
