
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
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f">>> Running on: {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- BEST PARAMETERS ---
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
BEST_PARAMS['fusion_hidden_dim'] = BEST_PARAMS['latent_dim'] 

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

    def forward(self, x, noise_level=0.0, noise_type='gaussian', encode_only=True):
        if self.training and noise_level > 0:
            noise = (torch.rand_like(x) - 0.5) * 2 * noise_level # Uniform
            x_corrupted = x + noise
        else:
            x_corrupted = x
        z = self.encoder(x_corrupted)
        return None, None, z, x_corrupted


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
        return self.classifier(z_fused)


class ConcatFusion(nn.Module):
    """Simple Concatenation Fusion (No Gating)"""
    def __init__(self, latent_dim=64, num_classes=4, dropout_rate=0.3, hidden_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self.drop_rate = dropout_rate

    def forward(self, z_rna, z_meth, z_clin, apply_dropout=False):
        if apply_dropout and self.training:
            # Still keeping modality dropout for fair comparison
            if torch.rand(1).item() < self.drop_rate: z_rna = torch.zeros_like(z_rna)
            if torch.rand(1).item() < self.drop_rate: z_meth = torch.zeros_like(z_meth)
            if torch.rand(1).item() < self.drop_rate: z_clin = torch.zeros_like(z_clin)
            
        z_fused = torch.cat([z_rna, z_meth, z_clin], dim=1)
        return self.classifier(z_fused)

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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def load_data():
    pheno_path = "Data/phenotype_clean.csv"
    if not os.path.exists(pheno_path): pheno_path = "../Data/phenotype_clean.csv"
    if not os.path.exists(pheno_path): return None

    pheno = pd.read_csv(pheno_path, index_col=0)
    SUBTYPES = ['Leiomyosarcoma, NOS', 'Dedifferentiated liposarcoma',
                'Undifferentiated sarcoma', 'Fibromyxosarcoma']
    pheno = pheno[pheno['primary_diagnosis.diagnoses'].isin(SUBTYPES)]
    
    data_dir = os.path.dirname(pheno_path)
    def load(name):
        p = os.path.join(data_dir, name)
        return pd.read_csv(p, index_col=0).T if os.path.exists(p) else None

    rna = load("expression_log.csv")
    meth = load("methylation_mvalues.csv")
    cnv = load("cnv_log.csv")
    
    comm = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    pheno, rna, meth, cnv = pheno.loc[comm], rna.loc[comm], meth.loc[comm], cnv.loc[comm]
    
    le = LabelEncoder()
    Y = le.fit_transform(pheno['primary_diagnosis.diagnoses'])
    
    # Weights
    w = len(Y) / (len(np.unique(Y)) * np.bincount(Y))
    w = w / w.sum()
    
    return rna, meth, cnv, Y, le.classes_, w

def prepare_fold(tr_idx, val_idx, rna, meth, cnv):
    # Variance Filter + Impute + Scale
    def process(df, tidx, vidx, max_f):
        tr, val = df.values[tidx], df.values[vidx]
        if tr.shape[1] > max_f:
            idx = np.argpartition(np.nanvar(tr, axis=0), -max_f)[-max_f:]
            tr, val = tr[:, idx], val[:, idx]
        imp = KNNImputer(n_neighbors=12)
        tr = imp.fit_transform(tr)
        val = imp.transform(val)
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        val = sc.transform(val)
        return torch.FloatTensor(tr).to(DEVICE), torch.FloatTensor(val).to(DEVICE), tr.shape[1]

    r_tr, r_v, dr = process(rna, tr_idx, val_idx, 5000)
    m_tr, m_v, dm = process(meth, tr_idx, val_idx, 5000)
    c_tr, c_v, dc = process(cnv, tr_idx, val_idx, 5000)
    return r_tr, r_v, dr, m_tr, m_v, dm, c_tr, c_v, dc

def run_ablation(ablation_name, rna, meth, cnv, Y, classes, weights, params):
    print(f"\n--- Running Ablation: {ablation_name} ---")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f1s, accs = [], []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(rna, Y)):
        r_tr, r_v, dr, m_tr, m_v, dm, c_tr, c_v, dc = prepare_fold(tr_idx, val_idx, rna, meth, cnv)
        y_tr = torch.LongTensor(Y[tr_idx]).to(DEVICE)
        y_val = torch.LongTensor(Y[val_idx]).to(DEVICE)
        
        # Determine Architecture
        enc_r = PerOmicCMAE(dr, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        enc_m = PerOmicCMAE(dm, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        enc_c = PerOmicCMAE(dc, params['latent_dim'], params['hidden_dim'], params['dropout_encoder']).to(DEVICE)
        
        if ablation_name == 'No Gating (Concat)':
            fusion = ConcatFusion(params['latent_dim'], len(classes), params['dropout_rate'], params['fusion_hidden_dim']).to(DEVICE)
        else:
            fusion = GatedAttentionFusion(params['latent_dim'], len(classes), params['dropout_rate'], params['fusion_hidden_dim']).to(DEVICE)
            
        optimizer = optim.AdamW(list(enc_r.parameters()) + list(enc_m.parameters()) + list(enc_c.parameters()) + list(fusion.parameters()),
                               lr=params['lr_fine'], weight_decay=params['weight_decay'])
        
        if ablation_name == 'CrossEntropy Loss':
             # alpha_scale not used for CE usually, but we can pass weights if we want.
             # Strict ablation of Focal Loss meant using standard CE.
             cw = torch.FloatTensor(weights).to(DEVICE)
             criterion = nn.CrossEntropyLoss(weight=cw)
        else:
             alpha = torch.FloatTensor(weights * params['alpha_scale']).to(DEVICE)
             criterion = FocalLoss(gamma=params['focal_gamma'], alpha=alpha)
             
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_f1 = 0
        patience, cur_pat = 20, 0
        warmup = 10
        noise = 0.0 if ablation_name == 'No Noise' else params['noise_level']
        
        for ep in range(300): # Shorter epochs for ablation speed
            if ep < warmup:
                lr = params['lr_fine'] * ((ep + 1) / warmup)
                for g in optimizer.param_groups: g['lr'] = lr
                
            enc_r.train(); enc_m.train(); enc_c.train(); fusion.train()
            _, _, zr, _ = enc_r(r_tr, noise_level=noise, noise_type='uniform')
            _, _, zm, _ = enc_m(m_tr, noise_level=noise, noise_type='uniform')
            _, _, zc, _ = enc_c(c_tr, noise_level=noise, noise_type='uniform')
            
            logits = fusion(zr, zm, zc, apply_dropout=True)
            loss = criterion(logits, y_tr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            enc_r.eval(); enc_m.eval(); enc_c.eval(); fusion.eval()
            with torch.no_grad():
                _, _, zrv, _ = enc_r(r_v); _, _, zmv, _ = enc_m(m_v); _, _, zcv, _ = enc_c(c_v)
                lv = fusion(zrv, zmv, zcv)
                pred = lv.argmax(1).cpu().numpy()
                cur_f1 = f1_score(y_val.cpu().numpy(), pred, average='macro')
                
            if ep > warmup: scheduler.step(cur_f1)
            
            if cur_f1 > best_f1:
                best_f1 = cur_f1
                cur_pat = 0
            else:
                cur_pat += 1
                if cur_pat >= patience and ep >= warmup: break
        
        f1s.append(best_f1)
        
    mean, std = np.mean(f1s), np.std(f1s)
    print(f"Result {ablation_name}: F1-Macro = {mean:.4f} (+/- {std:.4f})")
    return mean, std

if __name__ == "__main__":
    if load_data() is None:
        print("Data not found.")
        sys.exit()
    rna, meth, cnv, Y, classes, w = load_data()
    
    results = []
    
    # 1. Baseline
    m, s = run_ablation('Baseline (Gated+Focal+Noise)', rna, meth, cnv, Y, classes, w, BEST_PARAMS)
    results.append({'Config': 'Baseline', 'F1_Mean': m, 'F1_Std': s})
    
    # 2. No Gating
    m, s = run_ablation('No Gating (Concat)', rna, meth, cnv, Y, classes, w, BEST_PARAMS)
    results.append({'Config': 'No Gating', 'F1_Mean': m, 'F1_Std': s})
    
    # 3. No Noise
    m, s = run_ablation('No Noise', rna, meth, cnv, Y, classes, w, BEST_PARAMS)
    results.append({'Config': 'No Noise', 'F1_Mean': m, 'F1_Std': s})
    
    # 4. Cross Entropy
    m, s = run_ablation('CrossEntropy Loss', rna, meth, cnv, Y, classes, w, BEST_PARAMS)
    results.append({'Config': 'CrossEntropy Loss', 'F1_Mean': m, 'F1_Std': s})
    
    df = pd.DataFrame(results)
    df.to_csv("Ablation_Results_Architecture.csv", index=False)
    print("\nAblation Results Saved.")
    print(df)
