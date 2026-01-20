# ==================================================================================
# 0. IMPORTS
# ==================================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, classification_report, silhouette_score
from sklearn.utils.class_weight import compute_class_weight

# ==================================================================================
# 1. CONFIGURATION
# ==================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CONFIG = {
    'latent_dim': 64,
    'dropout': 0.5,
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'epochs': 1000,
    'patience': 50,
    'warmup_epochs': 150,
    'n_features': 5000,
    'n_neighbors_impute': 5,
    'n_splits': 5,
    'proto_reg_weight': 0.01,
    'proto_sep_weight': 0.1,
    'proto_margin': 1.0
}

# ==================================================================================
# 2. PREPROCESSING (LEAKAGE SAFE)
# ==================================================================================
def fit_preprocess(train_df):
    variances = train_df.var()
    feats = variances.sort_values(ascending=False).head(CONFIG['n_features']).index
    train_df = train_df[feats]

    imputer = KNNImputer(n_neighbors=CONFIG['n_neighbors_impute'])
    train_filled = imputer.fit_transform(train_df)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)

    return train_scaled, feats, imputer, scaler


def apply_preprocess(df, feats, imputer, scaler):
    df = df[feats]
    df = imputer.transform(df)
    return scaler.transform(df)

# ==================================================================================
# 3. MODEL
# ==================================================================================
class PrototypicalHead(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, dim))

    def forward(self, z):
        p = F.normalize(self.prototypes, dim=1)
        return -torch.cdist(z, p)


class JointOmicsProtoAE(nn.Module):
    def __init__(self, dims, n_classes):
        super().__init__()

        self.enc_gene = nn.Sequential(
            nn.Linear(dims['gene'], 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['dropout'])
        )

        self.enc_meth = nn.Sequential(
            nn.Linear(dims['meth'], 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['dropout'])
        )

        self.enc_cnv = nn.Sequential(
            nn.Linear(dims['cnv'], 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['dropout'])
        )

        self.shared = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(512, CONFIG['latent_dim'])
        )

        self.proto = PrototypicalHead(CONFIG['latent_dim'], n_classes)

        self.dec_gene = nn.Linear(CONFIG['latent_dim'], dims['gene'])
        self.dec_meth = nn.Linear(CONFIG['latent_dim'], dims['meth'])
        self.dec_cnv  = nn.Linear(CONFIG['latent_dim'], dims['cnv'])

    def forward(self, g, m, c):
        hg = self.enc_gene(g)
        hm = self.enc_meth(m)
        hc = self.enc_cnv(c)

        z = F.normalize(self.shared(torch.cat([hg, hm, hc], dim=1)), dim=1)
        logits = self.proto(z)

        return (
            self.dec_gene(z),
            self.dec_meth(z),
            self.dec_cnv(z),
            logits,
            z
        )

# ==================================================================================
# 4. PROTOTYPE LOSSES (FIXED)
# ==================================================================================
def prototype_losses(proto):
    p = F.normalize(proto.prototypes, dim=1)

    norm_loss = torch.mean(torch.norm(p, dim=1))

    dist = torch.cdist(p, p)
    mask = ~torch.eye(dist.size(0), device=dist.device).bool()
    sep_loss = torch.mean(F.relu(CONFIG['proto_margin'] - dist[mask]))

    return norm_loss, sep_loss

# ==================================================================================
# 5. TRAINING WITH CORRECT EARLY STOPPING
# ==================================================================================
def train_model(model, train_data, train_y, val_data, val_y):
    weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_y.cpu()),
        y=train_y.cpu()
    )
    weights = torch.tensor(weights, device=DEVICE, dtype=torch.float)

    opt = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    best_loss = float('inf')
    patience_counter = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(CONFIG['epochs']):
        model.train()
        opt.zero_grad()

        rg, rm, rc, logits, _ = model(*train_data)

        recon = (
            F.mse_loss(rg, train_data[0]) +
            F.mse_loss(rm, train_data[1]) +
            F.mse_loss(rc, train_data[2])
        )

        cls_loss = F.cross_entropy(logits, train_y, weight=weights)
        proto_norm, proto_sep = prototype_losses(model.proto)

        if epoch < CONFIG['warmup_epochs']:
            loss = recon
        else:
            loss = (
                recon +
                5.0 * cls_loss +
                CONFIG['proto_reg_weight'] * proto_norm +
                CONFIG['proto_sep_weight'] * proto_sep
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                vrg, vrm, vrc, v_logits, _ = model(*val_data)

                if epoch < CONFIG['warmup_epochs']:
                    val_metric = (
                        F.mse_loss(vrg, val_data[0]) +
                        F.mse_loss(vrm, val_data[1]) +
                        F.mse_loss(vrc, val_data[2])
                    ).item()
                else:
                    val_metric = F.cross_entropy(v_logits, val_y).item()

                if epoch % 50 == 0:
                    tr_acc = (logits.argmax(1) == train_y).float().mean()
                    va_acc = (v_logits.argmax(1) == val_y).float().mean()
                    print(
                        f"Ep {epoch:03d} | "
                        f"Train Loss {loss.item():.4f} (Acc {tr_acc:.2f}) | "
                        f"Val Metric {val_metric:.4f} (Acc {va_acc:.2f})"
                    )

                if epoch > CONFIG['warmup_epochs']:
                    if val_metric < best_loss:
                        best_loss = val_metric
                        best_wts = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= CONFIG['patience']:
                            print(f">> Early stopping at epoch {epoch}")
                            break

    model.load_state_dict(best_wts)
    return model

# ==================================================================================
# 6. EVALUATION
# ==================================================================================
def evaluate(model, data, y, names):
    model.eval()
    with torch.no_grad():
        _, _, _, logits, z = model(*data)
        preds = logits.argmax(1).cpu().numpy()

    f1 = f1_score(y.cpu(), preds, average='macro')
    try:
        sil = silhouette_score(z.cpu(), y.cpu())
    except:
        sil = -1.0

    print(f"Macro-F1: {f1:.4f} | Silhouette: {sil:.4f}")
    print(classification_report(y.cpu(), preds, target_names=names, zero_division=0))
    return f1

# ==================================================================================
# 7. CROSS-VALIDATION
# ==================================================================================
def cross_validate(gene, meth, cnv, y, class_names):
    skf = StratifiedKFold(CONFIG['n_splits'], shuffle=True, random_state=SEED)
    scores = []

    for fold, (tr, va) in enumerate(skf.split(gene, y)):
        print(f"\n===== FOLD {fold + 1} =====")

        y_tr, y_va = y[tr], y[va]

        gtr, gf, gi, gs = fit_preprocess(gene.iloc[tr])
        mtr, mf, mi, ms = fit_preprocess(meth.iloc[tr])
        ctr, cf, ci, cs = fit_preprocess(cnv.iloc[tr])

        gva = apply_preprocess(gene.iloc[va], gf, gi, gs)
        mva = apply_preprocess(meth.iloc[va], mf, mi, ms)
        cva = apply_preprocess(cnv.iloc[va], cf, ci, cs)

        train_data = (
            torch.FloatTensor(gtr).to(DEVICE),
            torch.FloatTensor(mtr).to(DEVICE),
            torch.FloatTensor(ctr).to(DEVICE)
        )

        val_data = (
            torch.FloatTensor(gva).to(DEVICE),
            torch.FloatTensor(mva).to(DEVICE),
            torch.FloatTensor(cva).to(DEVICE)
        )

        model = JointOmicsProtoAE(
            {'gene': gtr.shape[1], 'meth': mtr.shape[1], 'cnv': ctr.shape[1]},
            len(class_names)
        ).to(DEVICE)

        model = train_model(
            model,
            train_data,
            torch.LongTensor(y_tr).to(DEVICE),
            val_data,
            torch.LongTensor(y_va).to(DEVICE)
        )

        scores.append(
            evaluate(model, val_data, torch.LongTensor(y_va).to(DEVICE), class_names)
        )

    print("\n===== FINAL RESULT =====")
    print(f"Macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# ==================================================================================
# 8. MAIN
# ==================================================================================
if __name__ == "__main__":
    G_PATH = "NewDatasets/processed_expression_4O.csv"
    M_PATH = "NewDatasets/processed_methylation_4O.csv"
    C_PATH = "NewDatasets/processed_cnv_4O.csv"
    L_PATH = "NewDatasets/processed_labels_3Omics_FXS_OG.csv"

    gene = pd.read_csv(G_PATH, index_col=0).T
    meth = pd.read_csv(M_PATH, index_col=0).T
    cnv  = pd.read_csv(C_PATH, index_col=0).T
    labels = pd.read_csv(L_PATH, index_col=0)

    common = gene.index.intersection(meth.index).intersection(cnv.index).intersection(labels.index)
    gene, meth, cnv = gene.loc[common], meth.loc[common], cnv.loc[common]

    class_names = sorted(labels.iloc[:, 0].unique())
    y = labels.loc[common].iloc[:, 0].map({c: i for i, c in enumerate(class_names)}).values

    print(f"Samples: {len(common)} | Classes: {len(class_names)}")
    cross_validate(gene, meth, cnv, y, class_names)
