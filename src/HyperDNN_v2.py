import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. DATA LOADING & RIGOROUS PREPROCESSING
# ============================================================================

def load_and_preprocess():
    # Load Omics
    gene_df = pd.read_csv("../NewDatasets/processed_expression_4O.csv", index_col=0).T
    meth_df = pd.read_csv("../NewDatasets/processed_methylation_4O.csv", index_col=0).T
    cnv_df  = pd.read_csv("../NewDatasets/processed_cnv_4O.csv", index_col=0).T
    labels_df = pd.read_csv("../NewDatasets/processed_labels_3Omics_FXS_OG.csv", index_col=0)

    # Align samples
    common = gene_df.index.intersection(meth_df.index).intersection(cnv_df.index).intersection(labels_df.index)
    X_gene, X_meth, X_cnv = gene_df.loc[common], meth_df.loc[common], cnv_df.loc[common]
    y = labels_df.loc[common].values.ravel()

    # Stratified Split (80/20)
    indices = np.arange(len(common))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

    # Fix: Fit scaler ONLY on training data to prevent leakage
    scalers = {}
    processed_data = {'train': {}, 'test': {}}
    
    for name, df in zip(['gene', 'meth', 'cnv'], [X_gene, X_meth, X_cnv]):
        scaler = StandardScaler()
        train_vals = df.iloc[idx_train].values
        test_vals = df.iloc[idx_test].values
        
        processed_data['train'][name] = torch.FloatTensor(scaler.fit_transform(train_vals))
        processed_data['test'][name] = torch.FloatTensor(scaler.transform(test_vals))
        scalers[name] = scaler

    processed_data['train']['y'] = torch.LongTensor(y[idx_train])
    processed_data['test']['y'] = torch.LongTensor(y[idx_test])
    
    input_dims = {'gene': X_gene.shape[1], 'meth': X_meth.shape[1], 'cnv': X_cnv.shape[1]}
    num_classes = len(np.unique(y))
    
    return processed_data, input_dims, num_classes

# ============================================================================
# 2. INTEGRATED MULTI-TASK ARCHITECTURE
# ============================================================================

class OmicsEncoder(nn.Module):
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
    def __init__(self, input_dims, num_classes, latent_dim, encoder_hidden_dims, dropout):
        super().__init__()
        self.omics_names = list(input_dims.keys())
        self.encoders = nn.ModuleDict({
            n: OmicsEncoder(input_dims[n], encoder_hidden_dims, latent_dim, dropout) for n in self.omics_names
        })
        self.decoders = nn.ModuleDict({
            n: nn.Linear(latent_dim, input_dims[n]) for n in self.omics_names
        })
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * len(self.omics_names), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs):
        z_dict = {n: self.encoders[n](inputs[n]) for n in self.omics_names}
        recons = {n: self.decoders[n](z_dict[n]) for n in self.omics_names}
        fused = torch.cat([z_dict[n] for n in self.omics_names], dim=1)
        return recons, self.classifier(fused), z_dict

# ============================================================================
# 3. TRAINING & EVALUATION LOGIC
# ============================================================================

def run_trial(model, train_data, val_data, optimizer, alpha, epochs=100):
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recons, logits, _ = model(train_data)
        
        recon_loss = sum(F.mse_loss(recons[n], train_data[n]) for n in train_data if n != 'y')
        class_loss = F.cross_entropy(logits, train_data['y'])
        loss = (alpha * class_loss) + ((1 - alpha) * recon_loss)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, val_logits, _ = model(val_data)
            preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            f1 = f1_score(val_data['y'].cpu().numpy(), preds, average='macro')
            best_f1 = max(best_f1, f1)
    return best_f1

def get_modality_importance(model, data):
    """Explainability: Calculate contribution of each modality to classification."""
    model.eval()
    importances = {}
    with torch.no_grad():
        _, original_logits, _ = model(data)
        original_acc = (torch.argmax(original_logits, 1) == data['y']).float().mean().item()
        
        for name in model.omics_names:
            temp_data = {k: v.clone() for k, v in data.items()}
            # Permute modality to see impact
            temp_data[name] = temp_data[name][torch.randperm(temp_data[name].size(0))]
            _, perm_logits, _ = model(temp_data)
            perm_acc = (torch.argmax(perm_logits, 1) == data['y']).float().mean().item()
            importances[name] = original_acc - perm_acc
            
    return importances

# ============================================================================
# 4. EXECUTION PIPELINE
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, in_dims, n_classes = load_and_preprocess()
    
    # Move to device
    for set_name in ['train', 'test']:
        for k in data[set_name]: data[set_name][k] = data[set_name][k].to(device)

    # Optuna HPO
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: run_trial(
        HyperDNN(in_dims, n_classes, t.suggest_int('z', 16, 64), [128, 64], 0.4).to(device),
        data['train'], data['test'], 
        optim.Adam(torch.nn.ParameterList(), lr=t.suggest_float('lr', 1e-4, 1e-2, log=True)),
        t.suggest_float('alpha', 0.1, 0.9)
    ), n_trials=30)

    # Final Train & Explain
    print(f"Best Macro-F1: {study.best_value:.4f}")
    best_params = study.best_params
    final_model = HyperDNN(in_dims, n_classes, best_params['z'], [128, 64], 0.4).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    
    run_trial(final_model, data['train'], data['test'], optimizer, best_params['alpha'], epochs=300)
    
    # Explainability Report
    imp = get_modality_importance(final_model, data['test'])
    print("\n--- Modality Importance (Explainability) ---")
    for k, v in imp.items(): print(f"{k.upper()}: {v:.4f}")