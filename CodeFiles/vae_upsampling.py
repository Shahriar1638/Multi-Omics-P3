
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import gc

# Configuration
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

RAW_DATA_PATH = r"f:\BRACU\Thesis Thingy[T2510589]\P3 Coddy Stuffs\RawData"
OUTPUT_PATH = r"f:\BRACU\Thesis Thingy[T2510589]\P3 Coddy Stuffs\Upsampled_dataset"
SEED = 42

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_and_preprocess():
    print("Loading data...")
    # Load datasets
    expression = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.star_tpm.tsv'), sep='\t', index_col=0)
    methylation = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.methylation450.tsv'), sep='\t', index_col=0)
    # Correct handling for potential issues in gene-level load
    try:
        cnv = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.gene-level_absolute.tsv'), sep='\t', index_col=0)
    except:
        cnv = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.gene-level_absolute.tsv'), sep='\t', index_col=0, on_bad_lines='skip')
    
    try:
        clinical = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.clinical.tsv'), sep='\t', index_col=0)
    except:
        clinical = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.clinical.tsv'), sep='\t', index_col=0, on_bad_lines='skip')

    print("Preprocessing Expression...")
    # Log2(x+1) -> Variance Filter
    expr_log = np.log2(expression + 1)
    variances = expr_log.var(axis=1)
    expr_filtered = expr_log.loc[variances > 0.01]
    
    print("Preprocessing Methylation...")
    # Drop rows with too many NaNs first (efficiency)
    meth_drop = methylation.dropna(thresh=0.20 * methylation.shape[1], axis=0)
    # M-value: log2(beta / (1-beta))
    # Clip to avoid inf
    meth_clipped = meth_drop.clip(lower=0.001, upper=0.999)
    meth_m = np.log2(meth_clipped / (1 - meth_clipped))
    meth_vars = meth_m.var(axis=1)
    meth_filtered = meth_m.loc[meth_vars > 0.01]
    
    print("Preprocessing CNV...")
    # Filter -> Clip -> Log
    cnv_drop = cnv.loc[cnv.isnull().mean(axis=1) < 0.2]
    cnv_clipped = cnv_drop.clip(lower=0.05, upper=6)
    cnv_log = np.log2(cnv_clipped / 2)
    cnv_vars = cnv_log.var(axis=1)
    cnv_filtered = cnv_log.loc[cnv_vars > 0.01]

    print("Preprocessing Clinical...")
    subtype_col = 'primary_diagnosis.diagnoses'
    selected_subtypes = [
        'Leiomyosarcoma, NOS',
        'Dedifferentiated liposarcoma',
        'Undifferentiated sarcoma'
    ]
    
    # Filter clinical
    clinical = clinical[clinical[subtype_col].isin(selected_subtypes)]
    
    # Align samples
    common_samples = list(
        set(expr_filtered.columns) & 
        set(meth_filtered.columns) & 
        set(cnv_filtered.columns) & 
        set(clinical.index)
    )
    print(f"Common samples: {len(common_samples)}")
    
    expr_aligned = expr_filtered[common_samples]
    meth_aligned = meth_filtered[common_samples]
    cnv_aligned = cnv_filtered[common_samples]
    clinical_aligned = clinical.loc[common_samples]
    
    return expr_aligned, meth_aligned, cnv_aligned, clinical_aligned, selected_subtypes

# CVAE Model
class CVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=128):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder_h1 = nn.Linear(input_dim + num_classes, 1024)
        self.encoder_h2 = nn.Linear(1024, 512)
        self.encoder_mu = nn.Linear(512, latent_dim)
        self.encoder_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_h1 = nn.Linear(latent_dim + num_classes, 512)
        self.decoder_h2 = nn.Linear(512, 1024)
        self.decoder_out = nn.Linear(1024, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h1 = self.relu(self.encoder_h1(x_c))
        h2 = self.relu(self.encoder_h2(h1))
        return self.encoder_mu(h2), self.encoder_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h1 = self.relu(self.decoder_h1(z_c))
        h2 = self.relu(self.decoder_h2(h1))
        return self.decoder_out(h2)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    expr, meth, cnv, clinical, target_subtypes = load_and_preprocess()
    
    # Capture necessary metadata before deleting large objects
    subtype_col = 'primary_diagnosis.diagnoses'
    y_raw = clinical[subtype_col].values
    original_indices = list(clinical.index)
    
    # Store index/columns metadata for reconstruction later
    expr_index = expr.index
    meth_index = meth.index
    cnv_index = cnv.index
    n_expr = expr.shape[0]
    n_meth = meth.shape[0]
    n_cnv = cnv.shape[0]
    
    # Transpose to (samples, features)
    print("Preparing data matrices...")
    X_expr = expr.values.T
    X_meth = meth.values.T
    X_cnv = cnv.values.T
    
    # Free huge dataframes immediately
    print("Consolidating memory involved in loading...")
    del expr, meth, cnv, clinical
    gc.collect()
    
    # Concatenate
    X = np.hstack([X_expr, X_meth, X_cnv])
    
    # Free individual numpy parts
    del X_expr, X_meth, X_cnv
    gc.collect()
    
    # Impute NaNs for VAE training
    print(f"Imputing NaNs on matrix shape {X.shape}...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Prepare Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    # One-hot encode
    y_onehot = np.zeros((y_encoded.size, num_classes))
    y_onehot[np.arange(y_encoded.size), y_encoded] = 1
    
    # Convert to Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimization: Float32 is often enough and saves half RAM vs Float64
    dataset = TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_onehot))
    
    # Optimization: pin_memory=True for faster CPU->GPU transfer
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # Train CVAE
    input_dim = X_scaled.shape[1]
    print(f"Input Dimension: {input_dim}")
    torch.cuda.empty_cache()
    vae = CVAE(input_dim, num_classes).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    
    # Mixed Precision Scaler
    scaler_amp = torch.cuda.amp.GradScaler()
    
    print("Training CVAE with Mixed Precision...")
    vae.train()
    num_epochs = 200
    
    for epoch in range(num_epochs):
        train_loss = 0
        for data, labels in dataloader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # AMP Context
            with torch.cuda.amp.autocast():
                recon_batch, mu, logvar = vae(data, labels)
                loss = loss_function(recon_batch, data, mu, logvar)
            
            # Scaled Backward
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset):.4f}')
            
    # Upsampling Strategy
    upsample_targets = {
        'Dedifferentiated liposarcoma': 17,
        'Undifferentiated sarcoma': 20
    }
    
    counts = dict(zip(*np.unique(y_raw, return_counts=True)))
    print(f"Class Counts: {counts}")
    
    new_X = []
    new_y = []
    new_indices = []
    
    vae.eval()
    with torch.no_grad():
        for i, subtype in enumerate(le.classes_):
            if subtype in upsample_targets:
                needed = upsample_targets[subtype]
                print(f"Generating {needed} samples for {subtype}...")
                
                c_tensor = torch.zeros((needed, num_classes)).to(device)
                c_tensor[:, i] = 1
                
                z = torch.randn(needed, 128).to(device)
                
                # Decode
                generated = vae.decode(z, c_tensor).cpu().numpy()
                
                # Inverse scale
                generated_rescaled = scaler.inverse_transform(generated)
                
                new_X.append(generated_rescaled)
                new_y.extend([subtype] * needed)
                new_indices.extend([f"Synthetic_{subtype}_{j}" for j in range(needed)])
            else:
                print(f"Skipping upsampling for {subtype} (not in target list)")
            
    # Combine Data
    if new_X:
        print("Reconstructing datasets...")
        X_synthetic = np.vstack(new_X)
        X_final = np.vstack([X_imputed, X_synthetic]) 
        
        final_indices = original_indices + new_indices
        
        # Split columns back to Expression, Meth, CNV
        # X_final is (samples, features), we need transpose for saving: (features, samples)
        X_final_T = X_final.T
        
        expr_end = n_expr
        meth_end = n_expr + n_meth
        
        print("Saving Expression...")
        new_expr = pd.DataFrame(X_final_T[:expr_end, :], index=expr_index, columns=final_indices)
        new_expr.to_csv(os.path.join(OUTPUT_PATH, 'upsampled_expression.csv'))
        del new_expr # clear immediately
        gc.collect()

        print("Saving Methylation...")
        new_meth = pd.DataFrame(X_final_T[expr_end:meth_end, :], index=meth_index, columns=final_indices)
        new_meth.to_csv(os.path.join(OUTPUT_PATH, 'upsampled_methylation.csv'))
        del new_meth
        gc.collect()

        print("Saving CNV...")
        new_cnv = pd.DataFrame(X_final_T[meth_end:, :], index=cnv_index, columns=final_indices)
        new_cnv.to_csv(os.path.join(OUTPUT_PATH, 'upsampled_cnv.csv'))
        del new_cnv
        gc.collect()
        
        # New Clinical
        print("Saving Clinical...")
        # Load fresh clinical just to get columns/structure correct easily
        clinical_fresh = pd.read_csv(os.path.join(RAW_DATA_PATH, 'TCGA-SARC.clinical.tsv'), sep='\t', index_col=0, on_bad_lines='skip')
        
        new_clinical = clinical_fresh.reindex(final_indices)
        new_clinical.loc[new_indices, subtype_col] = new_y
        new_clinical.to_csv(os.path.join(OUTPUT_PATH, 'upsampled_clinical.csv'))
        
        print("Done!")
        print(f"Final sizes: {new_clinical[subtype_col].value_counts()}")
        
    else:
        print("No upsampling needed or performed.")

if __name__ == "__main__":
    main()
