import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Based on Table 2 and experimental settings in the paper [cite: 157, 171, 181]
CONFIG = {
    'batch_size': 128,           # Optimal batch size [cite: 182]
    'learning_rate': 1e-3,       # Standard Adam LR
    'epochs': 100,               # Sufficient for convergence
    'latent_dim': 128,           # Representation dimension d_z [cite: 171, 217]
    'temp': 0.4,                 # Temperature parameter tau [cite: 171, 225]
    'num_clusters': 3,           # Example: LUAD has 3 subtypes [cite: 126]
    'input_dims': [1000, 500, 500], # Simulated dimensions for mRNA, miRNA, Methylation
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==========================================
# 2. DATA SIMULATION
# ==========================================
# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================

def load_raw_aligned_data():
    print(f"\n>>> LOADING RAW ALIGNED DATA")
    # Path relative to src/
    pheno_path = "../Data/phenotype_clean.csv" 
    if not os.path.exists(pheno_path):
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

    # Intersection
    common_samples = pheno.index.intersection(rna.index).intersection(meth.index).intersection(cnv.index)
    print(f"  Common Samples: {len(common_samples)}")

    pheno = pheno.loc[common_samples]
    rna = rna.loc[common_samples]
    meth = meth.loc[common_samples]
    cnv = cnv.loc[common_samples]

    le = LabelEncoder()
    Y = le.fit_transform(pheno[col_name])

    return rna, meth, cnv, Y, le.classes_

def preprocess_full_data(rna_df, meth_df, cnv_df, max_features=5000):
    datasets = [rna_df, meth_df, cnv_df]
    processed_tensors = []
    
    for df in datasets:
        vals = df.values
        
        # 1. Variance Filter
        if vals.shape[1] > max_features:
            variances = np.nanvar(vals, axis=0)
            top_indices = np.argpartition(variances, -max_features)[-max_features:]
            vals = vals[:, top_indices]
        
        # 2. KNN Imputation
        imputer = KNNImputer(n_neighbors=5)
        vals_imp = imputer.fit_transform(vals)
        
        # 3. Scaling
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(vals_imp)
        
        processed_tensors.append(torch.FloatTensor(vals_scaled))
        
    return processed_tensors[0], processed_tensors[1], processed_tensors[2]

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

class Encoder(nn.Module):
    """
    Standard Encoder for both Shared and Specific pathways.
    """
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # Using a simple MLP structure as implied by "deep neural networks" [cite: 234]
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        z = self.layer3(x) 
        return z

class Decoder(nn.Module):
    """
    Standard Decoder to reconstruct original data.
    """
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(latent_dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, output_dim)
    
    def forward(self, z):
        z = F.relu(self.layer1(z))
        z = F.relu(self.layer2(z))
        # Sigmoid because data is normalized to [0, 1] [cite: 129]
        x_rec = torch.sigmoid(self.layer3(z)) 
        return x_rec

class MOCSS_OmicsNet(nn.Module):
    """
    Network for a SINGLE Omics view.
    Contains 2 Autoencoders: one for Shared, one for Specific[cite: 108].
    """
    def __init__(self, input_dim, latent_dim):
        super(MOCSS_OmicsNet, self).__init__()
        
        # Shared Autoencoder
        self.enc_shared = Encoder(input_dim, latent_dim)
        self.dec_shared = Decoder(latent_dim, input_dim)
        
        # Specific Autoencoder
        self.enc_specific = Encoder(input_dim, latent_dim)
        self.dec_specific = Decoder(latent_dim, input_dim)
        
        # Nonlinear Projection Head for Contrastive Learning [cite: 300, 873]
        # Maps z_c -> h_c
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        # 1. Extract Representations
        z_c = self.enc_shared(x)   # Shared representation
        z_s = self.enc_specific(x) # Specific representation
        
        # 2. Reconstruct [cite: 848]
        # Note: Paper implies separate reconstruction from shared and specific
        x_rec_c = self.dec_shared(z_c)
        x_rec_s = self.dec_specific(z_s)
        
        # 3. Project shared features for contrastive loss [cite: 874]
        h_c = self.projection_head(z_c)
        
        return x_rec_c, x_rec_s, z_c, z_s, h_c

class MOCSS_Model(nn.Module):
    """
    Full Multi-Omics Model wrapper.
    """
    def __init__(self, input_dims, latent_dim):
        super(MOCSS_Model, self).__init__()
        self.views = nn.ModuleList([
            MOCSS_OmicsNet(dim, latent_dim) for dim in input_dims
        ])
    
    def forward(self, xs):
        outputs = []
        for i, net in enumerate(self.views):
            outputs.append(net(xs[i]))
        return outputs

# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================

def reconstruction_loss(x, x_rec_c, x_rec_s):
    """
    MSE Loss for both shared and specific reconstructions[cite: 853].
    L_rec = ||x - x_c||^2 + ||x - x_s||^2
    """
    loss_c = F.mse_loss(x_rec_c, x, reduction='mean')
    loss_s = F.mse_loss(x_rec_s, x, reduction='mean')
    return loss_c + loss_s

def contrastive_loss_pair(h_i, h_j, temperature=0.5):
    """
    NT-Xent Loss between two views (i and j)[cite: 888].
    """
    batch_size = h_i.shape[0]
    
    # Normalize vectors
    h_i = F.normalize(h_i, dim=1)
    h_j = F.normalize(h_j, dim=1)
    
    # Similarity matrix
    # sim[k, m] is similarity between sample k in view i and sample m in view j
    sim_matrix = torch.matmul(h_i, h_j.T) / temperature
    
    # Positive pairs are the diagonals (sample k in view i matches sample k in view j)
    labels = torch.arange(batch_size).to(h_i.device)
    
    # We want sim_matrix[k, k] to be high compared to sim_matrix[k, m]
    # CrossEntropy automatically does log-softmax
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def orthogonality_loss(z_c, z_s):
    """
    Ensures shared and specific features are distinct[cite: 916].
    Minimize Frobenius norm of the product of Shared and Specific matrices.
    """
    # z_c: (Batch, Latent), z_s: (Batch, Latent)
    # We want features to be orthogonal. 
    # Loss = || z_c^T * z_s ||_F^2
    
    # Normalize to prevent magnitude from dominating
    z_c_n = F.normalize(z_c, dim=0)
    z_s_n = F.normalize(z_s, dim=0)
    
    correlation_matrix = torch.matmul(z_c_n.T, z_s_n)
    loss = torch.norm(correlation_matrix, p='fro')
    return loss

# ==========================================
# 5. TRAINING & CLUSTERING UTILS
# ==========================================

def calculate_clustering_metrics(y_true, y_pred):
    """
    Calculates ACC, NMI, ARI as per paper[cite: 133].
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # Accuracy with Hungarian Algorithm for label matching [cite: 148]
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = w[row_ind, col_ind].sum() / y_pred.size
    
    return acc, nmi, ari

def train_mocss():
    print("Initializing MOCSS Model...")
    
    # Load Real Data
    rna_df, meth_df, cnv_df, Y, class_names = load_raw_aligned_data()
    
    # Preprocess
    print("Preprocessing data (Variance Filter + KNN Impute + Scaling)...")
    v1, v2, v3 = preprocess_full_data(rna_df, meth_df, cnv_df, max_features=5000)
    
    # Update Config dynamically
    CONFIG['input_dims'] = [v1.shape[1], v2.shape[1], v3.shape[1]]
    CONFIG['num_clusters'] = len(class_names)
    print(f"Data Shapes: {v1.shape}, {v2.shape}, {v3.shape}")
    print(f"Num Clusters: {CONFIG['num_clusters']}")
    
    # Create Dataset/Loader
    dataset = TensorDataset(v1, v2, v3, torch.LongTensor(Y))
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=False)
    
    # Initialize Model
    model = MOCSS_Model(CONFIG['input_dims'], CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training Loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss_epoch = 0
        
        for batch_idx, (v1, v2, v3, _) in enumerate(dataloader):
            views = [v1.to(CONFIG['device']), v2.to(CONFIG['device']), v3.to(CONFIG['device'])]
            
            # Forward pass
            # outputs is a list of tuples (x_rec_c, x_rec_s, z_c, z_s, h_c) for each view
            outputs = model(views)
            
            loss_rec = 0
            loss_orth = 0
            loss_con = 0
            
            # Calculate Intra-View Losses (Recon + Ortho)
            for i in range(len(views)):
                x_in = views[i]
                x_rec_c, x_rec_s, z_c, z_s, h_c = outputs[i]
                
                # Reconstruction Loss [cite: 853]
                loss_rec += reconstruction_loss(x_in, x_rec_c, x_rec_s)
                
                # Orthogonality Loss [cite: 916]
                loss_orth += orthogonality_loss(z_c, z_s)
            
            # Calculate Inter-View Losses (Contrastive)
            # Sum contrastive loss for all pairs of views [cite: 907]
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    h_i = outputs[i][4] # h_c from view i
                    h_j = outputs[j][4] # h_c from view j
                    
                    # Symmetric contrastive loss [cite: 903]
                    l_ij = contrastive_loss_pair(h_i, h_j, CONFIG['temp'])
                    l_ji = contrastive_loss_pair(h_j, h_i, CONFIG['temp'])
                    loss_con += (l_ij + l_ji)
            
            # Total Loss [cite: 921]
            total_loss = loss_rec + loss_con + loss_orth
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {total_loss_epoch:.4f}")

    print("Training Complete.")
    return model, dataset

# ==========================================
# 6. EVALUATION PIPELINE
# ==========================================

def evaluate_mocss(model, dataset):
    model.eval()
    
    # We need to process the whole dataset to get full embeddings
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    final_embeddings = []
    ground_truth = []
    
    with torch.no_grad():
        for v1, v2, v3, labels in loader:
            views = [v1.to(CONFIG['device']), v2.to(CONFIG['device']), v3.to(CONFIG['device'])]
            outputs = model(views)
            
            # Aggregate Representations [cite: 934, 938]
            # 1. Average the Shared Representations (z_c) across all views
            z_c_list = [out[2] for out in outputs] # out[2] is z_c
            z_c_avg = torch.mean(torch.stack(z_c_list), dim=0) # [cite: 935]
            
            # 2. Collect all Specific Representations (z_s)
            z_s_list = [out[3] for out in outputs] # out[3] is z_s
            
            # 3. Concatenate: [z_c_avg, z_s^1, z_s^2, z_s^3] [cite: 938]
            concat_list = [z_c_avg] + z_s_list
            z_final = torch.cat(concat_list, dim=1) # [cite: 940]
            
            final_embeddings.append(z_final.cpu().numpy())
            ground_truth.append(labels.numpy())
            
    final_embeddings = np.concatenate(final_embeddings, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    print(f"Feature extraction complete. Shape: {final_embeddings.shape}")
    
    # Clustering (Stage 2) [cite: 943]
    print("Running K-Means Clustering...")
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(final_embeddings)
    
    # Metrics [cite: 133]
    acc, nmi, ari = calculate_clustering_metrics(ground_truth, y_pred)
    
    print("\n=== MOCSS Evaluation Results ===")
    print(f"ACC: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")

# ==========================================
# 7. EXECUTION
# ==========================================
if __name__ == "__main__":
    trained_model, full_dataset = train_mocss()
    evaluate_mocss(trained_model, full_dataset)