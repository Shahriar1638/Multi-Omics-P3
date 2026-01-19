import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import seaborn as sns

class ManifoldDenoisingInference:
    def __init__(self, n_neighbors=7, alpha=0.85):
        """
        Manifold Learning & Signal Denoising + Inductive Inference Pipeline.
        
        Args:
            n_neighbors (int): k for k-NN graph construction.
            alpha (float): Diffusion parameter (0 < alpha < 1).
        """
        self.k = n_neighbors
        self.alpha = alpha
        
        # Storage
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.Y_soft = None
        self.P_matrix = None
        self.Y_init = None
        
    def fit(self, X_train, y_train):
        """
        Phase 1: Manifold Learning & Signal Denoising (Training Phase)
        Constructs adaptive graph and performs signal diffusion.
        """
        print(f"Fitting Manifold Denoising (k={self.k}, alpha={self.alpha})...")
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.classes = np.unique(y_train)
        n_samples = self.X_train.shape[0]
        n_classes = len(self.classes)
        
        # --- 3.1 Adaptive Graph Construction (Zelnik-Manor) ---
        print("  1. Building Adaptive k-NN Graph...")
        nbrs = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nbrs.fit(self.X_train)
        distances, indices = nbrs.kneighbors(self.X_train)
        
        # Adaptive Scaling: sigma_i = distance to k-th neighbor
        # distances[:, k-1] is the distance to the k-th neighbor (0-indexed)
        # Note: indices[:,0] is self, distances[:,0] is 0. 
        # We need distance to k-th neighbor. k-neighbors returns k neighbors including self?
        # sklearn returns k neighbors. If query is in data, it is included.
        # So k-th neighbor is at index k-1.
        sigmas = distances[:, self.k - 1]
        
        # Construct Adjacency Matrix W
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for idx, j in enumerate(indices[i]):
                if i == j: continue # Skip self-loop for now? Usually W_ii = 0 or 1. Let's say 0.
                
                # Similarity formula: exp( -d(xi, xj)^2 / (sigma_i * sigma_j) )
                d_sq = distances[i, idx] ** 2
                sigma_prod = sigmas[i] * sigmas[j]
                
                # Avoid division by zero
                if sigma_prod < 1e-10: 
                    sigma_prod = 1e-10
                    
                w_ij = np.exp(-d_sq / sigma_prod)
                W[i, j] = w_ij
                
        # Symmetrize W? Zelnik-Manor typically implies symmetric affinity.
        # The k-NN relationship is not symmetric.
        # "An adjacency matrix is constructed using a k-Nearest Neighbor graph..."
        # Often we do W = max(W, W.T) or W = (W + W.T)/2
        # Let's use W = (W + W.T) / 2 to ensure symmetry for spectral properties, 
        # though Random Walk doesn't strictly adhere to symmetry.
        # Given "Result: ... Diffusion Operator", usually stable on symmetric graphs first.
        # But let's stick to the connectivity defined by the loop.
        
        # --- 3.2 Class-Balanced Signal Injection ---
        print("  2. Injecting Class-Balanced Signals...")
        Y_init = np.zeros((n_samples, n_classes))
        
        # Count per class
        class_counts = {c: np.sum(self.y_train == c) for c in self.classes}
        
        for i in range(n_samples):
            label = self.y_train[i]
            class_idx = np.where(self.classes == label)[0][0]
            # Energy = 1 / |C_c|
            Y_init[i, class_idx] = 1.0 / class_counts[label]
            
        self.Y_init = Y_init
            
        # --- 3.3 Graph Diffusion (Smoothing) ---
        print("  3. Propagating Signals (Diffusion)...")
        # Random Walk Matrix P = D^-1 W
        # D is diagonal matrix of row sums
        degrees = np.sum(W, axis=1)
        degrees[degrees == 0] = 1e-10 # Avoid nan
        
        # D_inv = np.diag(1/degrees)
        # P = D_inv @ W
        # Optimized:
        P = W / degrees[:, np.newaxis]
        self.P_matrix = P
        
        # Diffusion Operator: Y_soft = (I - alpha * P)^-1 @ Y_init
        # We solve (I - alpha * P) Y_soft = Y_init
        I = np.eye(n_samples)
        
        # Using linear solver for stability
        A = I - self.alpha * P
        self.Y_soft = np.linalg.solve(A, Y_init)
        
        print("  Training phase complete.")
        return self
        
    def predict(self, X_test):
        """
        Phase 2: Inductive Inference (Testing Phase)
        Predict without reconstructing the graph (Zero Leakage).
        """
        X_test = np.array(X_test)
        n_test = X_test.shape[0]
        n_classes = len(self.classes)
        
        print("Running Inductive Inference...")
        
        # 1. Neighbor Identification in Training Latent Space
        # We need neighbors in X_train for each x in X_test
        nbrs = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nbrs.fit(self.X_train)
        
        distances, indices = nbrs.kneighbors(X_test)
        
        y_pred = []
        y_scores = np.zeros((n_test, n_classes))
        
        for i in range(n_test):
            # Neighbors for test patient i
            neighbor_indices = indices[i]
            neighbor_dists = distances[i]
            
            # 2. Distance Weighting
            # Compute weights w_j
            # Prompt: "based on Euclidean distance"
            # Using Gaussian kernel style weight: w = exp(-dist) or 1/(dist+eps)
            # Let's use Inverse Distance Weighting (IDW) or similar since explicit formula wasn't captured?
            # Or re-use the Zelnik-Manor logic?
            # "Similarity between patient i and neighbor j" in training used ZM.
            # Local scaling for test point? sigma_test = dist to k-th neighbor?
            # Let's compute sigma_test for the test point i
            sigma_test = neighbor_dists[-1] # dist to k-th neighbor
            if sigma_test < 1e-10: sigma_test = 1.0
            
            weights = []
            features_smooth = np.zeros(n_classes)
            
            for rank, n_idx in enumerate(neighbor_indices):
                dist = neighbor_dists[rank]
                
                # Retrieve "sigma_j" from training data for the neighbor
                # We need to re-compute or store sigmas from training?
                # In training we computed sigmas. Let's re-compute them or just use a simpler weight.
                # "based on Euclidean distance" -> e.g. w = 1 / (d + 1e-6)
                # Let's use a standard robust kernel: Gaussian
                w = np.exp(- (dist**2) / (sigma_test**2 + 1e-10))
                # Or simply 1/dist
                # w = 1.0 / (dist + 1e-9)
                
                weights.append(w)
            
            weights = np.array(weights)
            
            # 3. Weighted Voting
            # Score = average of Soft Labels (Y_soft) of neighbors, weighted by w
            # Y_soft for neighbor n_idx
            neighbor_soft_labels = self.Y_soft[neighbor_indices] # shape (k, n_classes)
            
            # weighted sum
            # sum(w * y_soft) / sum(w)
            weighted_sum = np.sum(neighbor_soft_labels * weights[:, np.newaxis], axis=0)
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                score = weighted_sum / total_weight
            else:
                score = np.mean(neighbor_soft_labels, axis=0) # Fallback
                
            y_scores[i] = score
            
            # 4. Final Prediction
            # Max score wins
            best_class_idx = np.argmax(score)
            y_pred.append(self.classes[best_class_idx])
            
        return np.array(y_pred), y_scores

def load_mofa_embeddings(results_dir="Results_Mofaflex", n_features=None):
    """Load the embeddings generated by mofaflex pipeline."""
    # Find the file
    if n_features is None:
        # Try to find available files
        files = [f for f in os.listdir(results_dir) if f.startswith("mofa_embeddings_") and f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No embeddings found in results directory.")
        # Pick the one with highest features or ask? Let's pick 2000 or the first one.
        # Usually user prefers 'best'. Let's pick 6000? Or just the first found.
        # User prompt pointed to code with [2000, 4000, 6000].
        # Let's default to the middle one or look for 'best' logic if implemented.
        target_file = files[0] # Fallback
        
        # Try to parse n_features from filename
        import re
        for f in files:
            if "2000" in f: target_file = f # Preference for 2000 often baseline
    else:
        target_file = f"mofa_embeddings_{n_features}_features.csv"
        
    path = os.path.join(results_dir, target_file)
    print(f"Loading embeddings from: {path}")
    
    df = pd.read_csv(path, index_col=0)
    return df

if __name__ == "__main__":
    # --- Configuration ---
    RESULTS_DIR = "Results_Mofaflex"
    N_FEATURES = 2000 # Can be changed to 4000, 6000
    K_NEIGHBORS = 5
    DIFFUSION_ALPHA = 0.8
    
    try:
        # 1. Load Data
        embeddings_df = load_mofa_embeddings(RESULTS_DIR, N_FEATURES)
        
        # Split back into Train and Test
        train_df = embeddings_df[embeddings_df['Split'] == 'train']
        test_df = embeddings_df[embeddings_df['Split'] == 'test']
        
        # Extract X (features) and y (labels)
        feature_cols = [c for c in embeddings_df.columns if c.startswith('Factor_')]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['Label'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['Label'].values
        
        print(f"Data Loaded: Train {X_train.shape}, Test {X_test.shape}")
        
        # 2. Run Pipeline
        model = ManifoldDenoisingInference(n_neighbors=K_NEIGHBORS, alpha=DIFFUSION_ALPHA)
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        y_pred, y_scores = model.predict(X_test)
        
        # 3. Evaluate
        acc = accuracy_score(y_test, y_pred)
        print("\n" + "="*50)
        print(f"RESULTS (k={K_NEIGHBORS}, alpha={DIFFUSION_ALPHA})")
        print("="*50)
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save Predictions
        output_df = test_df.copy()
        output_df['Predicted_Label'] = y_pred
        for i, cls in enumerate(model.classes):
            output_df[f'Score_{cls}'] = y_scores[:, i]
            
        output_path = os.path.join(RESULTS_DIR, "manifold_denoising_predictions.csv")
        output_df.to_csv(output_path)
        print(f"\nPredictions saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MOFA embeddings are generated first using 'mofaflex_with_nan_pipline_fixed.py'.")
