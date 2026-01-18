
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Try importing fancyimpute for SoftImpute, handle unavailability gracefully
try:
    from fancyimpute import SoftImpute
    HAS_SOFTIMPUTE = True
except ImportError:
    HAS_SOFTIMPUTE = False

def preprocess_omics(df, variance_pct=0.3, impute_method='missforest', random_state=42):
    """
    Preprocess omics data with imputation, variance filtering, and scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe (Samples x Features).
        variance_pct (float): Percentage of top variance features to keep (e.g., 0.3 for top 30%).
        impute_method (str): 'softimpute' or 'missforest'.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Processed dataframe (Samples x Features).
    """
    # Work on a copy
    data = df.copy()
    
    print(f"--- Preprocessing Start ---")
    print(f"Shape: {data.shape}")
    print(f"Imputation Method: {impute_method}")
    print(f"Variance Filter: Top {int(variance_pct*100)}%")

    # 1. VARIANCE FILTERING (Keep Top % Features)
    # Calculate variance per feature (Pandas var() ignores NaNs by default)
    print("Calculating variance (ignoring NaNs)...")
    variances = data.var()
    n_features = len(variances)
    n_keep = int(n_features * variance_pct)
    n_keep = max(1, n_keep) # Keep at least 1
    
    # Identify top features
    top_features = variances.nlargest(n_keep).index
    data = data[top_features]
    print(f"Variance Filtering: Keeping {n_keep}/{n_features} features.")

    # 2. IMPUTATION
    if data.isnull().values.any():
        print(f"Missing values detected. Starting imputation...")
        if impute_method.lower() == 'missforest':
            # MissForest approximation using IterativeImputer with RF
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_jobs=-1, 
                    random_state=random_state
                ),
                max_iter=10,
                random_state=random_state,
                verbose=1
            )
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
            
        elif impute_method.lower() == 'softimpute':
            if not HAS_SOFTIMPUTE:
                raise ImportError("fancyimpute library is required for SoftImpute. Please install it via `pip install fancyimpute`.")
            
            # SoftImpute works on matrix directly
            imputer = SoftImpute(verbose=True)
            # SoftImpute expects rows=samples, cols=features
            data_imputed = imputer.fit_transform(data.values)
            data = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
            
        else:
            raise ValueError(f"Unknown imputation method: {impute_method}. Use 'missforest' or 'softimpute'.")
        print("Imputation complete.")
    else:
        print("No missing values found. Skipping imputation.")

    # 3. SCALING
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
    
    print("--- Preprocessing Done ---")
    return data

if __name__ == "__main__":
    # Example Usage Test
    print("Running test with synthetic data...")
    
    # Create synthetic data with missing values
    np.random.seed(42)
    n_samples, n_features = 20, 50
    X = np.random.randn(n_samples, n_features)
    # Introduce missing values
    mask = np.random.choice([True, False], size=X.shape, p=[0.1, 0.9])
    X_missing = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    X_missing[mask] = np.nan
    
    try:
        # Test MissForest
        print("\nTesting MissForest:")
        res_mf = preprocess_omics(X_missing, variance_pct=0.5, impute_method='missforest')
        print("MissForest Result Shape:", res_mf.shape)
        
        # Test SoftImpute (will fail if not installed, but checking logic)
        # print("\nTesting SoftImpute:")
        # res_si = preprocess_omics(X_missing, variance_pct=0.5, impute_method='softimpute')
        # print("SoftImpute Result Shape:", res_si.shape)
        
    except Exception as e:
        print(f"Test failed: {e}")
