# GatedFusionCMLP (Experiment V3)

> **Objective**: A robust multi-omics deep learning architecture designed for **High F1-Macro** performance on the **TCGA-SARC** dataset, specifically engineered to handle **low sample sizes** and **severe class imbalance**.

## 🧬 Project Overview

This research project introduces **GatedFusionCMLP**, a neural network framework that integrates heterogeneous omics data (RNA-Seq, DNA Methylation, Copy Number Variation) for the classification of Sarcoma subtypes.

By employing a **Gated Attention Mechanism**, the model dynamically learns the importance of each omics modality for every individual patient, filtering out specific noise and forcing the model to focus on the most confident signals. This is critical for the noisy, high-dimensional landscape of biological datasets.

## 🚀 Key Features

*   **Multi-Modal Fusion**: Simultaneous learning from RNA, Methylation, and CNV layers.
*   **Gated Attention**: A learnable gating mechanism ($\sigma(W \cdot z)$) that assigns an interpretability score $\alpha \in [0,1]$ to each modality before fusion.
*   **Imbalance Handling**:
    *   **Focal Loss**: Heavily penalized loss function to focus on "hard" examples.
    *   **Stratified Splits**: Ensures minority classes are represented in validation.
*   **Advanced Feature Selection**:
    *   **GPU-Accelerated mRMR**: Uses `cupy` to perform Minimum Redundancy Maximum Relevance selection on tens of thousands of features in seconds.
    *   **Variance Filtering**: Pre-screening to remove non-informative features.
*   **Biological Interpretability**:
    *   **SHAP Analysis**: Gradient-based feature importance ranking.
    *   **Gene Mapping**: Automatic conversion of Ensembl IDs to standard Gene Symbols using `MyGene`.

## 📊 Dataset

This project utilizes the TCGA-SARC (The Cancer Genome Atlas - Soft Tissue Sarcoma) dataset.

**[[LINK TO DATASET - REPLACE WITH ORIGINAL URL](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Sarcoma%20(SARC)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)]**

### Input Data Structure
The notebook expects the following CSV files in a `Data/` directory:
- `phenotype_clean.csv`: Clinical labels and diagnoses.
- `expression_log.csv`: Log-transformed RNA-Seq counts.
- `methylation_mvalues.csv`: Methylation Beta/M-values.
- `cnv_log.csv`: Copy Number Variation data.

## 🛠️ Architecture

1.  **Encoders**: Three parallel **PerOmicCMAE** (Contractive Autoencoder-style) blocks that compress high-dimensional omics into compact latent vectors ($z \in \mathbb{R}^{32}$).
2.  **Gating Layer**: 
    $$ \alpha_{modality} = \sigma(Linear(z_{modality})) $$
3.  **Fusion**: 
    $$ Z_{fused} = Concat(\alpha_{rna} \cdot z_{rna}, \ \alpha_{meth} \cdot z_{meth}, \ \alpha_{cnv} \cdot z_{cnv}) $$
4.  **Classification**: A fused Multi-Layer Perceptron (MLP) predicts the subtype.

## 📦 Dependencies

To run the experiment notebook, ensure the following are installed:

```bash
pip install torch pandas numpy scikit-learn seaborn matplotlib
pip install mrmr-selection shap mygene
# Optional for GPU acceleration
pip install cupy-cuda11x  # Adjust for your CUDA version
```

## 📉 Usage

Open the Jupyter Notebook `gatedfusioncmlp_FINAL_V3_Experiment_edition.ipynb` and run all cells.

The notebook will:
1.  Load and preprocess the data (Imputation + Scaling + mRMR).
2.  Train the model using Stratified 5-Fold Cross-Validation.
3.  Report **F1-Macro**, **Accuracy**, and **Precision/Recall**.
4.  Visualize the latent space using **t-SNE** and **UMAP**.
5.  Perform **SHAP** analysis to identify the top 15 biomarkers per omics.

## 🔬 Results & Interpretability

The model output includes:
- **Comparison**: Gated Fusion vs. Baseline (No Gating).
- **Omics Impact**: A percentage breakdown of how much each modality contributed to the decision.
- **Biomarkers**: List of top-ranked genes driving the predictions (e.g., *MDM2*, *CDK4*).

---
*Thesis Research Project [T2510589]*
