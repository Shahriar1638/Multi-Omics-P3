import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "Data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DATA PREPROCESSING FOR 4 OMICS - NO DATA LEAKAGE VERSION")
print("=" * 70)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
print("NOTE: Each omics saved separately (no sample matching)")

print("\n" + "=" * 70)
print("STEP 1: LOADING RAW DATA")
print("=" * 70)

expression_data = pd.read_csv('RawData/TCGA-SARC.star_tpm.tsv', sep='\t', index_col=0)
methylation_data = pd.read_csv('RawData/TCGA-SARC.methylation450.tsv', sep='\t', index_col=0)
copy_number_data = pd.read_csv('RawData/TCGA-SARC.gene-level_absolute.tsv', sep='\t', index_col=0)
protein_data = pd.read_csv('RawData/TCGA-SARC.protein.tsv', sep='\t', index_col=0)

try:
    phenotype_data = pd.read_csv('RawData/TCGA-SARC.clinical.tsv', sep='\t', index_col=0)
except Exception as e:
    print(f"Warning: Initial load failed ({e}), using error handling...")
    phenotype_data = pd.read_csv('RawData/TCGA-SARC.clinical.tsv', sep='\t', index_col=0, on_bad_lines='skip')

print("Raw data shapes:")
print(f"  Expression: {expression_data.shape}")
print(f"  Methylation: {methylation_data.shape}")
print(f"  Copy Number: {copy_number_data.shape}")
print(f"  Protein: {protein_data.shape}")
print(f"  Clinical: {phenotype_data.shape}")

print("\n" + "=" * 70)
print("STEP 2: DROP ENTIRELY NaN ROWS AND DUPLICATE FEATURES")
print("=" * 70)

def drop_nan_and_duplicates(data, name):
    before_rows = data.shape[0]
    
    all_nan_mask = data.isna().all(axis=1)
    data = data.loc[~all_nan_mask]
    dropped_nan = all_nan_mask.sum()
    
    dup_mask = data.index.duplicated(keep='first')
    data = data.loc[~dup_mask]
    dropped_dup = dup_mask.sum()
    
    print(f"  {name}: {before_rows} -> {data.shape[0]} (dropped {dropped_nan} NaN rows, {dropped_dup} duplicates)")
    return data

expression_data = drop_nan_and_duplicates(expression_data, "Expression")
methylation_data = drop_nan_and_duplicates(methylation_data, "Methylation")
copy_number_data = drop_nan_and_duplicates(copy_number_data, "Copy Number")
protein_data = drop_nan_and_duplicates(protein_data, "Protein")

print("\n" + "=" * 70)
print("STEP 3: LOG TRANSFORMATIONS (Safe - No Data Leakage)")
print("=" * 70)

expression_log = np.log2(expression_data + 1)
print(f"  Expression: log2(x+1) transformation applied")

methylation_filtered = methylation_data.dropna(thresh=0.20 * methylation_data.shape[1], axis=0)
epsilon = 1e-6
methylation_clipped = methylation_filtered.clip(epsilon, 1 - epsilon)
methylation_mvalues = np.log2(methylation_clipped / (1 - methylation_clipped))
print(f"  Methylation: Dropped probes with >80% NaN, converted beta to M-values")
print(f"    Before: {methylation_data.shape[0]} -> After: {methylation_mvalues.shape[0]} probes")

cnv_filtered = copy_number_data.loc[copy_number_data.isnull().mean(axis=1) < 0.2]
cnv_clipped = cnv_filtered.clip(lower=0.05, upper=6)
cnv_log = np.log2(cnv_clipped / 2)
print(f"  Copy Number: Dropped genes with >20% NaN, clipped [0.05, 6], log2(x/2)")
print(f"    Before: {copy_number_data.shape[0]} -> After: {cnv_log.shape[0]} genes")

protein_filtered = protein_data.loc[protein_data.isnull().mean(axis=1) < 0.2]
print(f"  Protein: Dropped proteins with >20% NaN")
print(f"    Before: {protein_data.shape[0]} -> After: {protein_filtered.shape[0]} proteins")

print("\n" + "=" * 70)
print("STEP 4: PHENOTYPE PROCESSING AND SUBTYPE FILTERING")
print("=" * 70)

subtype_column = 'primary_diagnosis.diagnoses'
selected_subtypes = [
    'Leiomyosarcoma, NOS',
    'Dedifferentiated liposarcoma',
    'Undifferentiated sarcoma',
    'Fibromyxosarcoma'
]

print(f"Target column: '{subtype_column}'")
print(f"Selected subtypes: {selected_subtypes}")

phenotype_filtered = phenotype_data[phenotype_data[subtype_column].isin(selected_subtypes)]
phenotype_clean = phenotype_filtered.dropna(subset=[subtype_column])

subtypes = phenotype_clean[subtype_column]
label_encoder = LabelEncoder()
labels = pd.Series(
    label_encoder.fit_transform(subtypes),
    index=subtypes.index,
    name='label'
)

print(f"\nLabel encoding:")
for i, cls in enumerate(label_encoder.classes_):
    count = (labels == i).sum()
    print(f"  {i}: {cls} ({count} samples)")

print("\n" + "=" * 70)
print("STEP 5: SAVING EACH OMICS SEPARATELY (NO SAMPLE MATCHING)")
print("=" * 70)
print("Each omics will be saved with ALL its samples.")
print("Sample matching can be done later for different omics combinations.")

expression_log.to_csv(f"{OUTPUT_DIR}expression_log.csv")
print(f"  Saved: expression_log.csv {expression_log.shape}")

methylation_mvalues.to_csv(f"{OUTPUT_DIR}methylation_mvalues.csv")
print(f"  Saved: methylation_mvalues.csv {methylation_mvalues.shape}")

cnv_log.to_csv(f"{OUTPUT_DIR}cnv_log.csv")
print(f"  Saved: cnv_log.csv {cnv_log.shape}")

protein_filtered.to_csv(f"{OUTPUT_DIR}protein_filtered.csv")
print(f"  Saved: protein_filtered.csv {protein_filtered.shape}")

phenotype_clean.to_csv(f"{OUTPUT_DIR}phenotype_clean.csv")
print(f"  Saved: phenotype_clean.csv {phenotype_clean.shape}")

labels.to_csv(f"{OUTPUT_DIR}labels_all.csv", header=True)
print(f"  Saved: labels_all.csv ({len(labels)} samples)")

label_mapping = pd.DataFrame({
    'encoded': list(range(len(label_encoder.classes_))),
    'subtype': label_encoder.classes_
})
label_mapping.to_csv(f"{OUTPUT_DIR}label_encoding.csv", index=False)
print(f"  Saved: label_encoding.csv")

print("\n" + "=" * 70)
print("STEP 6: SAMPLE AVAILABILITY INFO")
print("=" * 70)

print("\nSamples per omics:")
print(f"  Expression: {len(expression_log.columns)} samples")
print(f"  Methylation: {len(methylation_mvalues.columns)} samples")
print(f"  Copy Number: {len(cnv_log.columns)} samples")
print(f"  Protein: {len(protein_filtered.columns)} samples")
print(f"  Phenotype (with labels): {len(phenotype_clean)} samples")

sample_info = pd.DataFrame({
    'omics': ['Expression', 'Methylation', 'Copy Number', 'Protein', 'Phenotype'],
    'n_samples': [
        len(expression_log.columns),
        len(methylation_mvalues.columns),
        len(cnv_log.columns),
        len(protein_filtered.columns),
        len(phenotype_clean)
    ],
    'n_features': [
        expression_log.shape[0],
        methylation_mvalues.shape[0],
        cnv_log.shape[0],
        protein_filtered.shape[0],
        phenotype_clean.shape[1]
    ]
})
sample_info.to_csv(f"{OUTPUT_DIR}omics_info.csv", index=False)
print(f"\n  Saved: omics_info.csv")

print("\nPossible omics combinations for experiments:")
expr_samples = set(expression_log.columns)
meth_samples = set(methylation_mvalues.columns)
cnv_samples = set(cnv_log.columns)
prot_samples = set(protein_filtered.columns)
pheno_samples = set(phenotype_clean.index)

combos = {
    'Expr + Meth': len(expr_samples & meth_samples & pheno_samples),
    'Expr + CNV': len(expr_samples & cnv_samples & pheno_samples),
    'Expr + Prot': len(expr_samples & prot_samples & pheno_samples),
    'Meth + CNV': len(meth_samples & cnv_samples & pheno_samples),
    'Meth + Prot': len(meth_samples & prot_samples & pheno_samples),
    'CNV + Prot': len(cnv_samples & prot_samples & pheno_samples),
    'Expr + Meth + CNV': len(expr_samples & meth_samples & cnv_samples & pheno_samples),
    'Expr + Meth + Prot': len(expr_samples & meth_samples & prot_samples & pheno_samples),
    'Expr + CNV + Prot': len(expr_samples & cnv_samples & prot_samples & pheno_samples),
    'Meth + CNV + Prot': len(meth_samples & cnv_samples & prot_samples & pheno_samples),
    'All 4 Omics': len(expr_samples & meth_samples & cnv_samples & prot_samples & pheno_samples),
}

combo_df = pd.DataFrame(list(combos.items()), columns=['Combination', 'N_Samples'])
combo_df.to_csv(f"{OUTPUT_DIR}sample_combinations.csv", index=False)
print(combo_df.to_string(index=False))
print(f"\n  Saved: sample_combinations.csv")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nAll files saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\nOmics files (log-transformed, ready for preprocessing):")
print("  - expression_log.csv")
print("  - methylation_mvalues.csv")
print("  - cnv_log.csv")
print("  - protein_filtered.csv")

print("\nLabel/Phenotype files:")
print("  - phenotype_clean.csv")
print("  - labels_all.csv")
print("  - label_encoding.csv")

print("\nInfo files:")
print("  - omics_info.csv")
print("  - sample_combinations.csv")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  1. Choose omics combination for experiment")
print("  2. Match samples across chosen omics")
print("  3. Split into train/test")
print("  4. Use flexible_preprocessor.py for imputation/scaling")