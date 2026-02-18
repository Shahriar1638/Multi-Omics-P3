import json
import os

source_file = "gatedfusioncmlp_final_v3_experiment_edition_v2.py"
target_notebook = "GatedFusionCMLP_SHAP_Analysis.ipynb"

# Read source code
print(f"Reading {source_file}...")
with open(source_file, 'r', encoding='utf-8') as f:
    source_code = f.readlines()

# Normalize lines (ensure each line ends with \n except specific cases, but readlines keeps \n)

# Notebook structure
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gated Fusion CMLP SHAP Analysis\n",
    f"Code migrated from `{source_file}`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": source_code
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

print(f"Writing to {target_notebook}...")
with open(target_notebook, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Done.")
