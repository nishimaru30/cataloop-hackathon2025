# Asymmetric Dihydroxylation: Yield and Enantioselectivity Prediction

David, Rania, Doug, Nishi -- CATALOOP Hackathon 2025

Install requirements
pip install -r requirements.txt

cataloop-hackathon2025/
│
├── README.md                           # Explanation of your approach
├── requirements.txt                    # Dependencies
│
├── code/
│   ├── olefin_descriptors.py         
│   ├──hackathon_final.ipynb
│
├── results/
│   ├── cv_results_yield_rf.csv        # Cross-validation results
│   ├── cv_results_ee_rf.csv
│   ├── feature_importance_yield_rf.csv
│   ├── feature_importance_ee_rf.csv
│   ├── rf_predictions_with_olefin_desc.png
│   └── rf_ee_predictions.png
│
└── data/
    └── data_preprocessed.csv       

## Approach Summary

We developed two Random Forest models to predict:
1. **Reaction yield (%)** 
2. **Enantioselectivity (ddG er in kcal/mol)**

### Key Features
- **Reaction fingerprints**: DRFP (Difference of Reaction Fingerprints) - 2048 bits
- **Custom olefin descriptors**: 66 reaction-site-specific molecular descriptors
  - Substitution patterns (mono/di/tri/tetra)
  - Stereochemistry (E/Z configuration)
  - Electronic effects (conjugation, aromaticity)
  - Steric environment
  - Chirality information
- **Reaction conditions**: Categorical features (AD-mix type, olefin class, oxidant)

### Methodology

1. Generated DRFP fingerprints for reaction transformation
2. Calculated custom descriptors for the reactive olefin (site-specific)
3. Removed highly correlated features (threshold > 0.95)
4. 5-fold cross-validation with stratified splits
5. Random Forest Regressor (200 trees, max_depth=20)
6. Feature importance analysis
