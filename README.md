# Asymmetric Dihydroxylation: Yield and Enantioselectivity Prediction  
**Team:** David, Rania, Doug, Nishi  
**Event:** CATALOOP Hackathon 2025  

---

## Project Structure
```
cataloop-hackathon2025/
├── README.md
├── requirements.txt
├── code/
│ ├── olefin_descriptors.py
│ └── hackathon_final.ipynb
├── results/
│ ├── cv_results_yield_rf.csv
│ ├── cv_results_ee_rf.csv
│ ├── feature_importance_yield_rf.csv
│ ├── feature_importance_ee_rf.csv
│ ├── rf_predictions_yield.png
│ └── rf_predictions_ee.png
└── data/
  └── data_preprocessed.csv
```

## Approach Summary

We developed two **Random Forest models** to predict:

- **Reaction yield (%)**
- **Enantioselectivity (ddG er in kcal/mol)**

### Key Features (2107 total)

- **DRFP fingerprints:** 2048-bit reaction fingerprints  
- **Olefin descriptors:** 48 custom reaction-site descriptors  
- **Substitution patterns:** mono/di/tri/tetra-substituted  
- **Stereochemistry:** E/Z configuration  
- **Electronic effects:** aromaticity, conjugation, charge distribution  
- **Steric environment:** nearby substituents, ring systems  
- **Chirality information**  
- **Reaction conditions:** 11 categorical features (AD-mix, oxidant, olefin class)

## Methodology

- Generated DRFP fingerprints for reaction transformation  
- Calculated 66 custom olefin descriptors  
- Removed 18 highly correlated features (**correlation > 0.90**)  
- 5-fold cross-validation using `random_state=42`  
- **Random Forest Regressor**  
  - 200 trees  
  - max_depth = 20 
- Feature importance analysis for interpretability

### Install dependencies

```bash
pip install -r requirements.txt
```
Run the notebook:
```jupyter notebook code/hackathon_final.ipynb```

Running all cells will:
- Train yield + EE models via 5-fold CV
- Generate holdout predictions
- Save metrics and figures to results/

### Innovation

- We designed reaction-mapped olefin descriptors tailored to asymmetric dihydroxylation:
- Focus on the reactive C=C double bond
- Encodes steric + electronic environment around the reaction center
- Accounts for E/Z stereochemistry, substitution patterns, and conjugation
- Produces meaningful, interpretable features for chemists


_Built for CATALOOP Hackathon 2025_
