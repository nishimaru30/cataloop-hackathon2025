# predict.py
"""
Easy-to-use prediction interface for judges
"""

import pandas as pd
import numpy as np
import joblib
from drfp import DrfpEncoder
from olefin_descriptors import calculate_descriptors_for_dataset
from sklearn.preprocessing import StandardScaler

# Load pre-trained models
YIELD_MODEL = joblib.load('models/rf_yield_model_final.pkl')
EE_MODEL = joblib.load('models/rf_ee_model_final.pkl')

def predict_yield(df, reactant_col='Reactant SMILES', product_col='Product SMILES'):
    """
    Predict reaction yields for a dataframe of reactions
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with reactant and product SMILES
    
    Returns:
    --------
    predictions : np.array
        Predicted yields in %
    """
    
    print("Preparing features...")
    
    # Generate DRFP
    reaction_smiles = df[reactant_col] + '>>' + df[product_col]
    drfp = np.array(DrfpEncoder.encode(reaction_smiles.tolist(), n_folded_length=2048))
    
    # Calculate olefin descriptors
    olefin_desc = calculate_descriptors_for_dataset(df, reactant_col, product_col)
    
    # Handle NaN
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    olefin_desc = imputer.fit_transform(olefin_desc)
    
    # Scale descriptors
    scaler = StandardScaler()
    olefin_desc_scaled = scaler.fit_transform(olefin_desc)
    
    # Get categorical features
    categorical_cols = [
        'AD-mix alpha', 'AD-mix beta', 
        'Olefin Cis', 'Olefin Gem', 'Olefin Mono', 
        'Olefin Tetra', 'Olefin Trans', 'Olefin Tri',
        'Oxidant K3FeCN6', 'Oxidant NaClO2', 'Oxidant NMO'
    ]
    categorical = df[categorical_cols].values
    
    # Combine features
    X = np.hstack([drfp, olefin_desc_scaled, categorical])
    
    print("Making predictions...")
    predictions = YIELD_MODEL.predict(X)
    
    return predictions


def predict_ee(df, reactant_col='Reactant SMILES', product_col='Product SMILES'):
    """
    Predict enantioselectivity (ddG er) for reactions
    
    Returns:
    --------
    predictions : np.array
        Predicted ddG er in kcal/mol
    """
    # Same feature preparation as yield
    # ... (copy feature prep from predict_yield)
    
    predictions = EE_MODEL.predict(X)
    return predictions


# Example usage
if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv('test_data.csv')
    
    # Predict
    yield_pred = predict_yield(test_df)
    ee_pred = predict_ee(test_df)
    
    # Add to dataframe
    test_df['Predicted_Yield'] = yield_pred
    test_df['Predicted_EE'] = ee_pred
    
    # Save
    test_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
