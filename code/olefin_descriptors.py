"""
Reaction-Mapped Olefin Descriptor Calculator
For asymmetric dihydroxylation reactions
Includes custom olefin descriptors + additional RDKit descriptors
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
import numpy as np
import pandas as pd


def get_mapped_olefin_descriptors(reactant_smiles, product_smiles):
    """
    Calculate site-specific descriptors for the reactive olefin in dihydroxylation.
    
    Parameters:
    -----------
    reactant_smiles : str
        SMILES string of the reactant
    product_smiles : str
        SMILES string of the product
    
    Returns:
    --------
    dict : Dictionary of descriptors (with NaN for failures)
    """
    
    try:
        # Parse and add hydrogens
        reactant = Chem.MolFromSmiles(reactant_smiles)
        if reactant is None:
            return {key: np.nan for key in get_descriptor_names()}
        
        reactant = Chem.AddHs(reactant)
        
        # Find the olefin
        olefin_atoms, olefin_bond = find_reactive_olefin(reactant)
        
        if not olefin_atoms:
            return {key: np.nan for key in get_descriptor_names()}
        
        atom1, atom2 = olefin_atoms
        
        # Get neighbors (excluding the C=C bond itself)
        neighbors1 = [n for n in atom1.GetNeighbors() if n.GetIdx() != atom2.GetIdx()]
        neighbors2 = [n for n in atom2.GetNeighbors() if n.GetIdx() != atom1.GetIdx()]
        all_substituents = neighbors1 + neighbors2
        
        # ===== CUSTOM OLEFIN DESCRIPTORS =====
        
        # Basic molecular properties
        molwt = Descriptors.MolWt(reactant)
        logp = Crippen.MolLogP(reactant)
        tpsa = Descriptors.TPSA(reactant)
        
        # Substitution pattern
        h_on_c1 = sum(1 for n in neighbors1 if n.GetSymbol() == 'H')
        h_on_c2 = sum(1 for n in neighbors2 if n.GetSymbol() == 'H')
        total_h_on_olefin = h_on_c1 + h_on_c2
        
        non_h_neighbors1 = [n for n in neighbors1 if n.GetSymbol() != 'H']
        non_h_neighbors2 = [n for n in neighbors2 if n.GetSymbol() != 'H']
        
        num_substituents_c1 = len(non_h_neighbors1)
        num_substituents_c2 = len(non_h_neighbors2)
        total_substitution = num_substituents_c1 + num_substituents_c2
        
        # Classification
        is_monosubstituted = int(total_substitution == 1)
        is_gem = int((num_substituents_c1 == 2 and num_substituents_c2 == 0) or 
                      (num_substituents_c1 == 0 and num_substituents_c2 == 2))
        is_disubstituted = int(total_substitution == 2 and not is_gem)
        is_trisubstituted = int(total_substitution == 3)
        is_tetrasubstituted = int(total_substitution == 4)
        
        # Stereochemistry
        stereo = olefin_bond.GetStereo()
        is_E = int(stereo == Chem.BondStereo.STEREOE)
        is_Z = int(stereo == Chem.BondStereo.STEREOZ)
        is_unspecified = int(stereo == Chem.BondStereo.STEREONONE)
        
        # Environment
        olefin_in_ring = int(atom1.IsInRing() or atom2.IsInRing())
        olefin_aromatic = int(atom1.GetIsAromatic() or atom2.GetIsAromatic())
        
        # Ring size if in ring
        olefin_ring_size = 0
        if olefin_in_ring:
            for ring in reactant.GetRingInfo().BondRings():
                if olefin_bond.GetIdx() in ring:
                    olefin_ring_size = len(ring)
                    break
        
        # Substituent types
        carbon_substituents = sum(1 for n in all_substituents if n.GetSymbol() == 'C')
        aromatic_substituents = sum(1 for n in all_substituents if n.GetIsAromatic())
        hetero_substituents = sum(1 for n in all_substituents if n.GetSymbol() not in ['C', 'H'])
        
        # Steric environment
        nearby_heavy_atoms = sum(1 for n in all_substituents if n.GetSymbol() != 'H')
        
        # Conjugation
        conjugated = 0
        for neighbor in all_substituents:
            if neighbor.GetSymbol() == 'C':
                for bond in neighbor.GetBonds():
                    if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                        if bond.GetIdx() != olefin_bond.GetIdx():
                            conjugated = 1
                            break
        
        # Chirality
        chiral_centers_adjacent = 0
        for neighbor in all_substituents:
            try:
                if neighbor.HasProp('_CIPCode'):
                    chiral_centers_adjacent += 1
            except:
                pass
        
        total_stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(reactant)
        
        # Functional groups in molecule
        num_OH = len(reactant.GetSubstructMatches(Chem.MolFromSmarts('[OH]')))
        num_carbonyl = len(reactant.GetSubstructMatches(Chem.MolFromSmarts('[C]=[O]')))
        num_ester = len(reactant.GetSubstructMatches(Chem.MolFromSmarts('[C](=[O])[O]')))
        num_amine = len(reactant.GetSubstructMatches(Chem.MolFromSmarts('[N;!$(N=*);!$(N#*)]')))
        
        # Additional molecular properties
        num_aromatic_rings = Descriptors.NumAromaticRings(reactant)
        num_aliphatic_rings = Descriptors.NumAliphaticRings(reactant)
        num_rotatable_bonds = Lipinski.NumRotatableBonds(reactant)
        num_hdonors = Lipinski.NumHDonors(reactant)
        num_hacceptors = Lipinski.NumHAcceptors(reactant)
        
        # Fraction Csp3
        num_csp3 = sum(1 for atom in reactant.GetAtoms() 
                       if atom.GetSymbol() == 'C' and 
                       atom.GetHybridization() == Chem.HybridizationType.SP3)
        num_carbons = sum(1 for atom in reactant.GetAtoms() if atom.GetSymbol() == 'C')
        fraction_csp3 = num_csp3 / num_carbons if num_carbons > 0 else 0
        
        # ===== ADDITIONAL RDKIT DESCRIPTORS =====
        
        # Electronic properties
        mol_mr = Crippen.MolMR(reactant)
        max_partial_charge = Descriptors.MaxPartialCharge(reactant)
        min_partial_charge = Descriptors.MinPartialCharge(reactant)
        max_abs_partial_charge = Descriptors.MaxAbsPartialCharge(reactant)
        min_abs_partial_charge = Descriptors.MinAbsPartialCharge(reactant)
        num_valence_electrons = Descriptors.NumValenceElectrons(reactant)
        
        # Complexity indices
        balaban_j = Descriptors.BalabanJ(reactant)
        bertz_ct = Descriptors.BertzCT(reactant)
        hall_kier_alpha = Descriptors.HallKierAlpha(reactant)
        kappa1 = Descriptors.Kappa1(reactant)
        kappa2 = Descriptors.Kappa2(reactant)
        kappa3 = Descriptors.Kappa3(reactant)
        
        # Additional structural
        num_spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(reactant)
        num_bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(reactant)
        num_saturated_rings = Descriptors.NumSaturatedRings(reactant)
        num_aliphatic_carbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(reactant)
        num_aliphatic_heterocycles = rdMolDescriptors.CalcNumAliphaticHeterocycles(reactant)
        num_aliphatic_rings_calc = rdMolDescriptors.CalcNumAliphaticRings(reactant)
        
        # Counts
        num_radical_electrons = Descriptors.NumRadicalElectrons(reactant)
        heavy_atom_count = Descriptors.HeavyAtomCount(reactant)
        
        # Polarity/surface area
        labute_asa = Descriptors.LabuteASA(reactant)
        
        # Chi indices (connectivity)
        chi0 = Descriptors.Chi0(reactant)
        chi1 = Descriptors.Chi1(reactant)
        chi0v = Descriptors.Chi0v(reactant)
        chi1v = Descriptors.Chi1v(reactant)
        
        # Additional ring counts
        num_saturated_carbocycles = Descriptors.NumSaturatedCarbocycles(reactant)
        num_saturated_heterocycles = Descriptors.NumSaturatedHeterocycles(reactant)
        num_aliphatic_carbocycles_desc = Descriptors.NumAliphaticCarbocycles(reactant)
        num_aliphatic_heterocycles_desc = Descriptors.NumAliphaticHeterocycles(reactant)
        
        # ===== RETURN ALL DESCRIPTORS =====
        descriptors = {
            # Custom olefin descriptors (38)
            'MolWt': molwt,
            'LogP': logp,
            'TPSA': tpsa,
            'NumSubstituents_C1': num_substituents_c1,
            'NumSubstituents_C2': num_substituents_c2,
            'TotalSubstitution': total_substitution,
            'TotalH_OnOlefin': total_h_on_olefin,
            'H_OnC1': h_on_c1,
            'H_OnC2': h_on_c2,
            'IsMonosubstituted': is_monosubstituted,
            'IsGeminal': is_gem,
            'IsDisubstituted': is_disubstituted,
            'IsTrisubstituted': is_trisubstituted,
            'IsTetrasubstituted': is_tetrasubstituted,
            'IsE_Olefin': is_E,
            'IsZ_Olefin': is_Z,
            'IsUnspecifiedStereo': is_unspecified,
            'OlefinInRing': olefin_in_ring,
            'OlefinRingSize': olefin_ring_size,
            'OlefinAromatic': olefin_aromatic,
            'CarbonSubstituents': carbon_substituents,
            'AromaticSubstituents': aromatic_substituents,
            'HeteroSubstituents': hetero_substituents,
            'NearbyHeavyAtoms': nearby_heavy_atoms,
            'Conjugated': conjugated,
            'ChiralCentersAdjacent': chiral_centers_adjacent,
            'TotalStereocenters': total_stereocenters,
            'NumOH': num_OH,
            'NumCarbonyl': num_carbonyl,
            'NumEster': num_ester,
            'NumAmine': num_amine,
            'NumAromaticRings': num_aromatic_rings,
            'NumAliphaticRings': num_aliphatic_rings,
            'NumRotatableBonds': num_rotatable_bonds,
            'NumHDonors': num_hdonors,
            'NumHAcceptors': num_hacceptors,
            'FractionCsp3': fraction_csp3,
            
            # Additional RDKit descriptors (30)
            'MolMR': mol_mr,
            'MaxPartialCharge': max_partial_charge,
            'MinPartialCharge': min_partial_charge,
            'MaxAbsPartialCharge': max_abs_partial_charge,
            'MinAbsPartialCharge': min_abs_partial_charge,
            'NumValenceElectrons': num_valence_electrons,
            'BalabanJ': balaban_j,
            'BertzCT': bertz_ct,
            'HallKierAlpha': hall_kier_alpha,
            'Kappa1': kappa1,
            'Kappa2': kappa2,
            'Kappa3': kappa3,
            'NumSpiroAtoms': num_spiro_atoms,
            'NumBridgeheadAtoms': num_bridgehead_atoms,
            'NumSaturatedRings': num_saturated_rings,
            'NumAliphaticCarbocycles': num_aliphatic_carbocycles,
            'NumAliphaticHeterocycles': num_aliphatic_heterocycles,
            'NumAliphaticRingsCalc': num_aliphatic_rings_calc,
            'NumRadicalElectrons': num_radical_electrons,
            'HeavyAtomCount': heavy_atom_count,
            'LabuteASA': labute_asa,
            'Chi0': chi0,
            'Chi1': chi1,
            'Chi0v': chi0v,
            'Chi1v': chi1v,
            'NumSaturatedCarbocycles': num_saturated_carbocycles,
            'NumSaturatedHeterocycles': num_saturated_heterocycles,
            'NumAliphaticCarbocyclesDesc': num_aliphatic_carbocycles_desc,
            'NumAliphaticHeterocyclesDesc': num_aliphatic_heterocycles_desc,
        }
        
        return descriptors
    
    except Exception as e:
        # If any error occurs, return NaN for all descriptors
        print(f"Warning: Error calculating descriptors for {reactant_smiles}: {str(e)}")
        return {key: np.nan for key in get_descriptor_names()}


def find_reactive_olefin(mol):
    """
    Find the reactive C=C double bond in the molecule.
    Returns: (list of 2 atoms, bond object) or ([], None)
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
                return [atom1, atom2], bond
    
    return [], None


def calculate_descriptors_for_dataset(df, reactant_col='Reactant SMILES', product_col='Product SMILES'):
    """
    Calculate descriptors for an entire dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with reaction SMILES
    reactant_col : str
        Column name for reactant SMILES
    product_col : str
        Column name for product SMILES
    
    Returns:
    --------
    pd.DataFrame : DataFrame with calculated descriptors
    """
    
    all_descriptors = []
    
    for idx, row in df.iterrows():
        desc = get_mapped_olefin_descriptors(
            row[reactant_col],
            row[product_col]
        )
        all_descriptors.append(desc)
    
    descriptor_df = pd.DataFrame(all_descriptors)
    
    # Count NaN values
    nan_count = descriptor_df.isnull().any(axis=1).sum()
    if nan_count > 0:
        print(f" {nan_count} reactions have missing descriptors")
    
    return descriptor_df


def get_descriptor_names():
    """Return list of all descriptor names in order (68 total)"""
    return [
        # Custom olefin descriptors (38)
        'MolWt', 'LogP', 'TPSA',
        'NumSubstituents_C1', 'NumSubstituents_C2', 'TotalSubstitution',
        'TotalH_OnOlefin', 'H_OnC1', 'H_OnC2',
        'IsMonosubstituted', 'IsGeminal', 'IsDisubstituted',
        'IsTrisubstituted', 'IsTetrasubstituted',
        'IsE_Olefin', 'IsZ_Olefin', 'IsUnspecifiedStereo',
        'OlefinInRing', 'OlefinRingSize', 'OlefinAromatic',
        'CarbonSubstituents', 'AromaticSubstituents', 'HeteroSubstituents',
        'NearbyHeavyAtoms',
        'Conjugated',
        'ChiralCentersAdjacent', 'TotalStereocenters',
        'NumOH', 'NumCarbonyl', 'NumEster', 'NumAmine',
        'NumAromaticRings', 'NumAliphaticRings', 'NumRotatableBonds',
        'NumHDonors', 'NumHAcceptors', 'FractionCsp3',
        
        # Additional RDKit descriptors (30)
        'MolMR', 'MaxPartialCharge', 'MinPartialCharge',
        'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'NumValenceElectrons',
        'BalabanJ', 'BertzCT', 'HallKierAlpha',
        'Kappa1', 'Kappa2', 'Kappa3',
        'NumSpiroAtoms', 'NumBridgeheadAtoms', 'NumSaturatedRings',
        'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRingsCalc',
        'NumRadicalElectrons', 'HeavyAtomCount',
        'LabuteASA', 'Chi0', 'Chi1', 'Chi0v', 'Chi1v',
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
        'NumAliphaticCarbocyclesDesc', 'NumAliphaticHeterocyclesDesc'
    ]
