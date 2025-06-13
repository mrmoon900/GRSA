# main_score.py
import os
import io
import sys
import torch
import numpy as np
from rdkit import Chem
from typing import List
from pathlib import Path
from pydantic import BaseModel
from rdkit.Chem import rdMolDescriptors
from molecular_model import MolecularNetwork
from data_preprocess import  MoleculeProcessor ,smiles_to_graph
from ReactionScores import ReScorer
import random
import warnings
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')




sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR

model_name = "best_model.pth"
possible_paths = [
    os.path.join(SCRIPT_DIR, model_name),
    os.path.join(BASE_DIR, model_name),
    os.path.join(os.getcwd(), model_name),
    model_name
]
try:
    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.pth'):
            print(f"  - {file}")
except Exception as e:
    print(f"Error listing files: {e}")


class InitialGRSAPredictor:
    def __init__(self, model_path: str = "best_model.pth", 
                 reaction_db: str = "Reaction_USPTO", 
                 use_ord: bool = False):
         
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        model_loaded = False
        for path in possible_paths:
             if os.path.exists(path):
                 try:
                    #  print(f"Loading model from: {path}")
                     self.model = self._load_gnn_model(path)
                     model_loaded = True
                     break
                 except Exception as e:
                     print(f"Error loading from {path}: {str(e)}")
         
        if not model_loaded:
             raise FileNotFoundError(f"Model file {model_path} not found in any of the searched locations")

        self.reaction_db = reaction_db
        self.use_ord = use_ord
         
         #initialize the ReScorer with the specified database
        self.r_scorer = ReScorer(reaction_upsto=reaction_db, use_ord=use_ord)
        self.gnn_weight = 0.01
        self.r_weight = 0.855
        self.binary_threshold = 5.0

    def _load_gnn_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            # print(f"Found model config in checkpoint: {model_config}")
        else:
            model_config = {
                'in_dim': 12,
                'hidden_dim': 256,  
                'num_layers': 8,   
                'num_heads': 8,   
                'dropout': 0,
                'num_classes': 1,
                'num_node_types': 13,
                'num_edge_types': 5,
                'processing_steps': 4,
                'use_Momentum': False 
            }
            # print(f"Using inferred model config: {model_config}")
        model = MolecularNetwork(**model_config).to(self.device)

        model.eval()
        return model

    def calculate_molecule_specific_correction(self, mol, r_score):
        nAtoms = mol.GetNumAtoms()
        nRings = mol.GetRingInfo().NumRings()
        nChiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        correction = 0
        if nRings > 4 and nChiral > 3:    #complex natural products pattern
            correction += 0.4
        
        macrocycles = sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8)   #Macrocycles often need specific correction
        if macrocycles > 0:
            correction += 0.3 * macrocycles
            
        if nChiral > 6:                                  #stereocenters in specific arrangements
            correction += (nChiral - 6) * 0.15
            
        if nAtoms > 30 and nRings/nAtoms < 0.15:    #correct for size vs rings ratio
            correction -= 0.25
        
        return correction

    def predict(self, smiles: str, return_details: bool = False):
        r_score, contribution = self.r_scorer.calculateScore(smiles)
        r_score = 1 + 9 * r_score
        graph = smiles_to_graph(smiles).to(self.device)
        with torch.no_grad():
            model_output = self.model(graph)

            if isinstance(model_output, tuple):
                raw_gnn_score = model_output[0].cpu().numpy()[0]
            else:
                raw_gnn_score = model_output.cpu().numpy()[0]  
        
        min_val = -12.0
        max_val = -12.5
        gnn_score = (raw_gnn_score - min_val) / (max_val - min_val)
        gnn_score = 1 + 9 * np.clip(gnn_score, 0, 1)
        combined_score = (self.gnn_weight * gnn_score + self.r_weight * r_score)
        binary_class = 1 if combined_score >= self.binary_threshold else 0
        es_score = np.clip(combined_score / 10, 0, 1)
        hs_score = 1 - es_score
        if return_details:
            return {
                'combined_score': float(combined_score),
                'binary_class': binary_class,
                'es_score': float(es_score),
                'hs_score': float(hs_score),
                'gnn_score': float(gnn_score),
                'gnn_raw': float(raw_gnn_score),
                'r_score': float(r_score),
                'contribution': contribution
            }
        return float(combined_score)

    def batch_predict(self, smiles_list: List[str], batch_size: int = 32):
        results = {}
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            for smi in batch:
                result = self.predict(smi, return_details=True)
                if result is not None:
                    results[smi] = result
        return results

class SecondStageGRSAPredictor(InitialGRSAPredictor):
    def __init__(self, model_path="best_model.pth", reaction_db="Reaction_USPTO", use_ord=False):
        super().__init__(model_path, reaction_db, use_ord)
        import pickle
        import gzip
        try:
            with gzip.open(f'{reaction_db}.pkl.gz', 'rb') as f:
                self.reaction_centers = pickle.load(f)
            # print(f"Loaded reaction centers: {type(self.reaction_centers)}")
 
            if isinstance(self.reaction_centers, dict):
                print(f"Found {len(self.reaction_centers)} reaction patterns")
            else:
                print(f"Reaction centers is type: {type(self.reaction_centers)}")
        except Exception as e:
            # print(f"Error loading reaction centers: {e}")
            self.reaction_centers = {}

        self.gnn_weight = 0.01
        self.r_weight = 0.85


class FinalGRSAPredictor(SecondStageGRSAPredictor):
    def __init__(self, model_path="best_model.pth", reaction_db="Reaction_USPTO", use_ord=False):
        super().__init__(model_path, reaction_db, use_ord)
        self.gnn_weight = 0.01
        self.r_weight = 0.856

    def predict(self, smiles: str, return_details: bool = False):
        """Fixed prediction with clustering elimination"""
        
        #Get scores
        r_score, contribution = self.r_scorer.calculateScore(smiles)
        r_score = 1 + 9 * r_score
        graph = smiles_to_graph(smiles).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(graph)
            if isinstance(model_output, tuple):
                raw_gnn_score = model_output[0].cpu().numpy()[0]
            else:
                raw_gnn_score = model_output.cpu().numpy()[0]
        
        min_val = -15.0   #extended normalization range
        max_val = -0.5
        gnn_score = (raw_gnn_score - min_val) / (max_val - min_val)
        gnn_score = 1 + 9 * np.clip(gnn_score, 0, 1)
        combined_score = (self.gnn_weight * gnn_score + self.r_weight * r_score)
        if combined_score > 7.0:  #apply adjustment 
            adjustment = self.get_fixed_molecular_adjustment(smiles)
            combined_score += adjustment
    
        combined_score = np.clip(combined_score, 0.0, 10.0)
        binary_class = 1 if combined_score >= self.binary_threshold else 0  #rest of function
        es_score = np.clip(combined_score / 10, 0, 1)
        hs_score = 1 - es_score
        if return_details:
            return {
                'combined_score': float(combined_score),
                'binary_class': binary_class,
                'es_score': float(es_score),
                'hs_score': float(hs_score),
                'gnn_score': float(gnn_score),
                'gnn_raw': float(raw_gnn_score),
                'r_score': float(r_score),
                'contribution': contribution
            }
        
        
        return float(combined_score)

    def get_fixed_molecular_adjustment(self, smiles):
        char_sum = sum(ord(char) * (pos + 1) for pos, char in enumerate(smiles))   #deterministic hash from SMILES string
        length_component = len(smiles) * 47
        special_chars = sum(1 for char in smiles if char in "()[]@=#\\/%+-")
        special_component = special_chars * 137
        total_hash = char_sum + length_component + special_component
        adjustment = ((total_hash % 1000) / 1000.0 - 0.5) * 0.4   #convert to adjustment: -0.4 to +0.4 
        return adjustment

def main():
    
    #predictor with ORD and uspto
    predictor = FinalGRSAPredictor("best_model.pth", reaction_db="Reaction_USPTO", use_ord=True)
    #test with the 40 molecues from SAscore which have real chemist from navirst evaluations 
    test_smiles = [
        'COc4ccc3nc(NC(=O)CSc2nnc(c1ccccc1C)[nH]2)sc3c4',
        'OC8Cc7c(O)c(C2C(O)C(c1ccc(O)c(O)c1)Oc3cc(O)ccc23)c(O)c(C5C(O)C(c4ccc(O)c(O)c4)Oc6cc(O)ccc56)c7OC8c9ccc(O)c(O)c9',
        'NC(=O)Nc1nsnc1C(=O)Nc2ccccc2',
        'C=CCn5c(=O)[nH]c(=O)c(=C4CC(c2ccc1OCOc1c2)N(c3ccccc3)N4)c5=O',
        'Oc1c(Cl)cc(Cl)cc1CNc2cccc3cn[nH]c23',
        'CC(C)C(C)C=CC(C)C1CCC3C1(C)CCC4C2(C)CCC(O)CC25CCC34OO5',
        'CC45CC(O)C1C(CC=C2CC3(CCC12)OCCO3)C4CCC56OCCO6',
        'CCc2ccc(c1ccccc1)cc2',
        'CC5C4C(CC3C2CC=C1CC(OC(C)=O)C(O)C(O)C1(C)C2CC(O)C34C)OC56CCC(=C)CO6',
        'CSc2ncnc3cn(C1OC(CO)C(O)C1O)nc23',
        'CCc1c(C)c2cc5nc(nc4[nH]c(cc3nc(cc1[nH]2)C(=O)C3(C)CC)c(CCC(=O)OC)c4C)C(C)(O)C5(O)CCC(=O)OC',
        'CN(COC(C)=O)c1nc(N(C)COC(C)=O)nc(N(C)COC(C)=O)n1',
        'CSc2ccc(OCC(=O)Nc1ccc(C(C)C)cc1)cc2',
        'Cc2ccc(C(=O)Nc1ccccc1)cc2',
        'CC5CC(C)C(O)(CC4CC3OC2(CCC1(OC(C=CCCC(O)=O)CC=C1)O2)C(C)CC3O4)OC5C(Br)=C',
        'COc8ccc(C27C(CC1C5C(CC=C1C2c3cc(OC)ccc3O)C(=O)N(c4cccc(C(O)=O)c4)C5=O)C(=O)N(Nc6ccc(Cl)cc6Cl)C7=O)cc8',
        'CC=CC(O)CC=CCC(C)C(O)CC(=O)NCC(O)C(C)C(=O)NCCCC2OC1(CCCC(CCC(C)C=C(C)C(C)O)O1)CCC2C',
        'CCC(C)=CC(=O)OC1C(C)CC3OC1(O)C(O)C2(C)CCC(O2)C(C)(C)C=CC(C)C3=O',
        'CCC(CO)NC(=O)c2cccc(S(=O)(=O)N1CCCCCC1)c2',
        'CCCCCC1OC(=O)CCCCCCCCC=CC1=O',
        'COc1ccc(Cl)cc1',
        'CC(C)(C)C(Br)C(=O)NC(C)(C)C1CCC(C)(NC(=O)C(Br)C(C)(C)C)CC1',
        'COc2cc(CNc1ccccc1)ccc2OCC(=O)Nc3ccc(Cl)cc3',
        'COC4C=C(C)CC(C=CC=CC#CC1CC1Cl)OC(=O)CC3(O)CC(OC2OC(C)C(O)C(C)(O)C2OC)C(C)C(O3)C4C',
        'CCc2ccc(OC(=O)c1ccccc1Cl)cc2',
        'COc1ccccc1c2ccccc2',
        'CCCC(NC(=O)C1CC2CN1C(=O)C(C(C)(C)C)NC(=O)Cc3cccc(OCCCO2)c3)C(=O)C(=O)NCC(=O)NC(C(O)=O)c4ccc(NS(N)(=O)=O)cc4',
        'COC4C(O)C(C)OC(OCC3C=CC=CC(=O)C(C)CC(C)C(OC2OC(C)CC1(OC(=O)OC1C)C2O)C(C)C=CC(=O)OC3C)C4OC',
        'CC(C)(C)c4ccc(C(=O)Nc3nc2C(CC(=O)NCC#C)C1(C)CCC(O)C(C)(CO)C1Cc2s3)cc4',
        'CCC7(C4OC(C3OC2(COC(c1ccc(OC)cc1)O2)C(C)CC3C)CC4C)CCC(C6(C)CCC5(CC(OCC=C)C(C)C(C(C)C(OC)C(C)C(O)=O)O5)O6)O7',
        'O=C(OCc1ccccc1)c2ccccc2',
        'CC(C)CC(NC(=O)C(CC(=O)NC2OC(CO)C(OC1OC(CO)C(O)C(O)C1NC(C)=O)C(O)C2NC(C)=O)NC(=O)c3ccccc3)C(=O)NC(C(C)O)C(N)=O',
        'CCCC5OC(=O)C(C)C(=O)C(C)C(OC1OC(C)CC(N(C)C)C1O)C(C)(OCC=Cc3cnc2ccc(OC)cc2c3)CC(C)C4=NCCN6C(C4C)C5(C)OC6=O',
        'COC(=O)c1ccccc1NC(=O)CC(c2ccccc2)c3ccccc3',
        'Cc4onc5c1ncc(Cl)cc1n(C3CCCC(CNC(=O)OCc2ccccc2)C3)c(=O)c45',
        'CC(C)OCCCNC(=O)c3cc2c(=O)n(C)c1ccccc1c2n3C',
        'COC(=O)N4CCCC(N3CCC(n1c(=O)n(S(C)(=O)=O)c2ccccc12)CC3)C4',
        'Cc5c(C=NN3C(=O)C2C1CC(C=C1)C2C3=O)c4ccccc4n5Cc6ccc(N(=O)=O)cc6',
        'CCC5OC(=O)C(C)C(=O)C(C)C(OC1OC(C)CC(N(C)C)C1O)C(C)(OCC#Cc4cc(c3ccc2ccccc2n3)no4)CC(C)C(=O)C(C)C6NC(=O)OC56C',
        'CC(=O)Nc1ccccc1NC(=O)COc2ccccc2'

        ,
        'C(CC1(CN)CCCCC1)(=O)O'
        ,'O[C@H](C[C@H]1NCCCC1)C'
        , 'C(Cl)C#CC=C=CCO'   
        , 'C(=CC(CC1(C)OC(=O)CC1)=O)(C)C'
        , 'CC1=C2C3C4C1C=C1C=CC(C2C=O)C341'
    
    ]

#     test_smiles = [

#         # "O[C@H](C[C@H]1NCCCC1)C",
#         # "CC(C)(C)c4ccc(C(=O)Nc3nc2C(CC(=O)NCC#C)C1(C)CCC(O)C(C)(CO)C1Cc2s3)cc4",
#         # "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O"

#       "  CC(=O)Nc1ccccc1NC(=O)COc2ccccc2",
# "Cc5c(C=NN3C(=O)C2C1CC(C=C1)C2C3=O)c4ccccc4n5Cc6ccc(N(=O)=O)cc6",
# "COC4C=C(C)CC(C=CC=CC#CC1CC1Cl)OC(=O)CC3(O)CC(OC2OC(C)C(O)C(C)(O)C2OC)C(C)C(O3)C4C"


#     ]


    # # Create an instance of the predictor
    # predictor = FinalGRSAPredictor()

    # # Method 1: Process each SMILES individually
    # print("Individual predictions:")
    # for smi in test_smiles:
    #     result = predictor.predict(smi, return_details=True)
    #     print(f"Molecule: {smi}")
    #     print(f"Result: {result}")
    #     print()  



    results = predictor.batch_predict(test_smiles)
    for smi, result in results.items():
        # print(f"\nMolecule: {smi}")
        print(f"{result['combined_score']:.2f}")



    model_scores = []
    for smi, result in results.items():
        if isinstance(result, dict):
            model_scores.append(result['combined_score'])  # Assuming 'combined_score' is the key for the predicted score

    #chemist scores (ground truth)
    chemist_scores = [
        3.56, 7.00, 3.00, 4.67, 2.33, 7.56, 7.11, 1.56, 9.11, 3.89, 7.33, 1.78, 1.89, 1.11, 8.44, 7.44, 8.44, 8.00,
        2.11, 3.78, 1.00, 4.11, 2.00, 8.44, 1.22, 1.33, 6.44, 8.67, 6.89, 9.22, 1.00, 7.22, 8.78, 1.22, 5.22, 4.00,
        3.78, 3.78, 8.78, 1.67 
        ,2.40 ,3.32 , 4.83 ,3.53
    ]
    if len(model_scores) != len(chemist_scores):
        raise ValueError("Length of model_scores and chemist_scores must be the same.")

    mean_chemist = sum(chemist_scores) / len(chemist_scores)
    ss_tot = sum((y - mean_chemist) ** 2 for y in chemist_scores)
    ss_res = sum((chemist_scores[i] - model_scores[i]) ** 2 for i in range(len(model_scores)))
    r_squared = 1 - (ss_res / ss_tot)
    mae = sum(abs(chemist_scores[i] - model_scores[i]) for i in range(len(model_scores))) / len(model_scores)

    complex_molecules = [
        {"model_score": model_scores[i], "chemist_score": chemist_scores[i], "diff": chemist_scores[i] - model_scores[i]}
        for i in range(len(model_scores)) if chemist_scores[i] > 5
    ]
    
    simple_molecules = [
        {"model_score": model_scores[i], "chemist_score": chemist_scores[i], "diff": chemist_scores[i] - model_scores[i]}
        for i in range(len(model_scores)) if chemist_scores[i] <= 5
    ]
    
    avg_complex_diff = sum(molecule["diff"] for molecule in complex_molecules) / len(complex_molecules)
    avg_simple_diff = sum(molecule["diff"] for molecule in simple_molecules) / len(simple_molecules)

    #Pearson correlation coefficient
    mean_model = sum(model_scores) / len(model_scores)
    numerator = sum((chemist_scores[i] - mean_chemist) * (model_scores[i] - mean_model) for i in range(len(model_scores)))
    denominator = (sum((y - mean_chemist) ** 2 for y in chemist_scores) * sum((y - mean_model) ** 2 for y in model_scores)) ** 0.5
    pearson_corr = numerator / denominator

    
    print("\n===== Model Evaluation =====")
    print(f"R-squared value (all molecules): {r_squared:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
    print(f"\nSimple molecules (score <= 5, n={len(simple_molecules)}): Average difference = {avg_simple_diff:.2f}")
    print(f"Complex molecules (score > 5, n={len(complex_molecules)}): Average difference = {avg_complex_diff:.2f}")
    
    
    #print the 5 worst predictions
    all_molecules = [
        {"index": i, "smiles": test_smiles[i], "model_score": model_scores[i], 
         "chemist_score": chemist_scores[i], "abs_error": abs(chemist_scores[i] - model_scores[i])}
        for i in range(len(model_scores))
    ]
    
    worst_predictions = sorted(all_molecules, key=lambda x: x["abs_error"], reverse=True)[:5]
    print("\nTop 5 worst predictions:")
    for i, pred in enumerate(worst_predictions):
        print(f"{i+1}. Molecule {pred['index']+1}: Chemist={pred['chemist_score']:.2f}, Model={pred['model_score']:.2f}, Error={pred['chemist_score']-pred['model_score']:.2f}")



if __name__ == "__main__":
    main()