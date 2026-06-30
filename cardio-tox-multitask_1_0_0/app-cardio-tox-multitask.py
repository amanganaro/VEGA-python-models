## inference

import pandas as pd
import pickle, os, argparse, subprocess, sys
import torch
# from model import multigate_multimodal

# rdkit
from rdkit import Chem

def check_smiles(smiles):
    try: Chem.MolFromSmiles(smiles)
    except: 
        print("invalid smiles.")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='cardiotoxicity assessment')

    # Add an argument for input
    parser.add_argument('--output', required=True, help='CSV output file')
    parser.add_argument('--input', required=True, help='CSV cddd descriptors file input')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the input value
    input_file = args.input
    output_file = args.output
    return {
        "input_file": input_file,
        "output_file": output_file
    }

def inference(models:dict, smiles, target_dict:dict, singlemol=True):

    # Prediction
    results_final = {}
    for key, model in models.items():
        results_final[key] = model.predict(target_dict[key])

    if singlemol: 
        return pd.DataFrame(pd.DataFrame(results_final, index=[smiles]))
    else: 
        return pd.DataFrame(pd.DataFrame([results_final])) #, index=smiles)

def import_models():

    "Import models find in folder"
    
    path = os.path.join(file_dir, "model", "multitask_cardiotox_model.pth")
    loaded_model = torch.load(path)

    # Set the model to evaluation mode (if using for inference)
    loaded_model.eval()

    thr_path = os.path.join(file_dir, "model", "thr.pkl")

    with open(thr_path, 'rb') as file:
         thr = pickle.load(file)

    return loaded_model, thr

def localOutlierFactor_applicability_domain(test):
        
        """
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor
        
        The anomaly score of each sample is called the Local Outlier Factor. 
        It measures the local deviation of the density of a given sample with respect to its neighbors. 
        It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. 
        More precisely, locality is given by k-nearest neighbors, whose distance is used to estimate the local density. 
        By comparing the local density of a sample to the local densities of its neighbors, one can identify 
        samples that have a substantially lower density than their neighbors. These are considered outliers.

        pfloat, default=2
        Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. 
        When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. 
        For arbitrary p, minkowski_distance (l_p) is used.

        NOTE:
        noveltybool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier detection (novelty=False). 
        Set novelty to True if you want to use LocalOutlierFactor for novelty detection. 
        In this case be aware that you should only use predict, decision_function and score_samples on new unseen data and not on the training set; 
        and note that the results obtained this way may differ from the standard LOF results."""
        
        clf_path = os.path.join(file_dir, "model", "clf.pkl")

        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

        return ['in' if i==1 else 'out' for i in clf.predict(test).tolist()]

if __name__ == "__main__":

    tasks_names = ['Apical cardiotoxicity',
                    'Aryl hydrocarbon receptor',
                    'Cardiomyocyte Myocardial Injury',
                    'Change Action Potential',
                    'Change in Inotropy',
                    'Change In Vasoactivity',
                    'Endothelial injury coagulation',
                    'hERG channels inhibitors',
                    'Increase mitochondrial dysfunction',
                    'Inhibition mitochondrial complexes',
                    'OxidativeStress',
                    'Valvular Injury Proliferation']

    file_dir=os.path.dirname(os.path.abspath(__file__))
    # ask smiles of chemicals to evaluate
    files = main()
    target_CDDD = pd.read_csv(files['input_file'])
    
    for smi in target_CDDD['smiles_preprocessed']:
        check_smiles(smi)

    # import models pipeline and descriptors
    print("[INFO]: Models import...")
    model, thr = import_models()
    data_test = torch.tensor(target_CDDD.drop(['smiles_preprocessed', 'smiles'], axis=1).values, dtype=torch.float32)
    x = {'CDDD': data_test}
    prediction = model(x)
    if prediction.shape[0] == 1:
        results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int)).T
        results.columns = tasks_names
    else:
        results = pd.DataFrame((torch.sigmoid(prediction) > thr).numpy().squeeze().astype(int), columns=tasks_names)

    results_ = pd.concat([target_CDDD.loc[:, ['smiles_preprocessed', 'smiles']], results], axis=1)
    results_['ApplicabilityDomain'] = localOutlierFactor_applicability_domain(target_CDDD.drop(['smiles_preprocessed', 'smiles'], axis=1).values)
    
    # remove prediction for the task that don't reach satisfactory results
    results_.drop(['Inhibition mitochondrial complexes'], axis=1, inplace=True)
    results_.to_csv(files['output_file'])