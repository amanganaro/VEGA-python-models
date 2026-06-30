# import pypmml
from pypmml import Model
import pandas as pd
import argparse
import os

def adjust_column_name(col):
    if col.startswith('cddd_'):
        try:
            # Extract number, subtract 1, convert to string
            num = int(col.replace('cddd_', '')) - 1
            return str(num)
        except ValueError:
            pass
    return col  # leave unchanged if it doesn't match

#define the input and output file
# Set up argument parser
parser = argparse.ArgumentParser(description='Vgsc gb assessment')

# Add an argument for input
parser.add_argument('--output', required=True, help='CSV result output file')
parser.add_argument('--input', required=True, help='CSV cddd descriptors file input')

# Parse the command-line arguments
args = parser.parse_args()

# Access the input value
input_file = args.input
output_file = args.output

# Specify the path to your PMML file
file_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(file_dir, "models_ontox_vgsc_gb")
pmml_file = os.path.join(models_dir, "vgsc_gb_cddd.pmml")

# Create an instance of the Model class and load the PMML file
model = Model.fromFile(pmml_file)

# Load test set and prepare SMILES
test_df = pd.read_csv(input_file)
test_df.drop(test_df.columns[[0, 1, 2]], axis=1, inplace=True)
test_df_renamed = test_df.rename(columns=adjust_column_name)

results = []

for index, row in test_df_renamed.iterrows():
    prediction = model.predict(row)
    res_values = list(prediction.values)

    if prediction.get("P (active)", 0) > 0.5:
        res_values.append("active")
    else:
        res_values.append("not active")

    results.append([index] + res_values)

result_df = pd.DataFrame(results, columns=["Index"] + list(prediction.index) + ["predicted_Category"])

result_df.rename(columns={
    'predicted_Category': 'Predicted_Class', 
    'P (active)': 'Probability_Active',
    'P (not active)': 'Probability_NotActive',
}, inplace=True)    

# make prediction using the loaded molecules

result_df.to_csv(output_file)
print("Complete prediction of Vgsc gb PMML model")