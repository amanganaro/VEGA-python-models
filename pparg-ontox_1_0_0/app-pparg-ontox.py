from sklearn.svm import SVC
import pandas as pd
import pickle
import argparse
import os
import numpy as np

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
parser = argparse.ArgumentParser(description='pparg svm assessment')

# Add an argument for input
parser.add_argument('--output', required=True, help='CSV result output file')
parser.add_argument('--input', required=True, help='CSV cddd descriptors file input')

# Parse the command-line arguments
args = parser.parse_args()

# Access the input value
input_file = args.input
output_file = args.output

# Load the pre-trained model from the file
file_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(file_dir, "models_ontox_pparg_svm")
model_filename = os.path.join(models_dir, "pparg_svm_cddd.pkl")
with open(model_filename, 'rb') as file:
    model_metadata = pickle.load(file)

# Extract the model and feature column names from the metadata
model = model_metadata['model']
model_columns = model_metadata['features']

# Prepare the test data
cddd_df = pd.read_csv(input_file)
cddd_df.drop(cddd_df.columns[[0, 1]], axis=1, inplace=True)

X_test = cddd_df.rename(columns=adjust_column_name)

# Remove any extra columns in the test data
X_test = X_test[model_columns]

#take only not null value
valid_mask = ~X_test.isnull().all(axis=1)
X_valid = X_test[valid_mask]
X_invalid = X_test[~valid_mask]

# Make predictions on the test data
y_pred = model.predict(X_valid)
y_pred_prob = model.predict_proba(X_valid)

# Map the predicted classes to their corresponding labels
predicted_classes = ['active' if pred == 0 else 'not active' for pred in y_pred]

# Create a placeholder for invalid rows
invalid_table = pd.DataFrame({
    'Predicted_Class': [""] * len(X_invalid),
    'Probability_Active': [np.nan] * len(X_invalid),
    'Probability_NotActive': [np.nan] * len(X_invalid)
}, index=X_invalid.index)

# Create a DataFrame with predicted classes and prediction probabilities
prediction_probabilities = pd.DataFrame(y_pred_prob, columns=["Probability_Active", "Probability_NotActive"])
predicted_classes_df = pd.DataFrame(predicted_classes, columns=["Predicted_Class"])

# Concatenate the predicted classes and prediction probabilities
valid_table = pd.concat([predicted_classes_df, prediction_probabilities], axis=1)

result_df = pd.concat([valid_table, invalid_table]).sort_index()

result_df.to_csv(output_file)
print("Complete prediction of pparg SVC model")