import xgboost
import pandas as pd
import pickle
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
parser = argparse.ArgumentParser(description='Fgfr4 xgboost assessment')

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
models_dir = os.path.join(file_dir, "models_ontox_fgfr4_xgboost")
model_filename = os.path.join(models_dir, "fgfr4_xgboost_cddd.pkl")
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Prepare the test data (assuming input_table_1 contains the features only)
cddd_df = pd.read_csv(input_file)
cddd_df.drop(cddd_df.columns[[0, 1]], axis=1, inplace=True)
X_test = cddd_df.rename(columns=adjust_column_name)

# Remove any extra columns in the test data
model_columns = loaded_model.get_booster().feature_names

X_test = X_test[model_columns]
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Make predictions on the test data
y_pred = loaded_model.predict(X_test)
y_pred_prob = loaded_model.predict_proba(X_test)

# Map the predicted classes to their corresponding labels
predicted_classes = ['active' if pred == 0 else 'not active' for pred in y_pred]

# Create a DataFrame with predicted classes and prediction probabilities for each class
predicted_classes_df = pd.DataFrame(predicted_classes, columns=["Predicted_Class"])
prediction_probabilities = pd.DataFrame(y_pred_prob, columns=["Probability_Active", "Probability_NotActive"])

# Concatenate the predicted classes and prediction probabilities
output_table_1 = pd.concat([predicted_classes_df, prediction_probabilities], axis=1)
output_table_1.to_csv(output_file)
print("Complete prediction of Fgfr4 XgBoost model")
