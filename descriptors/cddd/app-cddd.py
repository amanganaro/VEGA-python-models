import pandas as pd
import sys
import os
import argparse
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles

# Number of rows the encoder tensor must be a multiple of for the result to be reproducible.
#
# The encoder is not bit-reproducible between processes: every process falls, at start-up, into one
# of exactly two states, and in one of them the same molecules come back with descriptors differing
# by about one unit in the last place of a float32 (measured: median 1.2e-07, at most 9.4e-07, on
# some 430 of the 512 components). Inside a single process the calculation is perfectly
# deterministic, so the difference only shows up between runs, and it is enough to move the output
# of a model: up to 0.0003 on a model with continuous output and up to 0.053 on one that votes over
# its members.
#
# The two states agree as soon as the tensor has at least 128 rows: from that size on, the common
# state produces exactly the value that the rarer one produces for every size. Measured sizes of 1,
# 10, 16, 32, 64, 96 and 128 rows: only 128 and above give that value, and it is also the value the
# anomalous process was observed to return. Filling up to a multiple of 128 therefore leaves the
# encoder only tensors of 256 or 128 rows, both reproducible.
#
# The cost is paid on short lists: a single molecule goes from 0.30 s to 2.43 s of encoder, against
# about 4.7 s that the process spends anyway starting Python and loading the model. Filling up to
# 256 instead would cost 5.31 s for no further benefit.
ENCODER_ROW_MULTIPLE = 128

# Molecule used to fill the tensor. It has to survive cddd's own preprocessing, which rejects
# anything with 3 heavy atoms or fewer (filter_smiles in cddd/preprocessing.py), so methane, ethane
# and propane are silently dropped and never reach the encoder; butane is the smallest that passes.
# It also has to be short, because each slice is padded to the length of its longest sequence: a
# short filler leaves that length, and with it the descriptors of the real molecules, untouched.
PADDING_SMILES = "CCCC"


def main():
    parser = argparse.ArgumentParser(description='Continuous data driven descriptors calculation')
    parser.add_argument('--output', required=True, help='Output directory of files')
    parser.add_argument('--input', required=True, help='CSV smiles file input')
    args = parser.parse_args()
    file_location = args.input
    output_dir = args.output

    print("Start calculating cddd descriptors")
    input_df = pd.read_csv(file_location)
    input_df["smiles_preprocessed"] = input_df.smiles.map(preprocess_smiles)
    
    #####input_df = input_df.dropna()
    smiles_list = input_df["smiles_preprocessed"].tolist()
    filtered_smiles_list = input_df.dropna()["smiles_preprocessed"].tolist()
    non_nan_mask = ~pd.isnull(smiles_list)

    # Process the non-NaN smiles with the model. The filler molecules are appended here, after the
    # rejected ones have been dropped, because this is the only point where the number of molecules
    # that actually reach the encoder is known. Their rows are discarded right away: the encoder
    # keeps the order of its input, so the real molecules are the first ones.
    real_count = len(filtered_smiles_list)
    padding = (ENCODER_ROW_MULTIPLE - real_count % ENCODER_ROW_MULTIPLE) % ENCODER_ROW_MULTIPLE
    padded_smiles_list = filtered_smiles_list + [PADDING_SMILES] * padding

    inference_model = InferenceModel()
    filtered_smiles_embedding = inference_model.seq_to_emb(padded_smiles_list)[:real_count]

    # Create the final embeddings list, filling NaN positions with None
    smiles_embedding = [[]] * len(smiles_list)
    j = 0
    for i, is_non_nan in enumerate(non_nan_mask):
        if is_non_nan:
            smiles_embedding[i] = filtered_smiles_embedding[j]
            j += 1

    if len(filtered_smiles_embedding) > 0:
        embedding_df = pd.DataFrame(smiles_embedding,
                                columns=[f'cddd_{i+1}' for i in range(len(filtered_smiles_embedding[0]))])
    else:
        embedding_df = pd.DataFrame(smiles_embedding)

    output_df = pd.concat([input_df.reset_index(drop=True), embedding_df], axis=1)

    for index, row in output_df.iterrows():
        file_name = f"{index}.csv"
        file_path = os.path.join(output_dir, file_name)
        row.to_frame().T.to_csv(file_path, index=False)

    print("Finish calculating cddd descriptors")

if __name__ == "__main__":
    main()