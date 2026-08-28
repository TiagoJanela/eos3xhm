# imports
import sys

import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

from hades_predict import predict

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]


# my model: SMILES -> HADES oral drug-likeness score
def my_model(smiles_list):
    return predict(smiles_list)


# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
assert len(smiles_list) == len(outputs)

# single output column, matching run_columns.csv
header = ["hades_score"]

# write output in a .csv file
write_out(outputs.reshape(-1, 1), header, output_file, np.float32)
