"""Inference for HADES, a soft-voting ensemble scoring oral drug-likeness.

Featurisation (Mordred descriptors, ADMET-AI predictions and QED terms) is taken
verbatim from the authors' `HADES/featurize.py`. The score is the mean predicted
probability of the positive class across the five fitted classifiers, matching
`HADES_calcualor.py` upstream.

Checkpoints live in `model/checkpoints/` and are fetched from eosvc at build time;
they are not committed to the repository.
"""

import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from hades_src.featurize import Featurizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "checkpoints"))
DESCRIPTOR_COLUMNS_FILE = os.path.join(ROOT, "hades_src", "descriptor_columns.txt")

# The five ensemble members, averaged with equal weight.
MODEL_NAMES = [
    "XGBClassifier",
    "LGBMClassifier",
    "HistGradientBoostingClassifier",
    "CatBoostClassifier",
    "RandomForestClassifier",
]


def _load_descriptor_columns():
    """Return the 298 feature names, in the order the classifiers were fitted on.

    Upstream reads these from `train_test_saved/X_train.parquet` and drops its last
    column. Only the names are ever used, so they are stored here as plain text
    rather than shipping a 13 MB parquet of training data.
    """
    with open(DESCRIPTOR_COLUMNS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def _load_models():
    """Load the five fitted classifiers from the checkpoint directory."""
    return [joblib.load(os.path.join(CHECKPOINT_DIR, f"{n}.pkl")) for n in MODEL_NAMES]


def predict(smiles_list):
    """Score each SMILES for resemblance to an approved oral drug.

    Parameters
    ----------
    smiles_list : list of str
        Input molecules, one SMILES per entry.

    Returns
    -------
    numpy.ndarray
        One HADES score per input, in input order. Entries whose SMILES could not
        be parsed by RDKit are returned as NaN.
    """
    descriptor_columns = _load_descriptor_columns()

    # Featurise only the parseable molecules, but remember where they came from:
    # the upstream featuriser silently drops invalid SMILES, which would shift the
    # outputs out of alignment with the inputs.
    valid_positions, valid_smiles = [], []
    for position, smiles in enumerate(smiles_list):
        if Chem.MolFromSmiles(smiles) is not None:
            valid_positions.append(position)
            valid_smiles.append(smiles)

    scores = np.full(len(smiles_list), np.nan, dtype=float)
    if not valid_smiles:
        return scores

    features = Featurizer().featurize_many_smiles(valid_smiles)
    features.columns = features.columns.astype(str)
    features[descriptor_columns] = features[descriptor_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    probabilities = [
        model.predict_proba(features[descriptor_columns])[:, 1]
        for model in _load_models()
    ]
    scores[valid_positions] = np.mean(probabilities, axis=0)
    return scores
