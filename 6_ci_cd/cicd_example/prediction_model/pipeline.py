import os
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression

import prediction_model.processing.preprocessing as pp
from prediction_model.config import config

classification_pipeline = Pipeline(
    [
        (
            "DomainProcessing",
            pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD),
        ),
        ("DropFeatures", pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ("LabelEncoder", pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ("LogTransform", pp.LogTransforms(variables=config.LOG_FEATURES)),
        ("LogisticClassifier", LogisticRegression(random_state=0)),
    ]
)
