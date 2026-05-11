import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import (
    load_dataset,
    load_pipeline,
    separate_data,
)

classification_pipeline = load_pipeline(config.MODEL_NAME)

# def generate_predictions(data_input):
#     data = pd.DataFrame(data_input)
#     pred = classification_pipeline.predict(data[config.FEATURES])
#     output = np.where(pred==1,'Approved','Not Approved')
#     result = {"prediction":output}
#     return result


def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    X, y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    output = np.where(pred == 1, "Approved", "Not Approved")
    print(output)
    return output


if __name__ == "__main__":
    generate_predictions()
