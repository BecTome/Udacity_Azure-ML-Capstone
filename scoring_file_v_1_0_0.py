# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"battery_power": pd.Series([0], dtype="int64"), "blue": pd.Series([0], dtype="int64"), "clock_speed": pd.Series([0.0], dtype="float64"), "dual_sim": pd.Series([0], dtype="int64"), "fc": pd.Series([0], dtype="int64"), "four_g": pd.Series([0], dtype="int64"), "int_memory": pd.Series([0], dtype="int64"), "m_dep": pd.Series([0.0], dtype="float64"), "mobile_wt": pd.Series([0], dtype="int64"), "n_cores": pd.Series([0], dtype="int64"), "pc": pd.Series([0], dtype="int64"), "px_height": pd.Series([0], dtype="int64"), "px_width": pd.Series([0], dtype="int64"), "ram": pd.Series([0], dtype="int64"), "sc_h": pd.Series([0], dtype="int64"), "sc_w": pd.Series([0], dtype="int64"), "talk_time": pd.Series([0], dtype="int64"), "three_g": pd.Series([0], dtype="int64"), "touch_screen": pd.Series([0], dtype="int64"), "wifi": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs/model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
