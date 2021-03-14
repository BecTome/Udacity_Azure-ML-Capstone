from lightgbm import LGBMClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from azureml.core import Workspace, Dataset
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
# path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
# ds = TabularDatasetFactory.from_delimited_files(path)
#ds = ds.to_pandas_dataframe()
ws = Workspace.from_config()
ds = Dataset.get_by_name(ws, name='mobile_prices')

def clean_data(data):
    data['Vol_Dens'] = data['weight'] / (data['sc_w'] * data['sc_h'] * data['depth'])
    data['px_dens'] = data['px_height'] * data['x_width'] / (data['sc_w'] * data['sc_h'])
    data['talk_cons'] = data['battery_power'] / data['talk_time']
    return x_df, y_df


x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.1, help="Learning Rate for LGBMClassifier. Gradient Descent rate.")
    parser.add_argument('--max_depth', type=int, default=7, help="Depth of weak learners.")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees.")
    parser.add_argument('--num_leaves', type=int, default=80, help="Number of leaves.")

    args = parser.parse_args()

    run.log("Learning Rate:", np.float(args.lr))
    run.log("Max Depth:", np.int(args.max_depth))
    run.log("Max Estimators:", np.int(args.n_estimators))
    run.log("Number of Leaves:", np.int(args.num_leaves))

    model = LGBMClassifier(lr=args.lr, max_depth=args.max_depth, n_estimators=args.n_estimators, num_leaves=args.num_leaves).fit(x_train, y_train)
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()