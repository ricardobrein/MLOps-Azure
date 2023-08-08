# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Define split_data function
def split_data(df):
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    data = {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}
    }
    return data

# Define train_model function
def train_model(reg_rate, X_train, y_train):
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model

# Define main function
def main(args):
    mlflow.autolog()

    # Load data from CSV file
    data_file = 'experimentation/data/diabetes-dev.csv'
    df = pd.read_csv(data_file)

    # Split data using split_data function
    data = split_data(df)
    X_train, y_train = data['train']['X'], data['train']['y']

    # Train the model
    model = train_model(args.reg_rate, X_train, y_train)

    # Log metrics and parameters with MLflow
    mlflow.log_param('test_size', 0.30)
    mlflow.log_param('random_state', 0)

    # Save the model
    mlflow.sklearn.log_model(model, 'model')

# Define get_csvs_df function
def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

# Define parse_args function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    args = parser.parse_args()
    return args

# Run script
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    
    args = parse_args()
    main(args)
    
    print("*" * 60)
    print("\n\n")
