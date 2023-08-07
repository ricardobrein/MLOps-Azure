# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# define functions
def main(args):
    # TO DO: enable autologging
    
    mlflow.autolog()

    # Load the data from the CSV file (assuming the data file is named 'data.csv')
    data_file = 'experimentation/data/diabetes-dev.csv'
    df = pd.read_csv(data_file)

    # Split the data using the split_data function
    X_train, X_test, y_train, y_test = split_data(df)

    # Train the model
    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)

    # Evaluate the model
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Log metrics and parameters with MLflow
    mlflow.log_metrics({'accuracy': acc, 'auc': auc})
    mlflow.log_param('test_size', 0.30)
    mlflow.log_param('random_state', 0)

    # Save the model
    mlflow.sklearn.log_model(model, 'model')



    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: agregar función para dividir datos en entrenamiento y prueba

def split_data(df):
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
