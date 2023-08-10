import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn


def split_data(df):
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin',
            'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=0)
    data = {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}
        }
    return data


def train_model(reg_rate, X_train, y_train):
    model = LogisticRegression(C=1, solver="liblinear")
    model.fit(X_train, y_train)
    return model

# Evaluacion del rendimiento del modelo y registro de metricas

# def evaluate_model(model, X, y):
#    y_pred = model.predict_proba(X)[:, 1]
#    roc_auc = roc_auc_score(y, y_pred)
#    fpr, tpr, _ = roc_auc_score(y, y_pred)


def main(args):
    mlflow.autolog()
    data_file = args.training_data
    df = pd.read_csv(data_file)
    data = split_data(df)
    X_train, y_train = data['train']['X'], data['train']['y']

    model = train_model(args.reg_rate, X_train, y_train)

    mlflow.log_param('test_size', 0.30)
    mlflow.log_param('random_state', 0)

    mlflow.sklearn.log_model(model, 'diabetes_model')
    mlflow.sklearn.log_model(model, 'model')

    # Evaluación del modelo y registro de métricas
    X_test, y_test = data['test']['X'], data['test']['y']
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)

    mlflow.log_metric('roc_auc', roc_auc)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No hay ningun CSV: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate",
                        dest='reg_rate',
                        type=float, default=0.01)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60)
    print("\n\n")
