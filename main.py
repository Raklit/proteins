import os
import collections

import click
import joblib

import numpy as np
import pandas as pd

import multimodel
import featurescalculator
from sklearn.model_selection import train_test_split

program_name = "program"
version_name = "0.0.1"
github_link = "github.com/raklit/proteins"

__version__ = version_name

def read_csv(src : str, is_prediction : bool = False, is_compare : bool = False):
    required_columns = ["id", "input", "target"]
    if is_prediction:
        required_columns.remove("target")
    if is_compare:
        required_columns.remove("input")

    result = None
    df = pd.read_csv(src)
    for column in required_columns:
        if column not in list(df.columns):
            raise ValueError(f"CSV file \"{src}\" must contains \"{column}\" column")
    result = df
    return result

def save_model(src : str, model : object):
    if os.path.exists(src):
        os.remove(src)
    joblib.dump(model, filename=src,compress=True)

def load_model(src : str):
    model = joblib.load(src)
    return model

@click.group()
@click.version_option(__version__)
def cli():
    pass

@cli.command()
@click.option("--src", type=click.Path(), default="train.csv", help="Source of training dataset")
@click.option("--mdl", type=click.Path(), default="model.dat", help="Distanation of model's file")
def train(src,mdl):
    """Train models on data in src and save model info into mdl."""
    calculator = featurescalculator.FeaturesCalculator()
    model = multimodel.MultiModel(features = calculator.features)
    df = read_csv(src,is_prediction=False,is_compare=False)
    inputs, targets = df["input"].to_numpy(), df["target"].to_numpy()
    model.fit(inputs, targets)
    save_model(mdl, model)

@cli.command()
@click.option("--src", type=click.Path(), default="test.csv", help="Source of testing dataset")
@click.option("--mdl", type=click.Path(), default="model.dat", help="Source of model's info")
@click.option("--dst", type=click.Path(), default="results.csv", help="Distanation of results")
def predict(src, mdl, dst):
    """Predict targets of inputs in src with model from mdl and save results into dst."""
    df = read_csv(src,is_prediction=True,is_compare=False)
    ids, inputs = df["id"].to_numpy(), df["input"].to_numpy()
    model = load_model(mdl)
    targets = model.predict(inputs)
    df = pd.DataFrame.from_dict({"id" : ids, "input" : inputs, "target" : targets})
    if os.path.exists(dst):
        os.remove(dst)
    df.to_csv(dst)

@cli.command()
@click.option("--src", type=click.Path(), default="train.csv", help="Source of dataset")
@click.option("--test_size", type=float, default=0.5,help="Test size percentage")
@click.option("--mdl", type=click.Path(),default="", help="If not empty save model to mdl path")
def crossvalidation(src, test_size, mdl):
    """Split dataset src on two parts: one for training and another for testing"""
    calculator = featurescalculator.FeaturesCalculator()
    model = multimodel.MultiModel(features = calculator.features)
    df = read_csv(src,is_prediction=False,is_compare=False)
    inputs, targets = df["input"].to_numpy(), df["target"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size, random_state=0)
    model.fit(X_train, y_train)
    if mdl != "":
        save_model(mdl, model)
    acc = model.score(X_test, y_test) * 100
    click.echo(f"Accuracy: {acc:.2f} %")

@cli.command()
@click.option("--src", type=click.Path(), default="test.csv", help="Source of testing dataset")
@click.option("--mdl", type=click.Path(), default="model.dat", help="Source of model's info")
def score(src, mdl):
    """Return accuracy of mdl's predictions on src dataset"""
    df = read_csv(src,is_prediction=False,is_compare=False)
    inputs, targets = df["input"].to_numpy(), df["target"].to_numpy()
    model = load_model(mdl)
    acc = model.score(inputs, targets) * 100
    click.echo(f"Accuracy: {acc:.2f} %")

if __name__ == '__main__':
    cli()