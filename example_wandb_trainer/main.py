import logging
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from example_dvc_dataset import IrisDataset
from joblib import dump
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import wandb

logging.basicConfig(level=logging.INFO)


class ModelType(str, Enum):
    svc = "svc"
    decision_tree = "decision_tree"


def run_multiple_experiments(
    dataset_path: str = typer.Option(
        "../example-dvc-dataset/dataset/iris.csv",
        help="Path to iris csv dataset",
    ),
    wandb_entity: str = typer.Option(..., help="WandB user profile"),
    experiment_type: ModelType = typer.Option(
        ..., help="Experiment model type"
    ),
):
    dump_dir = Path(f"./tmp/{experiment_type.value}")
    dump_dir.mkdir(exist_ok=True, parents=True)

    wandb.init(
        project="experiment-tracking-example",
        entity=wandb_entity,
        config={"experiment_type": experiment_type},
        name=experiment_type.value,
    )

    logging.info("Loading dataset..")
    dataset = IrisDataset(dataset_path=dataset_path)

    X_train = dataset.train.drop(columns=["category"]).to_numpy()
    X_test = dataset.test.drop(columns=["category"]).to_numpy()

    y_train = dataset.train["category"].to_numpy()
    y_test = dataset.test["category"].to_numpy()
    labels = ["setosa", "versicolor", "virginica"]
    feature_names = dataset.train.drop(columns=["category"]).columns

    if experiment_type == ModelType.decision_tree:
        logging.info("Running decision tree experiment..")
        run_decision_tree_experiment(
            X_train, y_train, X_test, y_test, labels, feature_names, dump_dir
        )
    elif experiment_type == ModelType.svc:
        logging.info("Running svc experiment..")
        run_svc_experiment(
            X_train, y_train, X_test, y_test, labels, feature_names, dump_dir
        )
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")


def run_decision_tree_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    feature_names: pd.core.indexes.base.Index,
    dump_dir: Path,
):
    random_state = 42
    criterion = "gini"
    splitter = "best"
    max_depth = 10

    wandb.config.update(
        {
            "random_state": random_state,
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
        }
    )

    clf = DecisionTreeClassifier(
        random_state=random_state,
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    logging.info("Uploading report to WandB..")
    wandb.sklearn.plot_classifier(
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probas,
        labels,
        model_name="DecisionTree",
        feature_names=feature_names,
    )

    model_file = "decision_tree.joblib"
    model_dump_path = dump_dir / model_file

    logging.info("Serialize model to file using joblib..")
    dump(clf, model_dump_path)

    logging.info("Uploading model file to WandB..")
    wandb.save(str(model_dump_path),base_path=dump_dir)


def run_svc_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    feature_names: pd.core.indexes.base.Index,
    dump_dir: Path,
):
    random_state = 42
    C = 0.1
    kernel = "rbf"

    wandb.config.update(
        {
            "random_state": random_state,
            "C": C,
            "kernel": kernel,
        }
    )

    clf = SVC(C=C, random_state=random_state, kernel=kernel, probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    logging.info("Uploading report to WandB..")
    wandb.sklearn.plot_classifier(
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        y_probas,
        labels,
        model_name="SVC",
        feature_names=feature_names,
    )

    model_file = "svc.joblib"
    model_dump_path = dump_dir / model_file

    logging.info("Serialize model to file using joblib..")
    dump(clf, model_dump_path)

    logging.info("Uploading model file to WandB..")
    wandb.save(str(model_dump_path),base_path=dump_dir)


if __name__ == "__main__":
    typer.run(run_multiple_experiments)
