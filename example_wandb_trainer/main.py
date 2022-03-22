import numpy as np
from example_dvc_dataset import IrisDataset
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def run_multiple_experiments(dataset_path: str):
    dataset = IrisDataset(dataset_path=dataset_path)

    X_train = dataset.train.drop(columns=["category"]).to_numpy()
    X_test = dataset.test.drop(columns=["category"]).to_numpy()

    y_train = dataset.train["category"].to_numpy()
    y_test = dataset.test["category"].to_numpy()

    run_decision_tree_experiment(X_train, y_train, X_test, y_test)
    run_svc_experiment(X_train, y_train, X_test, y_test)



def run_decision_tree_experiment(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    clf = DecisionTreeClassifier(
        random_state=42, criterion="gini", splitter="best", max_depth=10
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metric = f1_score(y_test, y_pred, average="weighted")
    print(metric)

def run_svc_experiment(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
):
    clf = SVC(
        C=.1,random_state=42,kernel="rbf"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metric = f1_score(y_test, y_pred, average="weighted")
    print(metric)


if __name__ == "__main__":
    run_multiple_experiments(dataset_path="../example-dvc-dataset/dataset/iris.csv")
