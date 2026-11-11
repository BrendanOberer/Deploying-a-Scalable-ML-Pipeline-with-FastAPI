import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference, train_model


def test_one():
    # Make sure training returns a a random forest classifier
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_two():
    # Make sure inference returns expected prediction number
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})


def test_three():
    # Make sure that classification metrics are correctly calculated
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.6666666667)
