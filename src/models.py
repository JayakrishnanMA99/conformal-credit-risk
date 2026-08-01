"""Base models. Every fit function returns predicted probabilities."""

import numpy as np


def _fit_lightgbm(X_train, y_train, seed):
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    m.fit(X_train, y_train)
    return m


def _fit_random_forest(X_train, y_train, seed):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m


def _fit_logistic(X_train, y_train, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    m.fit(X_train, y_train)
    return m


def _fit_mlp(X_train, y_train, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    )
    m.fit(X_train, y_train)
    return m


MODELS = {
    "lightgbm": _fit_lightgbm,
    "random_forest": _fit_random_forest,
    "logistic": _fit_logistic,
    "mlp": _fit_mlp,
}


def fit_predict(name, X, y, splits, seed):
    """Train on the train split, predict probabilities everywhere.

    Returns probs, shape (n_rows, 2). Column j is P(class j).
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")

    tr = splits["train"]
    model = MODELS[name](X[tr], y[tr], seed)

    probs = model.predict_proba(X)

    assert probs.shape == (len(y), 2), "expected two-column probabilities"
    assert np.allclose(probs.sum(axis=1), 1.0), "rows must sum to 1"

    return probs