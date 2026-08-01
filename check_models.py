from sklearn.metrics import roc_auc_score

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict, MODELS

X, y, _ = load_dataset("credit")
s = make_split(y, seed=0)

for name in MODELS:
    p = fit_predict(name, X, y, s, seed=0)
    auc_test = roc_auc_score(y[s["test"]], p[s["test"], 1])
    auc_train = roc_auc_score(y[s["train"]], p[s["train"], 1])
    print(f"{name:<15} test AUC {auc_test:.4f}   train AUC {auc_train:.4f}")