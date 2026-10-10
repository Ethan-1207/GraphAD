# -*- coding: utf-8 -*-
"""传统机器学习基线：LR / RF / SVM（scikit-learn 管线：中位数填补+标准化）。"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


def make_baseline(name: str, seed: int = 42):
    """name ∈ {lr, rf, svm}。返回 sklearn Pipeline（自带缺失值填补）。"""
    imputer = SimpleImputer(strategy="median")
    if name == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("clf", clf)])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        return Pipeline([("impute", imputer), ("clf", clf)])
    if name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
        return Pipeline([("impute", imputer), ("scale", StandardScaler()), ("clf", clf)])
    raise ValueError(f"未知基线: {name}（可选 lr / rf / svm）")


BASELINE_NAMES = ["lr", "rf", "svm"]
