import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from sklearn.preprocessing import QuantileTransformer


heart_disease_original = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/real/heart_disease_original.csv")
heart_disease_ctgan = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/heart_disease_synthetic_ctgan.csv")
heart_disease_distcorrgan = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/heart_disease_synthetic_distcorrgan.csv")
heart_disease_tabfairgan = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/heart_disease_synthetic_tabfairgan.csv")
heart_diease_multifairgan = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/heart_disease_synthetic_multifairgan.csv")
heart_disease_fairgan = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/heart_disease_synthetic_fairgan.csv")


data = heart_disease_fairgan

X = data.drop(columns="target")
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["sex"])
eo = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["sex"])
eod = equal_opportunity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test["sex"])


dpd = round(dpd, 4)
eo = round(eo, 4)
eod = round(eod, 4)

print(f"Demographic Parity difference: {dpd}")
print(f"Equalized Odds difference: {eo}")
print(f"Equality of Opportunities difference: {eod}")