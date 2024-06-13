import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from sklearn.preprocessing import QuantileTransformer


data = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/Preprocessing/stroke_preprocessed.csv")

X = data.drop(columns=["stroke"])
y = data["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["ever_married_No"])
eo = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["gender_Male"])
eod = equal_opportunity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test["gender_Male"])

dpd = round(dpd, 4)
eo = round(eo, 4)
eod = round(eod, 4)

print(f"Demographic Parity difference: {dpd}")
print(f"Equalized Odds difference: {eo}")
print(f"Equality of Opportunities difference: {eod}")