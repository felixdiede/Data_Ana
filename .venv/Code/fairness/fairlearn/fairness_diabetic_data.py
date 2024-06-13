import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference

data = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/real/diabetic_data_transformed.csv")
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
"""
sensitive_features = ["race_Other"]
def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    results = {}
    for attr in sensitive_features:
        results[attr] = {
            "Demographic Parity difference ": demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test[attr]),
            "Equalized Odds difference ": equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test[attr]),
            "Equality of Opportunities difference ": equal_opportunity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test[attr])
        }

    for attrs, metrics in results.items():
        print("------------------------------------")
        for metric_name, value in metrics.items():
            print(f"| {attrs} | {metric_name} | {value}")

calculate_fairness_metrics(y_test, y_pred, sensitive_features)

"""
dpd = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["race_Other"])
eo = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=X_test["race_Caucasian"])
eod = equal_opportunity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test["race_Caucasian"])

dpd = round(dpd, 4)
eo = round(eo, 4)
eod = round(eod, 4)

print(f"Demographic Parity difference: {dpd}")
print(f"Equalized Odds difference: {eo}")
print(f"Equality of Opportunities difference: {eod}")
