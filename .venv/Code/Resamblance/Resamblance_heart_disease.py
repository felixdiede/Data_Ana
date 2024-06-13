import pandas as pd
from Resamblance_Metrics import *
import os

os.chdir("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv")

heart_disease_original = pd.read_csv("data/real/heart_disease_original.csv")
heart_disease_ctgan = pd.read_csv("data/synth/heart_disease_synthetic_ctgan.csv")
#heart_disease_tvae =
heart_disease_distcorrgan = pd.read_csv("data/synth/heart_disease_synthetic_distcorrgan.csv")
heart_disease_tabfairgan = pd.read_csv("data/synth/heart_disease_synthetic_tabfairgan.csv")
heart_disease_multifairgan = pd.read_csv("data/synth/heart_disease_synthetic_multifairgan.csv")

cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "target"]
num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def statistical_tests(real_data, synthetic_data):
    print_results_t_test(real_data, synthetic_data, attribute=cat_features)
    print_results_mw_test(real_data, synthetic_data, attribute=cat_features)
    print_results_ks_test(real_data, synthetic_data, attribute=cat_features)
    print_results_chi2_test(real_data, synthetic_data, attribute=num_features)

# statistical_tests(heart_disease_original, heart_disease_ctgan)
calculate_and_display_distances(heart_disease_original, heart_disease_multifairgan, heart_disease_original.columns)



