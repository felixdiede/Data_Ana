import pandas as pdl
from Resamblance_Metrics import *
import os


os.chdir("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv")

diabetes_health_original = pd.read_csv("data/real/diabetes_health_original.csv")
diabetes_health_distcorrgan = pd.read_csv("data/synth/diabetes_health_synthetic_distcorrgan.csv")
diabetes_health_tabfairgan = pd.read_csv("data/synth/diabetes_health_synthetic_tabfairgan.csv")
diabetes_health_multifairgan = pd.read_csv("data/synth/diabetes_health_synthetic_multifairgan.csv")

cat_features = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex", "Income", "GenHlth"]
num_features = ["BMI", "MentHlth", "PhysHlth", "Education"]


def statistical_tests(real_data, synthetic_data):
    print_results_t_test(real_data, synthetic_data, attribute=cat_features)
    print_results_mw_test(real_data, synthetic_data, attribute=cat_features)
    print_results_ks_test(real_data, synthetic_data, attribute=cat_features)
    print_results_chi2_test(real_data, synthetic_data, attribute=num_features)

# statistical_tests(diabetes_health_original, diabetes_health_multifairgan)

calculate_and_display_distances(diabetes_health_original, diabetes_health_distcorrgan, diabetes_health_original.columns)

# ppc_matrix(diabetes_health_original, diabetes_health_tabfairgan, num_features)

# normalized_contingency_tables()

# data_labelling_analysis(diabetes_health_original, diabetes_health_tabfairgan)