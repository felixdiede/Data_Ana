from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

real_data = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/real/diabetes_health_original.csv")
synthetic_data = pd.read_csv("/Users/felixdiederichs/Desktop/Desktop - Felixs MacBook Pro/Wirtschaftsinformatik/Thesis/Code/Analysis/.venv/data/synth/diabetes_health_distcorrgan.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe()

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='Diabetes_binary'
)

fig.show()
