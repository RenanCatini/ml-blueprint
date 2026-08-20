from src.universal_model_tuner import UniversalModelTuner
import pandas as pd

X_test = pd.read_csv('data/processed/X_test.csv')
X_train = pd.read_csv('data/processed/X_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

teste = UniversalModelTuner(X_train, y_train, 'testes')
teste.tune_and_fit_all_models()
