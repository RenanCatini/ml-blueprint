from src.universal_model_tuner import UniversalModelTuner
import pandas as pd

# Carrega os dados usados no exemplo.
X_test = pd.read_csv('data/processed/X_test.csv')
X_train = pd.read_csv('data/processed/X_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

# Cria o tuner e treina todos os modelos configurados no YAML.
tuner = UniversalModelTuner(X_train, y_train, 'testes')
tuner.tune_and_fit_all_models()

# Acessa um dos modelos treinados para fazer previsões ou avaliações.
knn_treinado = tuner.fitted_models['KNN']['model']