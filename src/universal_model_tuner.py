from pathlib import Path
import yaml
import optuna
import importlib
from sklearn.model_selection import cross_val_score
import json
import joblib
import inspect


'''
    Classe Universal de Tunning e Treinamento
    Principais funções: Receber informações do yml e fazer o treinamento do modelo
'''
class UniversalModelTuner:
    # Contrutor global da classe
    # Parâmetros: X_Treino, X_Teste, _model_config
    def __init__(self, X_train, y_train, scenario_name, n_trials=10, scorer='accuracy', cv=3):
        self.X_train      = X_train
        self.y_train      = y_train 
        self.scenario_name = scenario_name
        self.n_trials     = n_trials
        self.scorer       = scorer
        self.cv           = cv
        self.fitted_models = {}  # Bib para salvar os modelos que foram treinados

        # Caminho do arquivo atual ee cnontra a raiz
        self.CURRENT_FILE = Path(__file__).resolve()
        self.PROJECT_ROOT = self.CURRENT_FILE.parent.parent

    # Importar e ler quais modelos vão ser rodados
    def _load_config(self):
        # Caminho do arquivo de configurações
        CONFIG_PATH = self.PROJECT_ROOT / 'configs' / 'search_spaces.yml'

        # Abre o arquivo yaml
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)

        return configs

    '''
        Importar modelo (Implementado pelo Gemini)
    '''
    def _build_model(self, params):
        complete_path = self._model_config["model_class"]
        module, class_name = complete_path.rsplit('.', 1)
        bib = importlib.import_module(module)
        ClasseClassificador = getattr(bib, class_name)

        # Para garantir reproditibilidade
        clf_params = inspect.signature(ClasseClassificador).parameters
        if 'random_state' in clf_params:
            params['random_state'] = 14
        
        return ClasseClassificador(**params)

    '''
        Exportar resultados (Implementado pelo Gemini)
    '''
    def _export_models(self):
        # Define os caminhos
        models_dir = self.PROJECT_ROOT / 'outputs' / 'models'
        logs_dir = self.PROJECT_ROOT / 'outputs' / 'logs'

        # Cria os diretórios no SO caso não existam
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Salva o modelo binário (.joblib ou .pkl)

        for model_name, model_data in self.fitted_models.items():
            best_model = model_data['model']
            study = model_data['study']

            model_filepath = models_dir / f"{self.scenario_name}_{model_name}_best.joblib"
            joblib.dump(best_model, model_filepath)

            # 2. Salva as informações da otimização em JSON
            info = {
                "scenario_name": self.scenario_name,
                "model_name": model_name,
                "best_params": model_data['params']
            }
            
            info_filepath = logs_dir / f"{self.scenario_name}_{model_name}_info.json"
            with open(info_filepath, "w") as f:
                json.dump(info, f, indent=4)

            print('Informações dos modelos exportadas! \n')
            
        print(f"Configurações e métricas salvas em: {info_filepath}")

    # Função de otimização do optuna
    def _objective(self, trial):
        # Espaço de busca
        params = {}

        # Montar o espaço de busca conforme o yaml
        for param_name, rules in self._model_config['hyperparameters'].items():
            # Tipo do Hiperparâmetro
            tipo = rules['type']

            if tipo == 'int':
                params[param_name] = trial.suggest_int(param_name, rules['low'], rules['high'])
            elif tipo == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, rules['low'], rules['high'], log=rules.get("log", False)
                )
            elif tipo == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, rules['choices'])

        # Definir o kernel caso seja SVM
        if self._model_config['model_class'] == 'sklearn.svm.SVC':
            if params.get('kernel') == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 4)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
            elif params.get('kernel') == 'sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

        # O classificador
        clf = self._build_model(params)

        score = cross_val_score(clf, self.X_train, self.y_train, scoring=self.scorer, cv=self.cv, n_jobs=-1).mean()
        return score

    def tune_and_fit(self, n_trials):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print(f" -- Iniciando otimização do {self._model_config['model_name']} --")
        
        # Colocado o self. para exportar depois
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)

        print(f'Melhor {self.scorer}: {study.best_value:.4f}')
        print(f'Melhores hiperparâmetros: {study.best_params}')

        # Colocado o self. para exportar depois
        best_model = self._build_model(study.best_params)
        best_model.fit(self.X_train, self.y_train)

        self.fitted_models[self._model_config['model_name']] = {
            'study': study,
            'model': best_model,  
            'params': study.best_params
        }

        print()

    def testar_coisas(self):
        for model_name, model_data in self.fitted_models.items():
            print(model_name)
            print(model_data)

    def tune_and_fit_all_models(self):
        configs = self._load_config()

        # Treinar todos os modelos
        for loaded_model in configs.values():
            self._model_config = loaded_model
            self.tune_and_fit(self.n_trials)

        # Salvar os modelos
        self._export_models()

        print('Fim, todos os modelos treinados.')
            