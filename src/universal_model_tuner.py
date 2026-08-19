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
    # Parâmetros: X_Treino, X_Teste, model_config
    def __init__(self, X_train, y_train, model_config, scenario_name, n_trials, scorer='accuracy', cv=3):
        self.X_train      = X_train
        self.y_train      = y_train
        self.model_config = model_config 
        self.scenario_name = scenario_name
        self.n_trials     = n_trials
        self.scorer       = scorer
        self.cv           = cv

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
        complete_path = self.model_config["model_class"]
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
    def _export_results(self):
        # Define os caminhos
        models_dir = self.PROJECT_ROOT / 'outputs' / 'models'
        logs_dir = self.PROJECT_ROOT / 'outputs' / 'logs'

        # Cria os diretórios no SO caso não existam
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Salva o modelo binário (.joblib ou .pkl)
        model_filepath = models_dir / f"{self.scenario_name}_best.joblib"
        joblib.dump(self.best_model, model_filepath)
        print(f"Modelo salvo em: {model_filepath}")

        # 2. Salva as informações da otimização em JSON
        info = {
            "scenario_name": self.scenario_name,
            "best_cv_score": float(self.study.best_value),
            "best_params": self.study.best_params
        }
        
        # Correção do caminho aqui (sem o self.PROJECT_ROOT repetido)
        info_filepath = logs_dir / f"{self.scenario_name}_info.json"
        with open(info_filepath, "w") as f:
            json.dump(info, f, indent=4)
            
        print(f"Configurações e métricas salvas em: {info_filepath}")

    # Função de otimização do optuna
    def _objective(self, trial):
        # Espaço de busca
        params = {}

        # Montar o espaço de busca conforme o yaml
        for param_name, rules in self.model_config['hyperparameters'].items():
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
        if self.model_config['model_class'] == 'sklearn.svm.SVC':
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
        print(f'Iniciando otimização...')
        
        # Colocado o self. para exportar depois
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self._objective, n_trials=n_trials)

        print(f'Melhor pontuação (CV): {self.study.best_value:.4f}')
        print(f'Melhores hiperparâmetros: {self.study.best_params}')

        # Colocado o self. para exportar depois
        self.best_model = self._build_model(self.study.best_params)
        self.best_model.fit(self.X_train, self.y_train)

        self._export_results()

        return self.best_model

            