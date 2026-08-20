from pathlib import Path
import yaml
import optuna
import importlib
from sklearn.model_selection import cross_val_score
import json
import joblib
import inspect


class UniversalModelTuner:
    """Otimiza, treina e exporta modelos de machine learning.

    As configurações dos modelos e dos hiperparâmetros são carregadas de
    ``configs/search_spaces.yml``. Para cada modelo configurado, a classe
    executa uma busca de hiperparâmetros com Optuna, avalia as combinações
    usando validação cruzada, ajusta o melhor modelo com todos os dados de
    treino e salva os resultados em ``outputs/models`` e ``outputs/logs``.

    Parameters
    ----------
    X_train : array-like
        Dados usados para treinar e avaliar os modelos.
    y_train : array-like
        Valores-alvo correspondentes a ``X_train``.
    scenario_name : str
        Nome do cenário usado como prefixo nos arquivos exportados.
    n_trials : int, default=10
        Número de tentativas realizadas pelo Optuna para cada modelo.
    scorer : str, default='accuracy'
        Métrica aceita pelo ``sklearn.model_selection.cross_val_score``.
    cv : int or cross-validation generator, default=3
        Estratégia de validação cruzada usada durante a otimização.
    """

    def __init__(self, X_train, y_train, scenario_name, n_trials=10, scorer='accuracy', cv=3):
        self.X_train      = X_train
        self.y_train      = y_train 
        self.scenario_name = scenario_name
        self.n_trials     = n_trials
        self.scorer       = scorer
        self.cv           = cv
        self.fitted_models = {}

        # Localiza a raiz do projeto a partir deste módulo.
        self.CURRENT_FILE = Path(__file__).resolve()
        self.PROJECT_ROOT = self.CURRENT_FILE.parent.parent

    def _load_config(self):
        """Carrega as configurações de modelos e hiperparâmetros do YAML.

        Returns
        -------
        dict
            Configurações definidas em ``configs/search_spaces.yml``.
        """
        CONFIG_PATH = self.PROJECT_ROOT / 'configs' / 'search_spaces.yml'

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)

        return configs

    def _build_model(self, params):
        """Instancia o classificador definido na configuração atual.

        Parameters
        ----------
        params : dict
            Hiperparâmetros que serão passados ao construtor do modelo.

        Returns
        -------
        object
            Instância do estimador configurado.
        """
        complete_path = self._model_config["model_class"]
        module, class_name = complete_path.rsplit('.', 1)
        bib = importlib.import_module(module)
        ClasseClassificador = getattr(bib, class_name)

        # Define uma semente quando o estimador oferece esse parâmetro.
        clf_params = inspect.signature(ClasseClassificador).parameters
        if 'random_state' in clf_params:
            params['random_state'] = 14
        
        return ClasseClassificador(**params)

    def _export_models(self):
        """Exporta os melhores modelos e seus hiperparâmetros.

        Os modelos são salvos como arquivos ``.joblib`` em
        ``outputs/models``. Os metadados da otimização são salvos como JSON
        em ``outputs/logs``.
        """
        models_dir = self.PROJECT_ROOT / 'outputs' / 'models'
        logs_dir = self.PROJECT_ROOT / 'outputs' / 'logs'

        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_data in self.fitted_models.items():
            best_model = model_data['model']

            model_filepath = models_dir / f"{self.scenario_name}_{model_name}_best.joblib"
            joblib.dump(best_model, model_filepath)

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

    def _objective(self, trial):
        """Avalia uma combinação de hiperparâmetros durante a otimização.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Objeto usado pelo Optuna para sugerir os hiperparâmetros.

        Returns
        -------
        float
            Média da métrica obtida na validação cruzada.
        """
        params = {}

        for param_name, rules in self._model_config['hyperparameters'].items():
            tipo = rules['type']

            if tipo == 'int':
                params[param_name] = trial.suggest_int(param_name, rules['low'], rules['high'])
            elif tipo == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, rules['low'], rules['high'], log=rules.get("log", False)
                )
            elif tipo == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, rules['choices'])

        # Alguns parâmetros do SVC dependem do kernel selecionado.
        if self._model_config['model_class'] == 'sklearn.svm.SVC':
            if params.get('kernel') == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 4)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
            elif params.get('kernel') == 'sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

        clf = self._build_model(params)

        score = cross_val_score(clf, self.X_train, self.y_train, scoring=self.scorer, cv=self.cv, n_jobs=-1).mean()
        return score

    def _tune_and_fit(self, n_trials):
        """Otimiza e ajusta o modelo definido na configuração atual.

        Parameters
        ----------
        n_trials : int
            Número de tentativas da busca de hiperparâmetros.

        O estudo, o modelo ajustado e os melhores parâmetros são armazenados
        em ``fitted_models`` para exportação posterior.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print(f" -- Iniciando otimização do {self._model_config['model_name']} --")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)

        print(f'Melhor {self.scorer}: {study.best_value:.4f}')
        print(f'Melhores hiperparâmetros: {study.best_params}')

        best_model = self._build_model(study.best_params)
        best_model.fit(self.X_train, self.y_train)

        self.fitted_models[self._model_config['model_name']] = {
            'study': study,
            'model': best_model,  
            'params': study.best_params
        }

        print()

    def tune_and_fit_all_models(self):
        """Otimiza, treina e exporta todos os modelos configurados.

        As configurações são processadas na ordem em que aparecem no arquivo
        YAML. Ao final, os modelos ajustados e os metadados são exportados.
        """
        configs = self._load_config()

        for loaded_model in configs.values():
            self._model_config = loaded_model
            self._tune_and_fit(self.n_trials)

        self._export_models()

        print('Fim, todos os modelos treinados.')
            