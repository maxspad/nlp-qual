import pandas as pd

import warnings

# Spacy NLP / sklearn
from ..skspacy import SpacyTokenFilter, SpacyDocFeats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import sklearn.metrics as mets
from sklearn.exceptions import ConvergenceWarning

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

# mlflow
import mlflow

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg : DictConfig):
    cfg = cfg.train 

    mlflow.set_experiment(experiment_name=cfg.mlflow_experiment_name)
    with mlflow.start_run():
        log.info('Training model...')
        log.debug(f"Parameters:\n{OmegaConf.to_yaml(cfg)}")
        mlflow.log_params(OmegaConf.to_object(cfg))
        
        log.info(f'Loading data from {cfg.train_path}')
        df = pd.read_pickle(cfg.train_path)
        log.info(f'Data is shape {df.shape}')
        log.info(f'There are {df[cfg.text_var].isna().sum()} blanks in {cfg.text_var}, dropping')
        log.info(f'There are {df[cfg.target_var].isna().sum()} blanks in {cfg.target_var}, dropping')
        df = df.dropna(subset=[cfg.text_var, cfg.target_var])
        log.debug(f'Data head\n{df.head()}')

        X = df[cfg.text_var].values.copy()[:, None]
        y = df[cfg.target_var].values.copy()
        if cfg.invert_target:
            y = y + 1
            y[y == 2] = 0
        log.info(f'Y value counts\n{pd.Series(y).value_counts().sort_index()}')
        log.debug(f'X shape {X.shape} / y shape {y.shape}')

        tokfilt = SpacyTokenFilter(punct=cfg.punct, lemma=cfg.lemma, stop=cfg.stop, pron=cfg.pron)
        vec = CountVectorizer(max_df=cfg.max_df, min_df=cfg.min_df, ngram_range=(cfg.ngram_min, cfg.ngram_max))
        docfeats = SpacyDocFeats(token_count=cfg.token_count, pos_counts=cfg.pos_counts, ent_counts=cfg.ent_counts, vectors=cfg.vectors)
        scaler = MinMaxScaler()
        mdl = LinearSVC(C=cfg.model_c, class_weight=cfg.class_weight, random_state=cfg.random_seed, max_iter=cfg.max_iter)
        if not any([cfg.token_count, cfg.pos_counts, cfg.ent_counts, cfg.vectors]):
            pipe = Pipeline((
                ('tokfilt', tokfilt),
                ('vec', vec),
                ('mdl', mdl)
            ))
        else:
            pipe = Pipeline((
                ('ct', ColumnTransformer((
                    ('bowpipe', Pipeline((
                        ('tokfilt', tokfilt),
                        ('vec', vec)
                    )), [0]),
                    ('docfeatspipe', Pipeline((
                        ('docfeats', docfeats),
                        ('scaler', scaler)
                    )), [0])
                ))),
                ('mdl', mdl)
            ))
        log.debug(f'Pipeline is\n{pipe}')


        log.info('Cross validating model...')
        n_failed_converge = 0
        res = cross_validate(pipe, X, y, scoring=_model_scorer, cv=5, n_jobs=1, return_estimator=True)
        n_failed_converge = sum([estim[-1].n_iter_ >= estim[-1].get_params()['max_iter'] for estim in res['estimator']])
        if n_failed_converge > 0:
            log.warning(f'{n_failed_converge} folds failed to converge!')
        del res['estimator']

        res = pd.DataFrame(res)
        res_mn = pd.DataFrame(res.mean()).T.rename(lambda x: 'mean_' + x, axis=1)
        res_std = pd.DataFrame(res.std()).T.rename(lambda x: 'std_' + x, axis=1)
        mlflow.log_metrics(res_mn.iloc[0,:].to_dict())
        mlflow.log_metrics(res_std.iloc[0,:].to_dict())
        mlflow.log_metric('n_failed_converge', n_failed_converge)
        log.info(f'Cross validation results:\n{res}\n{res_mn}\n{res_std}')

        mlflow.log_text(res.to_csv(),'fold_results.csv')
        mlflow.log_text(res_mn.to_csv(),'res_mn.csv')
        mlflow.log_text(res_std.to_csv(),'res_std.csv')
        mlflow.log_artifact(cfg.conda_yaml_path)
        mlflow.log_artifact(CONF_FILE)

        log.info('Fitting final model...')
        pipe.fit(X, y)     
        log.info('Saving final model...')   
        mlflow.sklearn.log_model(pipe, 'model')

        return res_mn['mean_test_balanced_accuracy']

def calculate_metrics(y, p, s):
    cm = mets.confusion_matrix(y, p)
    return {
        'balanced_accuracy': mets.balanced_accuracy_score(y, p),
        'accuracy': mets.accuracy_score(y, p),
        'roc_auc': mets.roc_auc_score(y, s),
        'f1': mets.f1_score(y, p),
        'precision': mets.precision_score(y, p),
        'recall': mets.recall_score(y, p),
        'tp': cm[1,1],
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        # 'confusion': str(cm)
        # 'confusion': mets.confusion_matrix(y, p),
        # 'clfrep': mets.classification_report(y, p)
    }
def _model_scorer(clf, X, y):
    p = clf.predict(X)
    s = clf.decision_function(X)
    return calculate_metrics(y, p, s)
    # cm = mets.confusion_matrix(y, p)

    # return {
    #     'balanced_accuracy': mets.balanced_accuracy_score(y, p),
    #     'accuracy': mets.accuracy_score(y, p),
    #     'roc_auc': mets.roc_auc_score(y, s),
    #     'f1': mets.f1_score(y, p),
    #     'precision': mets.precision_score(y, p),
    #     'recall': mets.recall_score(y, p),
    #     'tp': cm[0,0],
    #     'tn': cm[1,1],
    #     'fp': cm[0,1],
    #     'fn': cm[1,0],
    #     # 'confusion': mets.confusion_matrix(y, p),
    #     # 'clfrep': mets.classification_report(y, p)
    # }

if __name__ == "__main__":
    main()