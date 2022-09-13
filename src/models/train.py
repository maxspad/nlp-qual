from pyclbr import Function
import pandas as pd
import numpy as np
import pickle

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
    mlflow.set_tracking_uri(cfg.mlflow_tracking_dir)
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
        y_value_counts = pd.Series(y).value_counts().sort_index()
        multi_level = len(y_value_counts) > 2
        if multi_level:
            log.warning(f'Target {cfg.target_var} has {len(y_value_counts)} levels! Metrics will be multi-level.')
        if cfg.invert_target:
            if multi_level:
                log.warning(f'Cannot invert a multi-level target! Ignoring')
            else:
                y = y + 1
                y[y == 2] = 0
        log.info(f'Y value counts\n{y_value_counts}')
        log.debug(f'X shape {X.shape} / y shape {y.shape}')

        pipe_steps_subvars = get_pipe_steps_for_subvars(cfg)
    
        # Add the model to the pipeline at the end
        mdl = LinearSVC(C=cfg.model_c, class_weight=cfg.class_weight, random_state=cfg.random_seed, max_iter=cfg.max_iter)
        if cfg.target_var == 'QUAL':
            pipe_steps_QUAL = get_pipe_steps_for_QUAL(cfg)
            log.info(f'Target {cfg.target_var}, subtype {cfg.qual_fit_type}')
            if cfg.qual_fit_type == 'simultaneous':
                pipe = Pipeline((
                    ('ctouter', ColumnTransformer((
                        ('submodels', Pipeline(pipe_steps_QUAL), [0]),
                        ('text', Pipeline(pipe_steps_subvars), [0])
                    ))),
                    ('mdl', mdl)
                ))
            elif cfg.qual_fit_type == 'text_only':
                pipe_steps = pipe_steps_subvars + [('mdl', mdl)]
                pipe = Pipeline(pipe_steps)
            elif (cfg.qual_fit_type == 'submodels_only') or (cfg.qual_fit_type == 'text_first'):
                pipe_steps = pipe_steps_QUAL + [('mdl', mdl)]
                pipe = Pipeline(pipe_steps)
            else:
                raise ValueError('qual_fit_type must be one of text_only/submodels_only/simultaneous/text_first')
        else:            
            log.info(f'Target {cfg.target_var}')
            pipe_steps = pipe_steps_subvars + [('mdl', mdl)]
            pipe = Pipeline(pipe_steps)
        
        log.info(f'Pipeline is\n{pipe}')

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

def submodel_function(sm):
    def _submodel_func(X, y=None):
        df = sm.decision_function(X)
        if len(df.shape) < 2:
            df = df[:, None]
        # print(df.shape)
        return df
    return _submodel_func

from sklearn.preprocessing import FunctionTransformer

def load_submodel(mlflow_dir, mlflow_exp_name, mlflow_run_id):
    # Load the trained model
    exper = mlflow.get_experiment_by_name(mlflow_exp_name)
    log.info(f'Loaded experiment {mlflow_exp_name} with ID {exper.experiment_id} at artifact location {exper.artifact_location}')
    model_loc = f'{mlflow_dir}/{exper.experiment_id}/{mlflow_run_id}/artifacts/model/model.pkl'
    log.info(f'Loading model from {model_loc}')
    with open(model_loc, 'rb') as f:
        clf = pickle.load(f)
    return clf

def get_pipe_steps_for_QUAL(cfg: DictConfig):
    log.info('Loading submodels...')
    q1 = load_submodel(cfg.mlflow_dir, cfg.q1_mlflow_exp_name, cfg.q1_mlflow_run_id)
    q2 = load_submodel(cfg.mlflow_dir, cfg.q2_mlflow_exp_name, cfg.q2_mlflow_run_id)
    q3 = load_submodel(cfg.mlflow_dir, cfg.q3_mlflow_exp_name, cfg.q3_mlflow_run_id)
    submodels = [q1, q2, q3]
    sm_names = ['q1','q2','q3']
    if cfg.qual_fit_type == 'text_first':
        submodels.append(load_submodel(cfg.mlflow_dir, cfg.qual_text_mlflow_exp_name, cfg.qual_text_mlflow_run_id))
        sm_names.append('qt')

    ct_steps = [(f'{n}ft', FunctionTransformer(submodel_function(sm)), [0]) for n, sm in zip(sm_names, submodels)]
    pipe_steps = [
        ('ct', ColumnTransformer(ct_steps)),
        ('scaler', MinMaxScaler()),
    ]
    return pipe_steps

def get_pipe_steps_for_subvars(cfg: DictConfig):
    tokfilt = SpacyTokenFilter(punct=cfg.punct, lemma=cfg.lemma, stop=cfg.stop, pron=cfg.pron)
    vec = CountVectorizer(max_df=cfg.max_df, min_df=cfg.min_df, ngram_range=(cfg.ngram_min, cfg.ngram_max))
    docfeats = SpacyDocFeats(token_count=cfg.token_count, pos_counts=cfg.pos_counts, ent_counts=cfg.ent_counts, vectors=cfg.vectors)
    scaler = MinMaxScaler()
    # mdl = LinearSVC(C=cfg.model_c, class_weight=cfg.class_weight, random_state=cfg.random_seed, max_iter=cfg.max_iter)
    if not any([cfg.token_count, cfg.pos_counts, cfg.ent_counts, cfg.vectors]):
        pipe_steps = [
            ('tokfilt', tokfilt),
            ('vec', vec)
        ]
    else:
        pipe_steps = [
            ('ct', ColumnTransformer((
                ('bowpipe', Pipeline((
                    ('tokfilt', tokfilt),
                    ('vec', vec)
                )), [0]),
                ('docfeatspipe', Pipeline((
                    ('docfeats', docfeats),
                    ('scaler', scaler)
                )), [0])
            )))
        ]
    return pipe_steps

def calculate_metrics(y, p, s):
    cm = mets.confusion_matrix(y, p)
    n_classes = cm.shape[0]
    avg = 'binary' if n_classes == 2 else 'macro'
    toret = {
        'balanced_accuracy': mets.balanced_accuracy_score(y, p),
        'accuracy': mets.accuracy_score(y, p),
        'roc_auc': mets.roc_auc_score(y, s) if n_classes == 2 else np.nan,
        'f1': mets.f1_score(y, p, average=avg),
        'precision': mets.precision_score(y, p, average=avg),
        'recall': mets.recall_score(y, p, average=avg),
        'mae': mets.mean_absolute_error(y, p)
    }

    if n_classes == 2:
        toret['tp'] = cm[1,1]
        toret['tn'] = cm[0,0]
        toret['fp'] = cm[0,1]
        toret['fn'] = cm[1,0]
    else:
        precs, recs, f1s, supps = mets.precision_recall_fscore_support(y, p, average=None, labels=list(range(n_classes)))
        for c, prfs in enumerate(zip(precs, recs, f1s, supps)):
        # for c in range(n_classes):
            # prec, rec, f1, supp = mets.precision_recall_fscore_support(y, p, average='binary', pos_label=c)
            prec, rec, f1, supp = prfs
            toret[f'prec_{c}'] = prec
            toret[f'rec_{c}'] = rec
            toret[f'f1_{c}'] = f1 
            toret[f'supp_{c}'] = supp
            
            for j, v in enumerate(cm[c, :]):
                toret[f'cm_{c}_{j}'] = v
            toret['top_2_acc'] = mets.top_k_accuracy_score(y, s, k=2)
            toret['top_3_acc'] = mets.top_k_accuracy_score(y, s, k=3)
    return toret 

def _model_scorer(clf, X, y):
    p = clf.predict(X)
    s = clf.decision_function(X)
    return calculate_metrics(y, p, s)

if __name__ == "__main__":
    main()