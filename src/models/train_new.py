# helper code
from . import train_helpers as th

# data processing
import pickle
import pandas as pd
import numpy as np

# mlflow
import mlflow

# Spacy NLP / sklearn
from ..skspacy import SpacyTokenFilter, SpacyDocFeats, SpacyTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.model_selection import cross_validate

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

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
        
        # Load data
        X, y = th.load_data(cfg)
        augtf = th.SynonymAugTransformer()
        X = augtf.fit_transform(X)
        y = np.concatenate((y,y))

        # Prepare pipeline
        pipe = make_pipeline(cfg)
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


def make_pipeline(cfg: DictConfig):

    # the model step is the same for all targets
    mdl = LinearSVC(C=cfg.model_c, class_weight=cfg.class_weight, random_state=cfg.random_seed, max_iter=cfg.max_iter)

    # pipeline steps for Q1, Q2, Q3
    pipe_steps_subvars = get_pipe_steps_for_subvars(cfg)
    
    log.info(f'Targeting {cfg.target_var}')
    if cfg.target_var != 'QUAL':
        # targeting Q1/Q2/Q3
        pipe_steps = pipe_steps_subvars + [('mdl', mdl)]
        return Pipeline(pipe_steps)
    
    else:
        # targeting QuAL, several ways to do this
        log.info(f'QUAL subtype {cfg.qual_fit_type}')

        if cfg.qual_fit_type == 'text_only':
            # same as for the subvars
            pipe_steps = pipe_steps_subvars + [('mdl', mdl)]
            return Pipeline(pipe_steps)
        
        elif cfg.qual_fit_type in ['submodels_only', 'text_first']:
            # use pre-trained Q1/Q2/Q3 to output their decision functions
            # and then train a model on those. If qual_fit_type == 'text_first'
            # then include a separate text_only QuAL model as a variable as well
            pipe_steps_submodels = get_pipe_steps_for_QUAL(cfg)
            pipe_steps = pipe_steps_submodels + [('mdl', mdl)]
            return Pipeline(pipe_steps)
        
        elif cfg.qual_fit_type == 'simultaneous':
            # like submodels_only, but the text is included as well
            pipe_steps_submodels = get_pipe_steps_for_QUAL(cfg)
            return Pipeline((
                ('ctouter', ColumnTransformer((
                    ('submodels', Pipeline(pipe_steps_submodels), [0]),
                    ('text', Pipeline(pipe_steps_subvars), [0])
                ))),
                ('mdl', mdl)
            ))
        
        else:
            raise ValueError('qual_fit_type must be one of text_only/submodels_only/simultaneous/text_first')
    
def get_pipe_steps_for_subvars(cfg: DictConfig):
    syntf = th.SynonymAugTransformer()
    spacytf = SpacyTransformer(prog=cfg.spacy_prog, procs=cfg.spacy_procs)
    tokfilt = SpacyTokenFilter(punct=cfg.punct, lemma=cfg.lemma, stop=cfg.stop, pron=cfg.pron)
    vec = CountVectorizer(max_df=cfg.max_df, min_df=cfg.min_df, ngram_range=(cfg.ngram_min, cfg.ngram_max))
    docfeats = SpacyDocFeats(token_count=cfg.token_count, pos_counts=cfg.pos_counts, ent_counts=cfg.ent_counts, vectors=cfg.vectors)
    scaler = MinMaxScaler()

    if not any([cfg.token_count, cfg.pos_counts, cfg.ent_counts, cfg.vectors]):
        # if no count fatures, we don't need the ColumnTransformer
        pipe_steps = [
            # ('syntf', syntf), # TODO
            ('spacytf', spacytf),
            ('tokfilt', tokfilt),
            ('vec', vec)
        ]
    else:
        # need a more complex pipeline if using count features
        pipe_steps = [
            # ('syntf', syntf), # TODO
            ('spacytf', spacytf),
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

def submodel_function(sm, decision_function=True):
    def _submodel_func(X, y=None):
        df = sm.decision_function(X) if decision_function else sm.predict(X)
        if len(df.shape) < 2:
            df = df[:, None]
        # print(df.shape)
        return df
    return _submodel_func

def load_submodel(mlflow_dir, mlflow_exp_name, mlflow_run_id):
    # Load the trained model
    exper = mlflow.get_experiment_by_name(mlflow_exp_name)
    log.info(f'Loaded experiment {mlflow_exp_name} with ID {exper.experiment_id} at artifact location {exper.artifact_location}')
    model_loc = f'{mlflow_dir}/{exper.experiment_id}/{mlflow_run_id}/artifacts/model/model.pkl'
    log.info(f'Loading model from {model_loc}')
    with open(model_loc, 'rb') as f:
        clf = pickle.load(f)
    return clf

def load_all_submodels(cfg: DictConfig):
    log.info('Loading submodels...')
    q1 = load_submodel(cfg.mlflow_dir, cfg.q1_mlflow_exp_name, cfg.q1_mlflow_run_id)
    q2 = load_submodel(cfg.mlflow_dir, cfg.q2_mlflow_exp_name, cfg.q2_mlflow_run_id)
    q3 = load_submodel(cfg.mlflow_dir, cfg.q3_mlflow_exp_name, cfg.q3_mlflow_run_id)
    submodels = [q1, q2, q3]
    sm_names = ['q1','q2','q3']
    return submodels, sm_names

def get_pipe_steps_for_QUAL(cfg: DictConfig):
    submodels, sm_names = load_all_submodels(cfg)
    spacytf = SpacyTransformer(prog=cfg.spacy_prog, procs=cfg.spacy_procs)
    if cfg.qual_fit_type == 'text_first':
        submodels.append(load_submodel(cfg.mlflow_dir, cfg.qual_text_mlflow_exp_name, cfg.qual_text_mlflow_run_id))
        sm_names.append('qt')

    ct_steps = [(f'{n}ft', FunctionTransformer(submodel_function(sm)), [0]) for n, sm in zip(sm_names, submodels)]
    pipe_steps = [
        ('spacytf', spacytf),
        ('ct', ColumnTransformer(ct_steps)),
        ('scaler', MinMaxScaler()),
    ]
    return pipe_steps

def _model_scorer(clf, X, y):
    p = clf.predict(X)
    try:
        s = clf.decision_function(X)
    except AttributeError:
        s = None
    return th.calculate_metrics(y, p, s)

if __name__ == '__main__':
    main()
