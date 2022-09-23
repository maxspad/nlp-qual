import pandas as pd
import numpy as np
import pickle

import warnings

# Spacy NLP / sklearn
from sklearn.model_selection import ShuffleSplit
import sklearn.metrics as mets

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

# mlflow
import mlflow

# training pipeline
from . import train_helpers as th
from . import train_tf as ttf
from simpletransformers.classification import ClassificationArgs, ClassificationModel

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg : DictConfig):
    cfg = cfg.test_tf
    mlflow.set_tracking_uri(cfg.mlflow_tracking_dir)

    # Load the model parameters from cross-validation
    exper = mlflow.get_experiment_by_name(cfg.mlflow_source_experiment_name)
    log.info(f'Loaded experiment {cfg.mlflow_source_experiment_name} with ID {exper.experiment_id} at artifact location {exper.artifact_location}')
    run = mlflow.get_run(cfg.mlflow_run_id)
    log.info(f'Loaded run {run.info.run_id}')
    run_params = run.data.params
    train_cfg = OmegaConf.create(load_params(run_params))
    # config_loc = f'{cfg.mlflow_dir}/{exper.experiment_id}/{cfg.mlflow_run_id}/artifacts/config.yaml'
    # train_cfg = OmegaConf.load(config_loc).train_tf
    # log.info(f'Loaded train configuration from {config_loc}')
    log.info(f'Configuration is \n{train_cfg}')
    
    Xtr, ytr = th.load_data(train_cfg, train=True)
    model = ttf.train_tf_model(train_cfg, Xtr, ytr)
    
    mlflow.set_experiment(experiment_name=cfg.mlflow_target_experiment_name)
    with mlflow.start_run():
        log.info('Evaluating model...')
        mlflow.log_params(OmegaConf.to_object(cfg))

        X, y = th.load_data(cfg, train=False)
        df = pd.DataFrame({'text': X[:,0], 'labels': y})
        p, s = model.predict(df['text'].tolist())
        metrics = th.tf_calculate_metrics(y, p, s)

        mlflow.log_metrics(metrics)
        met_df = pd.DataFrame([metrics])
        log.info(f'Metrics\n{met_df}')

        mlflow.log_text(met_df.to_csv(), 'results.csv')
        mlflow.log_artifact(cfg.conda_yaml_path)
        mlflow.log_artifact(CONF_FILE)

def load_params(params_dict):
    res_dict = {}
    for key in params_dict:
        val = params_dict[key]
        if val == 'True':
            res_dict[key] = True
        elif val == 'False': 
            res_dict[key] = False
        else:
            if val.find('.') > -1:
                try:
                    res_dict[key] = float(val)
                except:
                    res_dict[key] = val
            else:
                try:
                    res_dict[key] = int(val)
                except:
                    res_dict[key] = val
    return res_dict

def load_model(cfg: DictConfig, model_dir: str):
    model = ClassificationModel(
            cfg.model,
            model_dir
        )
    return model

if __name__ == '__main__':
    main()