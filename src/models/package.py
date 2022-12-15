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

import shutil

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
    cfg = cfg.package_model
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
    
    # Train the model
    Xtr, ytr = th.load_data(train_cfg, train=True)
    model = ttf.train_tf_model(train_cfg, Xtr, ytr)

    # Move the saved model into the right directory
    log.info(f'Saving to {cfg.model_save_path}')
    shutil.copytree(train_cfg.output_dir, cfg.model_save_path, dirs_exist_ok=True)    
    # model.args.no_save = False
    # model.save_model(output_dir=cfg.model_save_path)
    # torchmodel = model.model
    # torchmodel.save_pretrained(cfg.model_save_path)

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

if __name__ == '__main__':
    main()