# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

# mlflow
import mlflow

# NLP/ML
import pickle
import pandas as pd 
import src.models.train as train 

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg: DictConfig):
    cfg = cfg.test

    log.info('Evaluating model on test set...')
    log.debug(f"Parameters:\n{OmegaConf.to_yaml(cfg)}")

    # Load the trained model
    exper = mlflow.get_experiment_by_name(cfg.mlflow_source_experiment_name)
    log.info(f'Loaded experiment {cfg.mlflow_source_experiment_name} with ID {exper.experiment_id} at artifact location {exper.artifact_location}')
    model_loc = f'{cfg.mlflow_dir}/{exper.experiment_id}/{cfg.mlflow_run_id}/artifacts/model/model.pkl'
    log.info(f'Loading model from {model_loc}')
    with open(model_loc, 'rb') as f:
        clf = pickle.load(f)

    # Evaluate model
    mlflow.set_experiment(experiment_name=cfg.mlflow_target_experiment_name)
    with mlflow.start_run():
        log.info('Evaluating model...')
        mlflow.log_params(OmegaConf.to_object(cfg))

        log.info(f'Loading data from {cfg.test_path}')
        df = pd.read_pickle(cfg.test_path)
        log.info(f'Data is shape {df.shape}')
        log.debug(f'Data head\n{df.head()}')

        X = df[['comment_spacy']].values.copy()
        y = df[cfg.target_var].values.copy()
        y = (y - 1) * -1 # invert
        p = clf.predict(X)
        s = clf.decision_function(X)

        metrics = train.calculate_metrics(y, p, s)
        mlflow.log_metrics(metrics)
        met_df = pd.DataFrame([metrics])
        log.info(f'Metrics\n{met_df}')

if __name__ == '__main__':
    main()