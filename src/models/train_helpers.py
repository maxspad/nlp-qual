# data science
import pandas as pd

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

def load_data(cfg : DictConfig):
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
        return X, y

