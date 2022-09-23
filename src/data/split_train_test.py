from logging.config import dictConfig
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg : DictConfig):
    cfg = cfg.split_train_test

    log.info("Splitting datset into train and test...")
    log.debug(f"Parameters:\n{OmegaConf.to_yaml(cfg)}")

    log.info(f'Loading data from {cfg.dataset_path}')
    df = pd.read_pickle(cfg.dataset_path)

    log.info(f'Setting aside {cfg.test_size} of data for final testing')
    train, test = train_test_split(df, test_size=cfg.test_size, random_state=cfg.random_state)
    log.info(f'Train shape {train.shape} / Test shape {test.shape}')
    
    # train_path = cfg.output_dir/cfg.train_file_name
    # test_path = cfg.output_dir/cfg.test_file_name
    log.info(f'Saving train file to {cfg.train_path}')
    train.to_pickle(cfg.train_path)
    log.info(f'Saving test file to {cfg.test_path}')
    test.to_pickle(cfg.test_path)

if __name__ == '__main__':
    main()