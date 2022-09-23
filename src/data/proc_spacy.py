import pandas as pd
import numpy as np

from ..skspacy import SpacyTransformer

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg : DictConfig):
    cfg = cfg.proc_spacy

    log.info('Processing text with spacy...')
    log.debug(f'Parameters:\n{OmegaConf.to_yaml(cfg)}')

    log.info(f'Loading data from {cfg.dataset_path}')
    df = pd.read_excel(cfg.dataset_path)
    log.info(f'Data is shape {df.shape}')

    spacytf = SpacyTransformer(spacy_model=cfg.spacy_model, procs=cfg.spacy_procs, prog=cfg.progress_bar)
    new_col = cfg.text_var + '_spacy'
    df[new_col] = spacytf.fit_transform(df[cfg.text_var].values.copy()[:,None].astype('str'))

    log.info(f'Saving processed data to {cfg.output_path}')
    df.to_pickle(cfg.output_path)

if __name__ == '__main__':
    main()
         