import pandas as pd
import numpy as np
import typer
import logging as log
from pathlib import Path
from ..skspacy import SpacyTransformer
import os

def main(dataset_path: Path = typer.Option('data/interim/masterdbForNLP.xlsx', exists=True, dir_okay=False),
         output_path: Path = typer.Option('data/interim/masterdbForNLPSpacyProc.pkl', dir_okay=False),
         text_var: str = 'comment',
         spacy_model: str = 'en_core_web_sm',
         spacy_procs: int = typer.Option(4, min=1, max=os.cpu_count()),
         progress_bar: bool = True,
         log_level: str = 'INFO'):
    
    params = locals()
    log.basicConfig(level=log_level)
    log.info('Processing text with spacy...')
    log.debug(f'Parameters:\n{params}')

    log.info(f'Loading data from {dataset_path}')
    df = pd.read_excel(dataset_path)
    log.info(f'Data is shape {df.shape}')

    spacytf = SpacyTransformer(spacy_model=spacy_model, procs=spacy_procs, prog=progress_bar)
    new_col = text_var + '_spacy'
    df[new_col] = spacytf.fit_transform(df[text_var].values.copy()[:,None].astype('str'))

    log.info(f'Saving processed data to {output_path}')
    df.to_pickle(output_path)

if __name__ == '__main__':
    typer.run(main)
         