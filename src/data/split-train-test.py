import pandas as pd
import numpy as np
import typer
import logging as log
from pathlib import Path
from sklearn.model_selection import train_test_split

def main(dataset_path: Path = typer.Option('data/interim/masterdbForNLPSpacyProc.pkl', exists=True, dir_okay=False),
         output_dir: Path = typer.Option('data/processed/', dir_okay=True),
         test_size: float = 0.15,
         random_state: int = 43,
         train_file_name: str = 'train.pkl',
         test_file_name: str = 'test.pkl',
         log_level: str = typer.Option('INFO')):

    params = locals()
    log.basicConfig(level=log_level)
    log.info("Splitting datset into train and test...")
    log.debug(f"Parameters:\n{params}")

    log.info(f'Loading data from {dataset_path}')
    df = pd.read_pickle(dataset_path)

    log.info(f'Setting aside {test_size} of data for final testing')
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    log.info(f'Train shape {train.shape} / Test shape {test.shape}')
    
    train_path = output_dir/train_file_name
    test_path = output_dir/test_file_name
    log.info(f'Saving train file to {train_path}')
    train.to_pickle(train_path)
    log.info(f'Saving test file to {test_path}')
    test.to_pickle(test_path)

if __name__ == '__main__':
    typer.run(main)