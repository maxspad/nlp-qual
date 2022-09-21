import subprocess
import typer
from typing import Optional, List
import datetime
import os

def main(sweep_name: str = typer.Argument('sweep'),
         append_time: bool = True,
         hydra_args: Optional[List[str]] = typer.Option(None, '-a', '--arg')):


    now = datetime.datetime.now()
    now_str = now.strftime('%y%m%d_%H%M%S')
    exp_name = sweep_name if not append_time else f'{now_str}_{sweep_name}'
    fullcmd = ['python', '-m', 'src.models.train_new', '--multirun', f'train.mlflow_experiment_name={exp_name}']
    fullcmd += hydra_args

    print('Running:')
    print(' '.join(fullcmd))

    subprocess.call(fullcmd)

if __name__ == '__main__':
    typer.run(main)