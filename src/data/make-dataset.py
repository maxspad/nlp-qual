import pandas as pd
import numpy as np
import typer 
from pathlib import Path
import logging as log


def _impute_macrob_score_for_imperfect_matches(df: pd.DataFrame):
    dPerfect = df[df['perfectMatch'] == True]
    dNonPerfect = df[df['perfectMatch'] == False]
    log.debug(f"dPerfect shape {dPerfect.shape}")
    log.debug(f"dNonPerfect shape {dNonPerfect.shape}")

    df['Q1'] = dNonPerfect['RobMacQ1']
    df['Q2'] = dNonPerfect['RobMacQ2']
    df['Q3'] = dNonPerfect['RobMacQ3']
    df['QUAL'] = dNonPerfect['RobMacQualScore']

    # now fill the blanks from the original ratings.
    # it does not matter getting P1 or P2 score as they are perfect match
    df['Q1'].fillna(dPerfect['q1p1T'], inplace=True)
    df['Q2'].fillna(dPerfect['q2p1T'], inplace=True)
    df['Q3'].fillna(dPerfect['q3p1T'], inplace=True)
    df['QUAL'].fillna(dPerfect['P1QualScore'], inplace=True)

    #calculate sum of qual scores and compare with the previous manually summed values to determine if they check out.
    df['summedQs'] = df['Q1']+df['Q2']+df['Q3']
    comparison_QUALScore_columns = np.where(df['summedQs'] == df['QUAL'], True, False)
    df["isQUALequal"] = comparison_QUALScore_columns
    log.info(f'Number of manually summed columns that equal auto-summed QuAL scores:\n{df["isQUALequal"].value_counts()}')
    log.info('Unequal will be replaced by auto-summed QuAL scores.')
    # for those that don't replace with the calculated qual scores
    df.loc[(df.isQUALequal == False),'QUAL'] = df['summedQs']
    # get rid of helper columns
    df.drop(['summedQs', 'isQUALequal'], axis=1, inplace=True)

    return df 

def main(masterdb_path: Path = typer.Option('data/raw/masterdbFromRobMac.xlsx', exists=True, dir_okay=False),
         output_path: Path = typer.Option('data/processed/masterdbForNLP.xlsx'),
         log_level: str = typer.Option('INFO')):

    params = locals()
    log.basicConfig(level=log_level)
    log.info("Generating final dataset...")
    log.debug(f"Parameters:\n{params}")
    
    # load the raw data
    log.info(f"Loading raw data from {masterdb_path}")
    masterdb = pd.read_excel(masterdb_path,  index_col=0)
    log.debug(f'Shape is {masterdb.shape}')

    # impute the corrected QuAL scores from above if they don't match
    log.info('Imputing corrected scores where necessary...')
    masterdb = _impute_macrob_score_for_imperfect_matches(masterdb)

    # output the result
    log.info(f'Saving to {output_path}')
    masterdb.to_excel(output_path)

if __name__ == '__main__':
    typer.run(main)
    exit(0)
