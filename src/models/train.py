from audioop import cross
from statistics import mode
from matplotlib.pyplot import text
import pandas as pd
import numpy as np
from skspacy import SpacyTransformer, SpacyTokenFilter, SpacyDocFeats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate
import sklearn.metrics as mets
from sklearn.metrics import get_scorer_names


import typer
import logging as log
from pathlib import Path

def int_to_bool(value: int):
    return True if value else False

true_opt = typer.Option(1, min=0, max=1, callback=int_to_bool)
false_opt = typer.Option(0, min=0, max=1, callback=int_to_bool)

def main(train_path: Path = typer.Option('data/processed/train.csv', exists=True, dir_okay=False),
         text_var : str = 'comment',
         target_var : str = 'Q2',
         spacy_model : str = 'en_core_web_sm',
         spacy_procs : int = typer.Option(4, min=-1, max=8),
         progress_bar : bool = True,
         random_seed : int = 43,
         punct : int = true_opt,
         pron: int = true_opt,
         stop: int = false_opt,
         lemma: int = false_opt,
         ngram_min: int = 1,
         ngram_max: int = 2,
         max_df: float = 1.0,
         min_df: int = 1,
         token_count: int = true_opt,
         pos_counts: int = false_opt,
         ent_counts: int = false_opt,
         vectors: int = false_opt,
         model_c: float = 0.01,
         class_weight: str = 'balanced',
         log_level: str = "INFO"):

    params = locals()
    log.basicConfig(level=log_level)
    log.info('Training model...')
    log.debug(f"Parameters:\n{params}")
    
    log.info(f'Loading data from {train_path}')
    df = pd.read_csv(train_path)
    log.info(f'Data is shape {df.shape}')
    log.info(f'There are {df[text_var].isna().sum()} blanks in {text_var}, dropping')
    log.info(f'There are {df[target_var].isna().sum()} blanks in {target_var}, dropping')
    df = df.dropna(subset=[text_var, target_var])
    log.debug(f'Data head\n{df.head()}')

    X = df[text_var].values.copy()[:, None].astype('str')
    spacytf = SpacyTransformer(spacy_model=spacy_model, procs=spacy_procs, prog=progress_bar)
    log.info(f'Processing text with spacy model {spacy_model}...')
    X = spacytf.fit_transform(X)
    y = df[target_var].values.copy()
    y = y + 1
    y[y == 2] = 0
    log.debug(f'X shape {X.shape} / y shape {y.shape}')

    mdl = LinearSVC(C=model_c, class_weight=class_weight, random_state=random_seed)
    pipe = Pipeline((
        ('ct', ColumnTransformer((
            ('bowpipe', Pipeline((
                ('tokfilt', SpacyTokenFilter(punct=punct, lemma=lemma, stop=stop, pron=pron)),
                ('vec', CountVectorizer(max_df=max_df, min_df=min_df, ngram_range=(ngram_min, ngram_max))),
            )), [0]),
            ('docfeatspipe', Pipeline((
                ('docfeats', SpacyDocFeats(token_count=token_count, pos_counts=pos_counts, ent_counts=ent_counts, vectors=vectors)),
                ('scaler', MinMaxScaler())
            )), [0])
        ))),
        ('mdl', mdl)
    ))
    log.debug(f'Pipeline is\n{pipe}')


    log.info('Fitting model...')
    res = cross_validate(pipe, X, y, scoring=_model_scorer, cv=5, n_jobs=1)
    res = pd.DataFrame(res)
    res_mn = pd.DataFrame(res.mean()).T.rename(lambda x: 'mean_' + x, axis=1)
    res_std = pd.DataFrame(res.std()).T.rename(lambda x: 'std_' + x, axis=1)
    print(res)
    print()
    print(res_mn)
    print()
    print(res_std)


def _model_scorer(clf, X, y):
    p = clf.predict(X)
    s = clf.decision_function(X)
    cm = mets.confusion_matrix(y, p)

    return {
        'balanced_accuracy': mets.balanced_accuracy_score(y, p),
        'accuracy': mets.accuracy_score(y, p),
        'roc_auc': mets.roc_auc_score(y, s),
        'f1': mets.f1_score(y, p),
        'precision': mets.precision_score(y, p),
        'recall': mets.recall_score(y, p),
        'tp': cm[0,0],
        'tn': cm[1,1],
        'fp': cm[0,1],
        'fn': cm[1,0],
        # 'confusion': mets.confusion_matrix(y, p),
        # 'clfrep': mets.classification_report(y, p)
    }

if __name__ == "__main__":
    typer.run(main)