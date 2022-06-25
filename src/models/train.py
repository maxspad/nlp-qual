from importlib_metadata import version
import pandas as pd

# Spacy NLP / sklearn
from ..skspacy import SpacyTokenFilter, SpacyDocFeats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import sklearn.metrics as mets

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

def int_to_bool(value: int):
    return True if value else False

@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg : DictConfig):
    cfg = cfg.train 

    log.info('Training model...')
    log.debug(f"Parameters:\n{OmegaConf.to_yaml(cfg)}")
    
    log.info(f'Loading data from {cfg.train_path}')
    df = pd.read_pickle(cfg.train_path)
    log.info(f'Data is shape {df.shape}')
    log.info(f'There are {df[cfg.text_var].isna().sum()} blanks in {cfg.text_var}, dropping')
    log.info(f'There are {df[cfg.target_var].isna().sum()} blanks in {cfg.target_var}, dropping')
    df = df.dropna(subset=[cfg.text_var, cfg.target_var])
    log.debug(f'Data head\n{df.head()}')

    X = df[cfg.text_var].values.copy()[:, None]
    # spacytf = SpacyTransformer(spacy_model=spacy_model, procs=spacy_procs, prog=progress_bar)
    # log.info(f'Processing text with spacy model {spacy_model}...')
    # X = spacytf.fit_transform(X)
    y = df[cfg.target_var].values.copy()
    y = y + 1
    y[y == 2] = 0
    log.debug(f'X shape {X.shape} / y shape {y.shape}')

    mdl = LinearSVC(C=cfg.model_c, class_weight=cfg.class_weight, random_state=cfg.random_seed)
    pipe = Pipeline((
        ('ct', ColumnTransformer((
            ('bowpipe', Pipeline((
                ('tokfilt', SpacyTokenFilter(punct=cfg.punct, lemma=cfg.lemma, stop=cfg.stop, pron=cfg.pron)),
                ('vec', CountVectorizer(max_df=cfg.max_df, min_df=cfg.min_df, ngram_range=(cfg.ngram_min, cfg.ngram_max))),
            )), [0]),
            ('docfeatspipe', Pipeline((
                ('docfeats', SpacyDocFeats(token_count=cfg.token_count, pos_counts=cfg.pos_counts, ent_counts=cfg.ent_counts, vectors=cfg.vectors)),
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
    main()