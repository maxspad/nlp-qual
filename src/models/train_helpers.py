# data science
from audioop import reverse
import pandas as pd
import numpy as np

# ML/NLP
import sklearn.metrics as mets
from sklearn.utils import shuffle
from nlpaug.augmenter.word.synonym import SynonymAug
from nlpaug.augmenter.word.context_word_embs import ContextualWordEmbsAug
from sklearn.base import BaseEstimator, TransformerMixin

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

def load_data(cfg : DictConfig, train=True):
        # Load data
        data_path = cfg.train_path if train else cfg.test_path
        log.info(f'Loading data from {data_path}')
        df = pd.read_pickle(data_path)
        log.info(f'Data is shape {df.shape}')

        # Drop any blanks
        log.info(f'There are {df[cfg.text_var].isna().sum()} blanks in {cfg.text_var}, dropping')
        log.info(f'There are {df[cfg.target_var].isna().sum()} blanks in {cfg.target_var}, dropping')
        df = df.dropna(subset=[cfg.text_var, cfg.target_var])
        log.debug(f'Data head\n{df.head()}')

        # Drop all level 4 if requested
        if (cfg.target_var == 'QUAL') and (cfg.qual_exclude_level4):
            log.warning('Filtering out level 4 as requested!')
            qual_level4 = df[cfg.target_var] == 4
            log.warning(f'Dropping {qual_level4.sum()} items.')
            df = df[~qual_level4]

        # split out comments and labels
        X = df[cfg.text_var].values.copy()[:, None].astype('str')
        y = df[cfg.target_var].values.copy()

        # if multi_level, warn that the metrics will be different
        y_value_counts = pd.Series(y).value_counts().sort_index()
        multi_level = len(y_value_counts) > 2
        if multi_level:
            log.warning(f'Target {cfg.target_var} has {len(y_value_counts)} levels! Metrics will be multi-level.')

        # invert target if requested
        if cfg.invert_target:
            if multi_level:
                log.warning(f'Cannot invert a multi-level target! Ignoring')
            else:
                y = y + 1
                y[y == 2] = 0

        # Report the value_counts and data shape
        log.info(f'Y value counts\n{y_value_counts}')
        log.debug(f'X shape {X.shape} / y shape {y.shape}')
        return X, y

def calculate_metrics(y, p, s):
    cm = mets.confusion_matrix(y, p)
    n_classes = cm.shape[0]
    avg = 'binary' if n_classes == 2 else 'macro'
    toret = {
        'balanced_accuracy': mets.balanced_accuracy_score(y, p),
        'accuracy': mets.accuracy_score(y, p),
        'roc_auc': mets.roc_auc_score(y, s) if n_classes == 2 else np.nan,
        'f1': mets.f1_score(y, p, average=avg),
        'precision': mets.precision_score(y, p, average=avg),
        'recall': mets.recall_score(y, p, average=avg),
        'mae': mets.mean_absolute_error(y, p)
    }

    if n_classes == 2:
        toret['tp'] = cm[1,1]
        toret['tn'] = cm[0,0]
        toret['fp'] = cm[0,1]
        toret['fn'] = cm[1,0]
    else:
        precs, recs, f1s, supps = mets.precision_recall_fscore_support(y, p, average=None, labels=list(range(n_classes)))
        for c, prfs in enumerate(zip(precs, recs, f1s, supps)):
        # for c in range(n_classes):
            # prec, rec, f1, supp = mets.precision_recall_fscore_support(y, p, average='binary', pos_label=c)
            prec, rec, f1, supp = prfs
            toret[f'prec_{c}'] = prec
            toret[f'rec_{c}'] = rec
            toret[f'f1_{c}'] = f1 
            toret[f'supp_{c}'] = supp
            
            for j, v in enumerate(cm[c, :]):
                toret[f'cm_{c}_{j}'] = v
            toret['top_2_acc'] = mets.top_k_accuracy_score(y, s, k=2) if s is not None else np.nan
            toret['top_3_acc'] = mets.top_k_accuracy_score(y, s, k=3) if s is not None else np.nan
    return toret 

class SynonymAugTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, aug_src='wordnet', model_path=None, name='Synonym_Aug', 
        aug_min=1, aug_max=10, aug_p=0.3, lang='eng', stopwords=None, 
        tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, 
        force_reload=False, verbose=0):

        self.aug_src=aug_src
        self.model_path=model_path
        self.name=name
        self.aug_min=aug_min
        self.aug_max=aug_max
        self.aug_p=aug_p
        self.lang=lang
        self.stopwords=stopwords
        self.tokenizer=tokenizer
        self.reverse_tokenizer=reverse_tokenizer
        self.stopwords_regex=stopwords_regex
        self.force_reload=force_reload
        self.verbose=verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = X[:,0].tolist()
        aug = SynonymAug(aug_src=self.aug_src, model_path=self.model_path, name=self.name, 
            aug_min=self.aug_min, aug_max=self.aug_max, aug_p=self.aug_p, lang=self.lang, stopwords=self.stopwords, 
            tokenizer=self.tokenizer, reverse_tokenizer=self.reverse_tokenizer, stopwords_regex=self.stopwords_regex, 
            force_reload=self.force_reload, verbose=self.verbose)
        augmented = np.array(aug.augment(data))[:,None]
        return np.vstack((X, augmented))


def augment_train(cfg: DictConfig, Xtr: np.ndarray, ytr: np.ndarray):
    if cfg.do_aug:
        levels, counts = np.unique(ytr, return_counts=True)
        highest_count = np.max(counts)
        most_freq_level = levels[np.argmax(counts)]
        target_highest_count = np.ceil(cfg.aug_factor * highest_count)
        match_count = np.ceil(cfg.match_factor * target_highest_count)
        aug_X = []
        aug_y = []
        for lvl in levels:
            X_lvl, y_lvl = Xtr[ytr == lvl], ytr[ytr == lvl]
            aug_X_lvl, aug_y_lvl = do_upsample_aug(X_lvl, y_lvl, 
                target_highest_count if lvl == most_freq_level else match_count, do_replacement=cfg.do_replacement)
            aug_X.append(aug_X_lvl)
            aug_y.append(aug_y_lvl)

        Xtr_aug = np.vstack(aug_X)
        ytr_aug = np.concatenate(aug_y)
        # these are stacked according to level and need to be shuffled
        Xtr_aug, ytr_aug = shuffle(Xtr_aug, ytr_aug, random_state=cfg.random_state)
        
        return Xtr_aug, ytr_aug
    else:
        return Xtr, ytr

def do_upsample_aug(X: np.ndarray, y: np.ndarray, target_count: int, do_replacement=True):
    # the actual target is len(X) - target_count
    n_to_sample = int(target_count - X.shape[0])
    if n_to_sample <= 0:
        log.info(f'Target count is {target_count}, X length is {X.shape[0]}, no need to upsample')
        return X, y
    sample_idxs = np.random.randint(0, X.shape[0], n_to_sample)
    X_sample = X[sample_idxs, :]
    y_sample = y[sample_idxs]
    if do_replacement:
        auger = SynonymAug()
        # auger = ContextualWordEmbsAug(device='cuda')
        X_sample_aug = np.array(auger.augment(X_sample[:, 0].tolist()))[:,None]
    else:
        X_sample_aug = X_sample
    return np.vstack((X, X_sample_aug)), np.concatenate((y, y_sample))


def tf_calculate_metrics(y, p, s):
    cm = mets.confusion_matrix(y, p)
    n_classes = cm.shape[0]
    avg = 'binary' if n_classes == 2 else 'macro'
    toret = {
        'balanced_accuracy': mets.balanced_accuracy_score(y, p),
        'accuracy': mets.accuracy_score(y, p),
        'roc_auc': np.nan, # mets.roc_auc_score(y, s) if n_classes == 2 else np.nan,
        'f1': mets.f1_score(y, p, average=avg),
        'precision': mets.precision_score(y, p, average=avg),
        'recall': mets.recall_score(y, p, average=avg),
        'mae': mets.mean_absolute_error(y, p)
    }

    if n_classes == 2:
        toret['tp'] = cm[1,1]
        toret['tn'] = cm[0,0]
        toret['fp'] = cm[0,1]
        toret['fn'] = cm[1,0]
    else:
        precs, recs, f1s, supps = mets.precision_recall_fscore_support(y, p, average=None, labels=list(range(n_classes)))
        for c, prfs in enumerate(zip(precs, recs, f1s, supps)):
        # for c in range(n_classes):
            # prec, rec, f1, supp = mets.precision_recall_fscore_support(y, p, average='binary', pos_label=c)
            prec, rec, f1, supp = prfs
            toret[f'prec_{c}'] = prec
            toret[f'rec_{c}'] = rec
            toret[f'f1_{c}'] = f1 
            toret[f'supp_{c}'] = supp
            
            for j, v in enumerate(cm[c, :]):
                toret[f'cm_{c}_{j}'] = v
            toret['top_2_acc'] = mets.top_k_accuracy_score(y, s, k=2)
            toret['top_3_acc'] = mets.top_k_accuracy_score(y, s, k=3)
    return toret
