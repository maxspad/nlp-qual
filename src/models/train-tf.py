import pandas as pd
import numpy as np
import pickle

import warnings

# Spacy NLP / sklearn
from ..skspacy import SpacyTokenFilter, SpacyDocFeats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
import sklearn.metrics as mets
from sklearn.exceptions import ConvergenceWarning

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

# mlflow
import mlflow

# training pipeline
from . import train_helpers as th

from simpletransformers.classification import ClassificationArgs, ClassificationModel

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg : DictConfig):
    cfg = cfg.train_tf

    X, y = th.load_data(cfg)

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    df = pd.DataFrame({'text': X[:,0], 'labels': y})
    train_df, test_df = train_test_split(df, test_size=0.25)
    model_args = ClassificationArgs(num_train_epochs=cfg.num_train_epochs, output_dir='st_outputs', use_multiprocessing=False, overwrite_output_dir=True, use_multiprocessing_for_evaluation=False)
    model = ClassificationModel(cfg.model, cfg.submodel, args=model_args, num_labels=len(df.labels.unique()))
    # log.info(train_df.labels.value_counts())
    model.train_model(train_df)
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=mets.accuracy_score, bac=mets.balanced_accuracy_score)

if __name__ == "__main__":
    main()