import pandas as pd
import numpy as np
import pickle

import warnings

# Spacy NLP / sklearn
from sklearn.model_selection import ShuffleSplit
import sklearn.metrics as mets

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

    mlflow.set_experiment(experiment_name=cfg.mlflow_experiment_name)
    with mlflow.start_run():
        df = pd.DataFrame({'text': X[:,0], 'labels': y})
        splitter = ShuffleSplit(n_splits=cfg.n_splits, test_size=cfg.test_size, random_state=cfg.random_state)
        fold_results = []
        for train_idx, test_idx in splitter.split(df):
            train_df = df.iloc[train_idx,:]
            test_df = df.iloc[test_idx,:]
            
            log.info('Training model...')
            model = fit_model(cfg, train_df)

            log.info('Evaluating model on this fold...')
            p, s = model.predict(test_df['text'].tolist())
            results = calculate_metrics(test_df['labels'].values, p, s)
            log.info(f'Fold Results: {results}')
            
            fold_results.append(results)

        res = pd.DataFrame(fold_results)
        res_mn = pd.DataFrame(res.mean()).T.rename(lambda x: 'mean_' + x, axis=1)
        res_std = pd.DataFrame(res.std()).T.rename(lambda x: 'std_' + x, axis=1)
        mlflow.log_metrics(res_mn.iloc[0,:].to_dict())
        mlflow.log_metrics(res_std.iloc[0,:].to_dict())
        log.info(f'Cross validation results:\n{res}\n{res_mn}\n{res_std}')

        mlflow.log_text(res.to_csv(),'fold_results.csv')
        mlflow.log_text(res_mn.to_csv(),'res_mn.csv')
        mlflow.log_text(res_std.to_csv(),'res_std.csv')
        mlflow.log_artifact(cfg.conda_yaml_path)
        mlflow.log_artifact(CONF_FILE)

        log.info('Fitting final model...')
        model = fit_model(cfg, df)  
        log.info('Saving final model...')
        mlflow.log_artifact(cfg.output_dir)

        return res_mn[cfg.objective_metric]
    
    
def fit_model(cfg: DictConfig, train_df):
        model_args = ClassificationArgs(
            num_train_epochs=cfg.num_train_epochs,
            output_dir=cfg.output_dir,
            overwrite_output_dir=cfg.overwrite_output_dir,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False
        )
        model = ClassificationModel(
            cfg.model,
            cfg.submodel,
            args=model_args,
            num_labels=len(train_df.labels.unique())
        )
        model.train_model(train_df)
        return model

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
            toret['top_2_acc'] = mets.top_k_accuracy_score(y, s, k=2)
            toret['top_3_acc'] = mets.top_k_accuracy_score(y, s, k=3)
    return toret



    train_df, test_df = train_test_split(df, test_size=0.25)
    model_args = ClassificationArgs(num_train_epochs=cfg.num_train_epochs, output_dir='st_outputs', use_multiprocessing=False, overwrite_output_dir=True, use_multiprocessing_for_evaluation=False)
    model = ClassificationModel(cfg.model, cfg.submodel, args=model_args, num_labels=len(df.labels.unique()))
    # log.info(train_df.labels.value_counts())
    model.train_model(train_df)
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=mets.accuracy_score, bac=mets.balanced_accuracy_score)

if __name__ == "__main__":
    main()