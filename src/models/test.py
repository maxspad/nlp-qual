# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)

# mlflow
import mlflow

# NLP/ML
import pickle
import pandas as pd 
import numpy as np
import src.models.train as train 

# displays
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import wordcloud 

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg: DictConfig):
    cfg = cfg.test

    log.info('Evaluating model on test set...')
    log.debug(f"Parameters:\n{OmegaConf.to_yaml(cfg)}")
    mlflow.set_tracking_uri(cfg.mlflow_dir)
    # Load the trained model
    exper = mlflow.get_experiment_by_name(cfg.mlflow_source_experiment_name)
    log.info(f'Loaded experiment {cfg.mlflow_source_experiment_name} with ID {exper.experiment_id} at artifact location {exper.artifact_location}')
    model_loc = f'{cfg.mlflow_dir}/{exper.experiment_id}/{cfg.mlflow_run_id}/artifacts/model/model.pkl'
    log.info(f'Loading model from {model_loc}')
    with open(model_loc, 'rb') as f:
        clf = pickle.load(f)

    # Evaluate model
    mlflow.set_experiment(experiment_name=cfg.mlflow_target_experiment_name)
    with mlflow.start_run():
        log.info('Evaluating model...')
        mlflow.log_params(OmegaConf.to_object(cfg))

        log.info(f'Loading data from {cfg.test_path}')
        df = pd.read_pickle(cfg.test_path)
        log.info(f'Data is shape {df.shape}')
        log.debug(f'Data head\n{df.head()}')

        X = df[['comment_spacy']].values.copy()
        y = df[cfg.target_var].values.copy()
        y_value_counts = pd.Series(y).value_counts().sort_index()
        multi_level = len(y_value_counts) > 2
        if multi_level:
            log.warning(f'Target {cfg.target_var} has {len(y_value_counts)} levels! Metrics will be multi-level.')
        if cfg.invert_target:
            if multi_level:
                log.warning(f'Cannot invert a multi-level target! Ignoring')
            else:
                y = y + 1
                y[y == 2] = 0
        p = clf.predict(X)
        s = clf.decision_function(X)

        metrics = train.calculate_metrics(y, p, s)
        mlflow.log_metrics(metrics)
        met_df = pd.DataFrame([metrics])
        log.info(f'Metrics\n{met_df}')

        mlflow.log_text(met_df.to_csv(), 'results.csv')
        mlflow.log_artifact(cfg.conda_yaml_path)
        mlflow.log_artifact(CONF_FILE)
        mlflow.sklearn.log_model(clf, 'model')

        if not cfg.posthoc:
            log.info('Post-hoc not requested, skipping.')
            return
        
        log.info('Performing post-hoc analysis...')

        log.info('Calculating feature weights...')
        feat_weights = get_feat_weights_df(clf)
        mlflow.log_text(feat_weights.to_csv(), 'feat_weights.csv')
        tfwfc = get_top_feat_weights_for_classes(feat_weights, cfg.target_var, cfg.n_top)
        mlflow.log_text(tfwfc, 'top_feat_weights_for_classes.txt')

        # exclude vectors from here on out
        feat_weights = feat_weights.reset_index()
        feat_weights = feat_weights[~feat_weights['feature'].str.contains('vec_')]
        feat_weights = feat_weights.set_index(['class','feature'])
        tfwfc_no_vec = get_top_feat_weights_for_classes(feat_weights, cfg.target_var, cfg.n_top)
        mlflow.log_text(tfwfc_no_vec, 'top_feat_weights_for_classes_no_vec.txt')

        log.info('Building word clouds...')
        for c in feat_weights.index.levels[0]:
            f = wordcloud_for_class(feat_weights, cfg.target_var, c)
            mlflow.log_figure(f, f'wordcloud_class{c}.png')

def wordcloud_for_class(feat_weights: pd.DataFrame, q : str, c : int,
                        figsize=(10,20), w = 800, h = 400, cmap = 'Greens'):
    fwc = feat_weights.loc[c, :]
    fw_for_wc = fwc[fwc['coef'] >= 0]['abs'].to_dict()
    wc = wordcloud.WordCloud(width=w, height=h, colormap=get_cmap(cmap))
    wc = wc.fit_words(fw_for_wc)
    f = plt.figure(figsize=figsize)
    plt.imshow(wc)
    plt.axis('off')
    plt.title(f'{q} = {c}')
    return f

def get_top_feat_weights_for_classes(feat_weights: pd.DataFrame, q : str, n_top = 10):
    toRet = ''
    for c in feat_weights.index.levels[0]:
        fwc = feat_weights.loc[c, :]
        toRet += f'Highest Weighted toward {q} = {c}\n'
        toRet += fwc.sort_values('coef', ascending=False).head(n_top).to_string() + '\n\n'
    return toRet

def get_feat_weights_df(mdl):
    coefs = mdl[-1].coef_.copy()
    feats = mdl[:-1].get_feature_names_out()
    fwdfs = [
        pd.DataFrame({
            'feature': [f.split('__')[1] for f in feats],
            'class': i,
            'coef': coefs[i, :]
        })
        for i in range(coefs.shape[0])
    ]
    fwdf = pd.concat(fwdfs, axis=0).sort_values(['class','feature']).set_index(['class','feature'])
    fwdf['abs'] = fwdf['coef'].abs()
    fwdf['pos'] = fwdf['coef'] >= 0
    return fwdf 

if __name__ == '__main__':
    main()