import pandas as pd
import numpy as np

# configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm

# mlflow
import mlflow

# transformers
from src.models import train_tf as ttf
from src.models import train_helpers as th

# plotting
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from wordcloud import WordCloud

CONF_PATH = '../../'
CONF_FOLDER = 'conf'
CONF_NAME = 'config'
CONF_FILE = f'{CONF_FOLDER}/{CONF_NAME}.yaml'

def load_params(params_dict):
    res_dict = {}
    for key in params_dict:
        val = params_dict[key]
        if val == 'True':
            res_dict[key] = True
        elif val == 'False': 
            res_dict[key] = False
        else:
            if val.find('.') > -1:
                try:
                    res_dict[key] = float(val)
                except:
                    res_dict[key] = val
            else:
                try:
                    res_dict[key] = int(val)
                except:
                    res_dict[key] = val
    return res_dict

@hydra.main(version_base=None, config_path=f'{CONF_PATH}/{CONF_FOLDER}', config_name=CONF_NAME)
def main(cfg : DictConfig):
    cfg = cfg.word_clouds
    mlflow.set_tracking_uri(cfg.mlflow_tracking_dir)

    # Load the model parameters from cross-validation
    log.info(f'Loading Run with id {cfg.mlflow_run_id}')
    run = mlflow.get_run(cfg.mlflow_run_id)
    run_params = run.data.params
    train_cfg = OmegaConf.create(load_params(run_params))
    log.info(f'Train Cfg:\n{train_cfg}')

    # Train the model
    log.info('Loading training data...')
    Xtr, ytr = th.load_data(train_cfg, train=True)
    log.info('Training model...')
    model = ttf.train_tf_model(train_cfg, Xtr, ytr)
    hftfmodel = model.model
    tok = model.tokenizer
    from transformers_interpret import SequenceClassificationExplainer
    cls_explainer = SequenceClassificationExplainer(
        hftfmodel,
        tok)

    # Load full dataset for word weights
    log.info(f'Loading train data from {cfg.train_path} and {cfg.test_path}')
    df_train = pd.read_pickle(cfg.train_path)
    df_test = pd.read_pickle(cfg.test_path)
    df = pd.concat([df_train, df_test])

    # Process word cloud data
    log.info("Processing word weights...")
    attrib_dict = {}
    n_skipped = 0
    for comment in tqdm(df.comment):
        seq_len = len(comment.split(' '))
        if seq_len > cfg.word_weight_len_thresh:
            n_skipped += 1
            continue
        else:
            attribs = cls_explainer(comment, class_name='LABEL_1')
            for w, score in attribs:
                try:
                    attrib_dict[w].append(score)
                except KeyError:
                    attrib_dict[w] = [score]
    log.info(f'Skipped for length (>{cfg.word_weight_len_thresh}) reasons: {n_skipped}')

    log.info('Calculating word scores...')
    word_scores = {k : {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} for k, v in attrib_dict.items()}
    word_scores = pd.DataFrame(word_scores).T
    word_scores = word_scores.sort_values('mean')
    log.info(f'Saving word scores to {cfg.word_scores_csv_path}')
    word_scores.to_csv(cfg.word_scores_csv_path)

    log.info('Generating and saving word clouds...')
    gen_wc_from_scores(cfg.word_cloud_pos_path, word_scores, max_words=cfg.wc_max_words,
        positive=True, count_thresh=cfg.wc_count_thresh,
        width=cfg.wc_width, height=cfg.wc_height, 
        figwidth=cfg.wc_figsize_width, figheight=cfg.wc_figsize_height,
        background_color=cfg.wc_background_color, colormap=cfg.wc_pos_colormap,
        exclude_partwords=cfg.wc_exclude_partwords)

    gen_wc_from_scores(cfg.word_cloud_neg_path, word_scores, max_words=cfg.wc_max_words,
        positive=False, count_thresh=cfg.wc_count_thresh,
        width=cfg.wc_width, height=cfg.wc_height, 
        figwidth=cfg.wc_figsize_width, figheight=cfg.wc_figsize_height,
        background_color=cfg.wc_background_color, colormap=cfg.wc_neg_colormap,
        exclude_partwords=cfg.wc_exclude_partwords)

def gen_wc_from_scores(fname, word_scores, max_words=500, positive=True, 
    count_thresh=10, width=800, height=400, figwidth=10, figheight=5, 
    background_color='white', colormap='Reds',
    exclude_partwords=True):
    wc = WordCloud(max_words=max_words, background_color=background_color, colormap=colormap, width=width, height=height)
    if exclude_partwords:
        word_scores = word_scores[~word_scores.index.str.contains('##')]
    wc.generate_from_frequencies(
        word_scores[
            (word_scores['count'] >= count_thresh) &
            ((word_scores['mean'] >= 0) if positive else (word_scores['mean'] < 0))
        ]['mean'].abs().to_dict()
    )

    plt.figure(figsize=(figwidth, figheight))
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig(fname, format='png')

if __name__ == '__main__':
    main()