"""
Load results from the MTEB benchmark and plot them
"""


# %%
# Load files
import pandas as pd
import collections
import json
import os
import numpy as np


def get_paths():
    files = collections.defaultdict(list)
    for model_dir in os.listdir("mteb_results/results"):
        results_model_dir = os.path.join("mteb_results/results", model_dir)
        for res_file in os.listdir(results_model_dir):
            if res_file.endswith(".json"):
                results_model_file = os.path.join(results_model_dir, res_file)
                files[model_dir].append(results_model_file)
    return files


files = get_paths()# %%

# %%

#data = list()
#
#for model, model_files in files.items():
#    print(model)
#    for i, model_file in enumerate(model_files):
#        with open(model_file, "r") as f:
#            results = pd.read_json(f)
#        results["model"] = model
#        results["run"] = i
#        data.append(results)


#######################################################################
# Constants (borrowed from the results.py of the MTEB)

SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

# Use "train" split instead
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
# Use "validation" split instead
VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MSMARCO-PL", "MultilingualSentiment", "Ocnli"]
# Use "dev" split instead
DEV_SPLIT = ["CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "MSMARCO-PL", "T2Reranking", "T2Retrieval", "VideoRetrieval"]
# Use "test.full" split
TESTFULL_SPLIT = ["OpusparcusPC"]

EVAL_LANGS = ['af', 'afr-eng', 'am', "amh", 'amh-eng', 'ang-eng', 'ar', 'ar-ar', 'ara-eng', 'arq-eng', 'arz-eng', 'ast-eng', 'awa-eng', 'az', 'aze-eng', 'bel-eng', 'ben-eng', 'ber-eng', 'bn', 'bos-eng', 'bre-eng', 'bul-eng', 'cat-eng', 'cbk-eng', 'ceb-eng', 'ces-eng', 'cha-eng', 'cmn-eng', 'cor-eng', 'csb-eng', 'cy', 'cym-eng', 'da', 'dan-eng', 'de', 'de-fr', 'de-pl', 'deu-eng', 'dsb-eng', 'dtp-eng', 'el', 'ell-eng', 'en', 'en-ar', 'en-de', 'en-en', 'en-tr', 'eng', 'epo-eng', 'es', 'es-en', 'es-es', 'es-it', 'est-eng', 'eus-eng', 'fa', 'fao-eng', 'fi', 'fin-eng', 'fr', 'fr-en', 'fr-pl', 'fra', 'fra-eng', 'fry-eng', 'gla-eng', 'gle-eng', 'glg-eng', 'gsw-eng', 'hau', 'he', 'heb-eng', 'hi', 'hin-eng', 'hrv-eng', 'hsb-eng', 'hu', 'hun-eng', 'hy', 'hye-eng', 'ibo', 'id', 'ido-eng', 'ile-eng', 'ina-eng', 'ind-eng', 'is', 'isl-eng', 'it', 'it-en', 'ita-eng', 'ja', 'jav-eng', 'jpn-eng', 'jv', 'ka', 'kab-eng', 'kat-eng', 'kaz-eng', 'khm-eng', 'km', 'kn', 'ko', 'ko-ko', 'kor-eng', 'kur-eng', 'kzj-eng', 'lat-eng', 'lfn-eng', 'lit-eng', 'lin', 'lug', 'lv', 'lvs-eng', 'mal-eng', 'mar-eng', 'max-eng', 'mhr-eng', 'mkd-eng', 'ml', 'mn', 'mon-eng', 'ms', 'my', 'nb', 'nds-eng', 'nl', 'nl-ende-en', 'nld-eng', 'nno-eng', 'nob-eng', 'nov-eng', 'oci-eng', 'orm', 'orv-eng', 'pam-eng', 'pcm', 'pes-eng', 'pl', 'pl-en', 'pms-eng', 'pol-eng', 'por-eng', 'pt', 'ro', 'ron-eng', 'ru', 'run', 'rus-eng', 'sl', 'slk-eng', 'slv-eng', 'spa-eng', 'sna', 'som', 'sq', 'sqi-eng', 'srp-eng', 'sv', 'sw', 'swa', 'swe-eng', 'swg-eng', 'swh-eng', 'ta', 'tam-eng', 'tat-eng', 'te', 'tel-eng', 'tgl-eng', 'th', 'tha-eng', 'tir', 'tl', 'tr', 'tuk-eng', 'tur-eng', 'tzl-eng', 'uig-eng', 'ukr-eng', 'ur', 'urd-eng', 'uzb-eng', 'vi', 'vie-eng', 'war-eng', 'wuu-eng', 'xho', 'xho-eng', 'yid-eng', 'yor', 'yue-eng', 'zh', 'zh-CN', 'zh-TW', 'zh-en', 'zsm-eng']

EXTERNAL_MODEL_TO_SIZE = {
    "DanskBERT": 125,
    "LASER2": 43,
    "LLM2Vec-Llama-supervised": 6607,
    "LLM2Vec-Llama-unsupervised": 6607,
    "LLM2Vec-Mistral-supervised": 7111,
    "LLM2Vec-Mistral-unsupervised": 7111,
    "LLM2Vec-Sheared-Llama-supervised": 1280,
    "LLM2Vec-Sheared-Llama-unsupervised": 1280,
    "LaBSE": 471,
    "allenai-specter": 110,
    "all-MiniLM-L12-v2": 33,
    "all-MiniLM-L6-v2": 23,
    "all-mpnet-base-v2": 110,
    "bert-base-10lang-cased": 138,
    "bert-base-15lang-cased": 138,
    "bert-base-25lang-cased": 138,
    "bert-base-multilingual-cased": 179,
    "bert-base-multilingual-uncased": 168,
    "bert-base-uncased": 110,
    "bert-base-swedish-cased": 125,
    "bge-base-zh-v1.5": 102,
    "bge-large-zh-v1.5": 326,
    "bge-large-zh-noinstruct": 326,
    "bge-small-zh-v1.5": 24,
    "camembert-base": 111,
    "camembert-large": 338,
    "cross-en-de-roberta-sentence-transformer": 278,
    "contriever-base-msmarco": 110,
    "distilbert-base-25lang-cased": 110,
    "distilbert-base-en-fr-cased": 110,
    "distilbert-base-en-fr-es-pt-it-cased": 110,
    "distilbert-base-fr-cased": 110,
    "distilbert-base-uncased": 110,
    "distiluse-base-multilingual-cased-v2": 135,
    "dfm-encoder-large-v1": 355,
    "dfm-sentence-encoder-large-1": 355,
    "e5-base": 110,
    "e5-large": 335,
    "e5-mistral-7b-instruct": 7111,
    "e5-small": 33,
    "electra-small-nordic": 23,
    "electra-small-swedish-cased-discriminator": 16,
    "flaubert_base_cased": 138,
    "flaubert_base_uncased": 138,
    "flaubert_large_cased": 372,
    "gbert-base": 110,
    "gbert-large": 337,
    "gelectra-base": 110,
    "gelectra-large": 335,
    "glove.6B.300d": 120,
    "google-gecko.text-embedding-preview-0409": 1200,
    "google-gecko-256.text-embedding-preview-0409": 1200,
    "gottbert-base": 127,
    "gtr-t5-base": 110,
    "gtr-t5-large": 168,
    "gtr-t5-xl": 1240,
    "gtr-t5-xxl": 4865,
    "herbert-base-retrieval-v2": 125,
    "komninos": 134,
    "luotuo-bert-medium": 328,
    "m3e-base": 102,
    "m3e-large": 102,
    "msmarco-bert-co-condensor": 110,
    "multi-qa-MiniLM-L6-cos-v1": 23,
    "multilingual-e5-base": 278,
    "multilingual-e5-small": 118,
    "multilingual-e5-large": 560,
    "nb-bert-base": 179,
    "nb-bert-large": 355,
    "nomic-embed-text-v1.5-64": 138,
    "nomic-embed-text-v1.5-128": 138,
    "nomic-embed-text-v1.5-256": 138,
    "nomic-embed-text-v1.5-512": 138,
    "norbert3-base": 131,
    "norbert3-large": 368,
    "paraphrase-multilingual-mpnet-base-v2": 278,
    "paraphrase-multilingual-MiniLM-L12-v2": 118,
    "sentence-camembert-base": 110,
    "sentence-camembert-large": 337,
    "sentence-croissant-llm-base": 1280,
    "sentence-bert-swedish-cased": 125,
    "sentence-t5-base": 110,
    "sentence-t5-large": 168,
    "sentence-t5-xl": 1240,
    "sentence-t5-xxl": 4865,
    "silver-retriever-base-v1": 125,
    "sup-simcse-bert-base-uncased": 110,
    "st-polish-paraphrase-from-distilroberta": 125,
    "st-polish-paraphrase-from-mpnet": 125,    
    "text2vec-base-chinese": 102,
    "text2vec-large-chinese": 326,
    "unsup-simcse-bert-base-uncased": 110,
    "use-cmlm-multilingual": 472,
    #"voyage-law-2": 1220,
    "voyage-lite-02-instruct": 1220,
    "xlm-roberta-base": 279,
    "xlm-roberta-large": 560,
}


#######################################################################
# Our addition to the table above

EXTERNAL_MODEL_TO_SIZE_MINE = {
    'SGPT-5.8B-weightedmean-nli-bitfit': 5.8,
    'sgpt-bloom-7b1-msmarco': 7,
    'SGPT-125M-weightedmean-msmarco-specb-bitfit': 125. / 1024,
    'SGPT-125M-weightedmean-nli-bitfit': 125. / 1024,
    'SGPT-125M-weightedmean-msmarco-specb-bitfit-que': 125. / 1024,
    'gte-Qwen1.5-7B-instruct': 7,
    'SGPT-2.7B-weightedmean-msmarco-specb-bitfit': 2.7,
    'SGPT-5.8B-weightedmean-msmarco-specb-bitfit': 5.8,
    'SGPT-5.8B-weightedmean-msmarco-specb-bitfit-que': 5.8,
}

#######################################################################
# Data loading

out = []

# %%
data = list()

for model, model_files in files.items():
 print(model)
 model_size = EXTERNAL_MODEL_TO_SIZE.get(model, np.nan)
 #model_size_gb = round(model_size * 1e6 * 4 / 1024**3, 2)
 if model in EXTERNAL_MODEL_TO_SIZE_MINE:
    model_size = 1000 * EXTERNAL_MODEL_TO_SIZE_MINE[model]
 for path in model_files:
    with open(path, encoding="utf-8") as f:
        res_dict = json.load(f)
        ds_name = res_dict["mteb_dataset_name"]
        split = "test"
        if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
            split = "train"
        elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
            split = "validation"
        elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
            split = "dev"
        elif (ds_name in TESTFULL_SPLIT) and ("test.full" in res_dict):
            split = "test.full"
        elif "test" not in res_dict:
            print(f"Skipping {ds_name} as split {split} not present.")
            continue
        res_dict = res_dict.get(split)
        is_multilingual = any(x in res_dict for x in EVAL_LANGS)
        langs = res_dict.keys() if is_multilingual else ["en"]
        for lang in langs:
            if lang in SKIP_KEYS: continue
            test_result_lang = res_dict.get(lang) if is_multilingual else res_dict
            for metric, score in test_result_lang.items():
                if not isinstance(score, dict):
                    score = {metric: score}
                for sub_metric, sub_score in score.items():
                    if any(x in sub_metric for x in SKIP_KEYS): continue
                    out.append({
                        "mteb_dataset_name": ds_name,
                        "eval_language": lang if is_multilingual else "",
                        "metric": (f"{metric}_{sub_metric}"
                                  if metric != sub_metric else metric),
                        "score": sub_score * 100,
                        "model": model,
                        "model_size": model_size,
                    })

df = pd.DataFrame(out)
df['eval_language'] = df['eval_language'].replace('fra', 'fr')

#########################################################################
# Extract good subsets of results and plot them

import matplotlib.pyplot as plt

df = df.query('eval_language != ""')
# Find the benchmark situation where there is most data
value_counts = df.drop(['score', 'model_size', 'model'],
                       axis=1).value_counts().reset_index()
count_max = value_counts['count'].max()
most_data_configuration = value_counts.query("count == @count_max")



for _, config in most_data_configuration.iterrows():
    mteb_dataset_name = config['mteb_dataset_name']
    eval_language = config['eval_language']
    metric = config['metric']
    this_df = df.query('eval_language == @eval_language '
                       '& mteb_dataset_name == @mteb_dataset_name '
                       '& metric == @metric')

    this_df = this_df.dropna()
    x = this_df['model_size']
    y = this_df['score']

    plt.figure(figsize=(4, 3))
    plt.scatter(x, y)

    ax = plt.gca()
    ax.set_xscale('log')

    plt.xticks([1e2, 1e3], ['100M', '1B'])
    plt.xlabel('Model size')
    plt.ylabel('Score')
    plt.text(.95, .03, f'{metric} on\n{mteb_dataset_name}',
            ha='right', va='bottom', transform=ax.transAxes)
    plt.subplots_adjust(left=.13, bottom=0.155, right=.99, top=.99)

    # Name the higest value
    best_model = this_df.sort_values(by='score').iloc[-1]

    # Break model names that are too long
    model_name = ['']
    for substr in best_model['model'].split('-'):
        if len(model_name[-1]) < 19:
            model_name[-1] += substr + ' '
        else:
            model_name.append(substr)
    model_name = '\n'.join(model_name)
    annotation = plt.annotate(model_name,
                              xy=(best_model['model_size'],
                                  best_model['score']),
                              xytext=(best_model['model_size'] - 100,
                                      best_model['score'] + 1),
                              ha='right', va='top',
                              color='.2')

    plt.savefig(f'{metric}_{mteb_dataset_name[:5]}.pdf')

