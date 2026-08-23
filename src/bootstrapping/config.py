"""Paths, registries, and constants for the bootstrapping package."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# marker files that identify the repository root
_ROOT_MARKERS = ('pyproject.toml', '.git')


def find_project_root() -> Path:
    """Return the repository root, which is where 'data/' lives."""
    override = os.getenv('PROJECT_ROOT')
    if override:
        return Path(override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS) and (parent / 'data').is_dir():
            return parent

    msg = (
        'Could not locate the project root. Expected an ancestor directory containing '
        f'one of {_ROOT_MARKERS} alongside a data/ directory. Set PROJECT_ROOT to override.'
    )
    raise RuntimeError(msg)


PROJECT_ROOT = find_project_root()

STANZA_RESOURCES_DIR = Path(
    os.getenv('STANZA_RESOURCES_DIR', default=Path.home() / 'stanza_resources'),
).expanduser()

LATIN_MODEL_DIR = STANZA_RESOURCES_DIR / 'la'
ENSEMBLE_LEMMA_SUFFIX = '_custom'


def lemma_model_path(package: str, *, ensemble: bool = True) -> str:
    """Return the path to a Stanza lemma model.

    Arguments:
        package: Stanza package name, e.g. 'marseille_s9' or 'ittb'.
        ensemble: return the patched variant, i.e. the model carrying the domain lexicon.

    """
    suffix = ENSEMBLE_LEMMA_SUFFIX if ensemble else ''
    return str(LATIN_MODEL_DIR / 'lemma' / f'{package}{suffix}.pt')


# gold standard
GOLD_STANDARD_PATH = f'{PROJECT_ROOT}/data/gold/la_marseille-ud-gold.conllu'

# seed training data (i.e. manually corrected outputs from training models)
SEED_TRAINING_PATHS = {
    'marseille_s1': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s1.conllu',
    'marseille_s2': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s2.conllu',
    'marseille_s3': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s3.conllu',
    'marseille_s4': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s4.conllu',
    'marseille_s5': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s5.conllu',
    'marseille_s6': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s6.conllu',
    'marseille_s7': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s7.conllu',
    'marseille_s8': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s8.conllu',
    'marseille_s9': f'{PROJECT_ROOT}/data/seeds/corrected/la_marseille-ud-fixeddep-dev_s9.conllu',
}

# seed predictions from preceeding model (i.e. ittb for iteration 1, marseille_s-X for iteration X)
SEED_PREDICTED_PATHS = {
    'marseille_s1': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s1.conllu',
    'marseille_s2': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s2.conllu',
    'marseille_s3': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s3.conllu',
    'marseille_s4': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s4.conllu',
    'marseille_s5': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s5.conllu',
    'marseille_s6': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s6.conllu',
    'marseille_s7': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s7.conllu',
    'marseille_s8': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s8.conllu',
    'marseille_s9': f'{PROJECT_ROOT}/data/seeds/predictions/la_marseille-ud-rawdep-dev_s9.conllu',
}

# seed raw data (i.e. unprocessed data for initial model predictions)
SEED_RAW_PATHS = {
    'marseille_s1': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s1.conllu',
    'marseille_s2': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s2.conllu',
    'marseille_s3': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s3.conllu',
    'marseille_s4': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s4.conllu',
    'marseille_s5': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s5.conllu',
    'marseille_s6': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s6.conllu',
    'marseille_s7': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s7.conllu',
    'marseille_s8': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s8.conllu',
    'marseille_s9': f'{PROJECT_ROOT}/data/seeds/base/la_marseille-ud-base_s9.conllu',
}

# corpus data paths (this are conllu files with all data for each corpus)
CORPUS_DATA_PATHS = {
    'ittb': f'{PROJECT_ROOT}/data/corpora/la_ittb-ud-full.conllu',
    'llct': f'{PROJECT_ROOT}/data/corpora/la_llct-ud-full.conllu',
    'perseus': f'{PROJECT_ROOT}/data/corpora/la_perseus-ud-full.conllu',
    'proiel': f'{PROJECT_ROOT}/data/corpora/la_proiel-ud-full.conllu',
    'marseille': f'{PROJECT_ROOT}/data/corpora/la_marseille-ud-full.conllu',
    'udante': f'{PROJECT_ROOT}/data/corpora/la_udante-ud-full.conllu',
    'circse': f'{PROJECT_ROOT}/data/corpora/la_circse-ud-full.conllu',
}

# path to json file with corpus statistics for files in CORPUS_DATA_PATHS and SEED_RAW_PATHS
CORPUS_STATISTICS_PATH = f'{PROJECT_ROOT}/data/reference/corpus_statistics.json'

# minutes spent annotating each seed batch, keyed by the model that batch produced
ANNOTATION_TIMES_PATH = f'{PROJECT_ROOT}/data/reference/brat_annotation_times.csv'

# per-record metadata for the DALME inventories: date, place, document type
INVENTORY_METADATA_PATH = f'{PROJECT_ROOT}/data/reference/inventory_metadata.csv'

# these conllu files are the result of evaluating a model on the gold standard
EVALUATION_DATA_PATHS = {
    'ittb': f'{PROJECT_ROOT}/data/evaluation/la_ittb-ud-gold_test.conllu',
    'ittb_ens': f'{PROJECT_ROOT}/data/evaluation/la_ittb_ens-ud-gold_test.conllu',  # with ensemble lemmatizer
    'llct': f'{PROJECT_ROOT}/data/evaluation/la_llct-ud-gold_test.conllu',
    'llct_ens': f'{PROJECT_ROOT}/data/evaluation/la_llct_ens-ud-gold_test.conllu',  # with ensemble lemmatizer
    'perseus': f'{PROJECT_ROOT}/data/evaluation/la_perseus-ud-gold_test.conllu',
    'perseus_ens': f'{PROJECT_ROOT}/data/evaluation/la_perseus_ens-ud-gold_test.conllu',  # with ensemble lemmatizer
    'proiel': f'{PROJECT_ROOT}/data/evaluation/la_proiel-ud-gold_test.conllu',
    'proiel_ens': f'{PROJECT_ROOT}/data/evaluation/la_proiel_ens-ud-gold_test.conllu',  # with ensemble lemmatizer
    'udante': f'{PROJECT_ROOT}/data/evaluation/la_udante-ud-gold_test.conllu',
    'udante_ens': f'{PROJECT_ROOT}/data/evaluation/la_udante_ens-ud-gold_test.conllu',  # with ensemble lemmatizer
    'marseille_s1': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s1-ud-gold_test.conllu',
    'marseille_s2': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s2-ud-gold_test.conllu',
    'marseille_s3': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s3-ud-gold_test.conllu',
    'marseille_s4': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s4-ud-gold_test.conllu',
    'marseille_s5': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s5-ud-gold_test.conllu',
    'marseille_s6': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s6-ud-gold_test.conllu',
    'marseille_s7': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s7-ud-gold_test.conllu',
    'marseille_s8': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s8-ud-gold_test.conllu',
    'marseille_s9': f'{PROJECT_ROOT}/data/evaluation/la_marseille_s9-ud-gold_test.conllu',
}

# json file with aggregated evaluation results (i.e. files in EVALUATION_DATA_PATHS evaluated against GOLD_STANDARD_PATH)
EVALUATION_RESULTS_PATH = f'{PROJECT_ROOT}/data/evaluation/results.json'

# figures written by the analysis notebooks
CHARTS_DIR = f'{PROJECT_ROOT}/data/charts'


def chart_path(filename: str) -> str:
    """Return the absolute path for a chart, creating 'CHARTS_DIR' if needed."""
    directory = Path(CHARTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / filename)


# list of morphological evaluation metrics
MORPHO_EVAL_METRICS = ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'AllTags']

# list of syntactic evaluation metrics
SYNTACTIC_EVAL_METRICS = ['UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX', 'ELAS', 'EULAS']

# PoS tag groups for morphological profile analyses
POS_TAG_GROUPS = {
    'Adjectives/Adverbs': ['ADJ', 'ADV'],
    'Verbs': ['VERB', 'AUX'],
    'Nominals': ['NOUN', 'PROPN', 'DET', 'PRON', 'NUM'],
    'Other': ['ADP', 'CCONJ', 'SCONJ', 'PART', 'INTJ'],
}


# list of iterations in order with label (name) and key used in results dictionaries
REPORTED_ITERATIONS = [
    {'name': 'Baseline (ITTB)', 'key': 'ittb'},
    *[{'name': f'Iteration {i}', 'key': f'marseille_s{i}'} for i in range(1, 10)],
]
REPORTED_SEED_KEYS = [i['key'] for i in REPORTED_ITERATIONS if i['key'].startswith('marseille_s')]
FINAL_REPORTED_KEY = REPORTED_SEED_KEYS[-1]

# list of models with label (name) and key used in results dictionaries
MODELS = [
    {'name': 'ITTB', 'key': 'ittb'},
    {'name': 'ITTB (ensemble)', 'key': 'ittb_ens'},
    {'name': 'LLCT', 'key': 'llct'},
    {'name': 'LLCT (ensemble)', 'key': 'llct_ens'},
    {'name': 'Perseus', 'key': 'perseus'},
    {'name': 'Perseus (ensemble)', 'key': 'perseus_ens'},
    {'name': 'PROIEL', 'key': 'proiel'},
    {'name': 'PROIEL (ensemble)', 'key': 'proiel_ens'},
    {'name': 'UDante', 'key': 'udante'},
    {'name': 'UDante (ensemble)', 'key': 'udante_ens'},
    {'name': 'Marseille S9', 'key': 'marseille_s9'},
]

# The corpora the paper compares, keyed as in CORPUS_DATA_PATHS
TREEBANK_NAMES = {
    'ittb': 'ITTB',
    'llct': 'LLCT',
    'perseus': 'Perseus',
    'proiel': 'PROIEL',
    'marseille': 'DALME-Marseille',
    'udante': 'UDante',
}

# lemmata lists
CLTK_LEMMATA_PATH = 'data/reference/cltk_lemmata.txt'  # list of all lemmata in CLTK
LEWIS_LEMMATA_PATH = 'data/reference/lewis_lemmata.txt'  # list of all lemmata in Lewis' An Elementary Latin Dictionary
LATIN_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/all_latin_lemmata.txt'
OCCITAN_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/all_occitan_lemmata.txt'
CLTK_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/cltk_lemmata.txt'
LEWIS_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/lewis_lemmata.txt'
ITTB_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/ittb_lemmata.txt'
LLCT_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/llct_lemmata.txt'
PERSEUS_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/perseus_lemmata.txt'
PROIEL_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/proiel_lemmata.txt'
MARSEILLE_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/marseille_lemmata.txt'
CIRCSE_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/circse_lemmata.txt'
UDANTE_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/udante_lemmata.txt'
OTTB_LEMMATA_PATH = f'{PROJECT_ROOT}/data/lemmata/occitan_ttb_lemmata.txt'
DOM_LEMMATA_PATH = f'{PROJECT_ROOT}/data/reference/dom_online_lemmata.txt'


TREEBANK_METADATA = {
    'marseille': {
        'genre': 'documentary',
        'dates': 'AD 1258-1446',
        'start_year': 1258,
        'end_year': 1446,
    },
    'ittb': {
        'genre': 'literary',
        'dates': 'AD 1256-1274',
        'start_year': 1256,
        'end_year': 1274,
    },
    'perseus': {
        'genre': 'literary',
        'dates': '63 BC-AD 382',
        'start_year': -63,
        'end_year': 382,
    },
    'llct': {
        'genre': 'documentary',
        'dates': 'AD 774-897',
        'start_year': 774,
        'end_year': 897,
    },
    'proiel': {
        'genre': 'literary',
        'dates': '58 BC-AD 450',
        'start_year': -58,
        'end_year': 450,
    },
    'udante': {
        'genre': 'literary',
        'dates': 'AD 1283-1320',
        'start_year': 1283,
        'end_year': 1320,
    },
    'circse': {
        'genre': 'literary',
        'dates': 'AD 54-98',
        'start_year': 54,
        'end_year': 98,
    },
}
