"""Compile the corpus statistics that back Tables 1, 4 and 8 and the PoS-distribution figure.

Rebuild with: python -m bootstrapping.statistics

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import (
    CIRCSE_LEMMATA_PATH,
    CLTK_LEMMATA_PATH,
    CORPUS_DATA_PATHS,
    CORPUS_STATISTICS_PATH,
    DOM_LEMMATA_PATH,
    GOLD_STANDARD_PATH,
    ITTB_LEMMATA_PATH,
    LEWIS_LEMMATA_PATH,
    LLCT_LEMMATA_PATH,
    MARSEILLE_LEMMATA_PATH,
    OTTB_LEMMATA_PATH,
    PERSEUS_LEMMATA_PATH,
    PROIEL_LEMMATA_PATH,
    SEED_TRAINING_PATHS,
    TREEBANK_METADATA,
    UDANTE_LEMMATA_PATH,
)
from .extractors import extract_unique_lemmata, extract_upos_counts
from .io import load_file
from .sentence_metrics import dep_distance, gbsc_score, token_count, tree_depth

if TYPE_CHECKING:
    import conllu

# lexical resources compared against the Marseille lemma inventory
LEMMATA_SOURCES: dict[str, str] = {
    'cltk': CLTK_LEMMATA_PATH,
    'lewis': LEWIS_LEMMATA_PATH,
    'ittb': ITTB_LEMMATA_PATH,
    'llct': LLCT_LEMMATA_PATH,
    'perseus': PERSEUS_LEMMATA_PATH,
    'proiel': PROIEL_LEMMATA_PATH,
    'udante': UDANTE_LEMMATA_PATH,
    'circse': CIRCSE_LEMMATA_PATH,
    'ottb': OTTB_LEMMATA_PATH,
    'dom': DOM_LEMMATA_PATH,
}


def corpus_sources() -> dict[str, str]:
    """Return every corpus the statistics cover, keyed by label."""
    sources = dict(CORPUS_DATA_PATHS)
    for iteration in range(1, len(SEED_TRAINING_PATHS) + 1):
        sources[f'marseille_s{iteration}'] = SEED_TRAINING_PATHS[f'marseille_s{iteration}']
    sources['marseille_gold'] = GOLD_STANDARD_PATH
    return sources


def sentence_statistics(sentence: conllu.TokenList, number: int) -> dict[str, Any]:
    """Return the per-sentence record stored under 'sentence_data'."""
    distances = dep_distance(sentence)
    return {
        'no': number,
        'id': sentence.metadata.get('sent_id', f'sent_no_{number}'),
        'word_count': token_count(sentence),
        'tree_depth': tree_depth(sentence),
        'gbsc_score': gbsc_score(sentence),
        'gbsc_norm': gbsc_score(sentence, normalize=True),
        'gbsc_norm_non_func': gbsc_score(sentence, normalize=True, exclude_fw=True),
        'avg_dep_dist': distances['average'],
        'median_dep_dist': distances['median'],
        'max_dep_dist': distances['max'],
        'min_dep_dist': distances['min'],
        'std_dev_dep_dist': distances['std_dev'],
        'pos_distribution': extract_upos_counts(sentence),
    }


def corpus_statistics(label: str, sentences: list[conllu.TokenList]) -> dict[str, Any]:
    """Return the full statistics record for one corpus.

    Arguments:
        label: corpus key, e.g. 'ittb' or 'marseille_s3'. Metadata is looked up on the part
            before the first underscore, so seed batches inherit the Marseille metadata.
        sentences: the parsed corpus.

    """
    per_sentence = [sentence_statistics(s, i + 1) for i, s in enumerate(sentences)]
    word_counts = [s['word_count'] for s in per_sentence]

    metadata = TREEBANK_METADATA[label.split('_', 1)[0]]
    num_sentences = len(sentences)

    return {
        'genre': metadata['genre'],
        'dates': metadata['dates'],
        'start_year': metadata['start_year'],
        'end_year': metadata['end_year'],
        'num_sentences': num_sentences,
        'num_tokens': sum(len(s) for s in sentences),
        'num_words': sum(word_counts),
        'num_unique_lemmata': len(extract_unique_lemmata(sentences)),
        'upos_counts': extract_upos_counts(sentences),
        'avg_words_per_sentence': sum(word_counts) / num_sentences,
        'mean_words_per_sentence': float(np.mean(word_counts)),
        'median_words_per_sentence': float(np.median(word_counts)),
        'std_dev_words_per_sentence': float(np.std(word_counts)),
        'avg_tree_depth': sum(s['tree_depth'] for s in per_sentence) / num_sentences,
        'avg_gbsc_score': sum(s['gbsc_score'] for s in per_sentence) / num_sentences,
        'avg_gbsc_norm': sum(s['gbsc_norm'] for s in per_sentence) / num_sentences,
        'avg_gbsc_norm_non_func': sum(s['gbsc_norm_non_func'] for s in per_sentence) / num_sentences,
        'avg_dependency_distance': sum(s['avg_dep_dist'] for s in per_sentence) / num_sentences,
        'sentence_data': per_sentence,
    }


def lemmata_overlap(sources: dict[str, str] | None = None) -> dict[str, float]:
    """Return the percentage of Marseille lemmata attested in each lexical resource."""
    sources = sources or LEMMATA_SOURCES
    marseille = set(load_file(MARSEILLE_LEMMATA_PATH).split())
    return {
        name: len(marseille & set(load_file(path).split())) / len(marseille) * 100 for name, path in sources.items()
    }


def build_corpus_statistics(*, progress: bool = True) -> dict[str, Any]:
    """Compile statistics for every corpus, including the Marseille lemma overlap."""
    stats: dict[str, Any] = {}
    for label, path in corpus_sources().items():
        if progress:
            print(f'  {label}...', flush=True)
        stats[label] = corpus_statistics(label, load_file(path))

    stats['marseille']['lemmata_overlap'] = lemmata_overlap()
    return stats


def write_corpus_statistics(path: Path | str = CORPUS_STATISTICS_PATH, *, progress: bool = True) -> Path:
    """Compile the statistics and write them to 'path'."""
    stats = build_corpus_statistics(progress=progress)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # trailing newline matches the original notebook's json.dump output byte for byte
    destination.write_text(json.dumps(stats, indent=4) + '\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    print('Compiling corpus statistics...')
    written = write_corpus_statistics()
    size_mb = written.stat().st_size / 1_000_000
    print(f'Wrote {written} ({size_mb:.1f} MB)')
