"""Training-set construction and the Stanza training procedure."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GOLD_STANDARD_PATH, PROJECT_ROOT, SEED_TRAINING_PATHS
from .io import load_conllu_file

if TYPE_CHECKING:
    import conllu

# Stanza's training scripts, driven by environment variables from stanza/utils/default_paths.py
STANZA_TRAINING_SCRIPTS = ('run_pos', 'run_depparse', 'run_lemma')
DEFAULT_TRAIN_FRACTION = 0.9


def treebank_name(iteration: int) -> str:
    """Return the UD treebank directory name for an iteration, e.g. 'UD_Latin-Marseille_s3'."""
    return f'UD_Latin-Marseille_s{iteration}'


def treebank_code(iteration: int) -> str:
    """Return the UD treebank code for an iteration, e.g. 'la_marseille_s3'."""
    return f'la_marseille_s{iteration}'


def pooled_training_data(iteration: int) -> list[conllu.TokenList]:
    """Return every corrected seed sentence up to and including 'iteration'.

    Arguments:
        iteration: highest seed batch to include.

    """
    sentences: list[conllu.TokenList] = []
    for i in range(1, iteration + 1):
        sentences.extend(load_conllu_file(SEED_TRAINING_PATHS[f'marseille_s{i}']))

    # drop the two sentences that also appear in the evaluation gold standard
    gold_ids = {s.metadata['sent_id'] for s in load_conllu_file(GOLD_STANDARD_PATH)}

    return [s for s in sentences if s.metadata.get('sent_id') not in gold_ids]


def build_ud_dirs(
    output_root: Path | str,
    max_iteration: int = 10,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int | None = None,
) -> dict[int, Path]:
    """Write one UD treebank directory per iteration, each pooling all earlier batches.

    Arguments:
        output_root: directory used to create the treebank directories in.
                    Stanza reads this as '$UDBASE'.
        max_iteration: highest iteration to build.
        train_fraction: proportion of pooled sentences used for training.
        seed: RNG seed for the shuffle.

    Returns:
        Mapping of iteration number to the directory written.

    """
    root = Path(output_root)
    rng = random.Random(seed)  # nosec: corpus partitioning, not security-sensitive
    written: dict[int, Path] = {}

    for iteration in range(1, max_iteration + 1):
        sentences = pooled_training_data(iteration)
        rng.shuffle(sentences)

        cutoff = int((len(sentences) + 1) * train_fraction)
        directory = root / treebank_name(iteration)
        directory.mkdir(parents=True, exist_ok=True)
        code = treebank_code(iteration)

        for split, subset in (('train', sentences[:cutoff]), ('test', sentences[cutoff:])):
            path = directory / f'{code}-ud-{split}.conllu'
            with path.open('w', encoding='utf-8') as handle:
                for sentence in subset:
                    handle.write(sentence.serialize())

        written[iteration] = directory

    return written


def training_commands(
    iteration: int,
    udbase: Path | str,
    data_root: Path | str,
    wordvec_dir: Path | str,
) -> list[str]:
    """Return the stock Stanza 1.2.1 training commands for one iteration.

    Arguments:
        iteration: which seed iteration to train.
        udbase: directory holding the 'UD_Latin-Marseille_s{n}' treebanks ('$UDBASE').
        data_root: Stanza's working data directory ('$DATA_ROOT').
        wordvec_dir: directory holding the pretrained vectors ('$WORDVEC_DIR').

    """
    code = treebank_code(iteration)
    env = (
        f'UDBASE={udbase} DATA_ROOT={data_root} WORDVEC_DIR={wordvec_dir} '
        f'POS_DATA_DIR={data_root}/pos DEPPARSE_DATA_DIR={data_root}/depparse '
        f'LEMMA_DATA_DIR={data_root}/lemma'
    )
    return [f'{env} python -m stanza.utils.training.{script} {code}' for script in STANZA_TRAINING_SCRIPTS]


def default_udbase() -> Path:
    """Return the conventional location for generated UD treebanks in this repo."""
    return PROJECT_ROOT / 'data' / 'udbase'
