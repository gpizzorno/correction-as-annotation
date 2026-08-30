r"""LaTeX emitters for the paper's tables.

Every table in 'tex/paper.tex' is generated from data in the repository:

    python -m bootstrapping.report.tables
    python -m bootstrapping.report.tables --install   # and copy into tex/tables/
    python -m bootstrapping.report.tables --check     # verify the paper is current

Each file is named after its '\label' and the paper reads it with '\input{tables/label}'.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

from bootstrapping.config import (
    CORPUS_STATISTICS_PATH,
    EVALUATION_DATA_PATHS,
    EVALUATION_RESULTS_PATH,
    GOLD_STANDARD_PATH,
    MODELS,
    PROJECT_ROOT,
    REPORTED_ITERATIONS,
    REPORTED_SEED_KEYS,
    SEED_PREDICTED_PATHS,
    SEED_TRAINING_PATHS,
    TREEBANK_NAMES,
)
from bootstrapping.io import load_file

BUILD_DIR = Path(PROJECT_ROOT) / 'build' / 'tables'
MODERN_RESULTS_PATH = Path(PROJECT_ROOT) / 'data' / 'evaluation' / 'modern' / 'zero_shot_results.json'
PAPER_PATH = Path(PROJECT_ROOT) / 'tex' / 'paper.tex'
INSTALL_DIR = Path(PROJECT_ROOT) / 'tex' / 'tables'
INPUT_DIR_NAME = 'tables'
TABLE_INPUT = re.compile(r'\\input\{' + INPUT_DIR_NAME + r'/([^}]+)\}')
TABLE_BLOCK = re.compile(r'\n[ \t]*\\begin\{table\*?\}.*?\\end\{table\*?\}', re.DOTALL)
LABEL = re.compile(r'\\label\{([^}]*)\}')

# treebanks in the chronological order the paper presents them
TREEBANK_ORDER = ['perseus', 'proiel', 'llct', 'ittb', 'udante', 'marseille']
BASELINE_ORDER = ['ittb', 'llct', 'perseus', 'proiel', 'udante']

MORPHO_COLUMNS = ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'Lemmas (ensemble)', 'AllTags']
SYNTAX_COLUMNS = ['UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX']
HEADINGS = {
    'UPOS': 'UPoS',
    'XPOS': 'XPoS',
    'UFeats': 'Features',
    'AllTags': 'All Tags',
    'Lemmas': 'Lemmata',
    'Lemmas (ensemble)': 'Ensemble',
}

# a construction type needs more than this many sentences before its accuracy means anything
INTERPRETABLE_MIN_SENTENCES = 5
EM_DASH = '—'
COL_WIDTH_IN = 3.04


def _header(text: str, expand: str | None = None) -> str:
    output = f'{expand}{text}{expand}' if expand and len(text) < 5 else text  # noqa: PLR2004
    return f'\\scriptsize\\textbf{{{output}}}'


def _iteration(key: str) -> str:
    """Render an iteration key as the maths subscript."""
    return f'$s_{{{key.split("_s")[1]}}}$'


def _dates(text: str) -> str:
    """Typeset a date range with an en-dash."""
    return text.replace('-', '--')


def _half_up(value: float, places: int = 2) -> str:
    """Round half away from zero."""
    from decimal import ROUND_HALF_UP, Decimal  # noqa: PLC0415

    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _bare(text: str) -> str:
    """Drop the leading zero from a formatted score."""
    return re.sub(r'(?<![\d.])0\.', '.', text)


def _major_minor(version: str) -> str:
    """Label a Stanza releases by their feature version."""
    return '.'.join(version.split('.')[:2])


def _latex_table(  # noqa: PLR0913
    *,
    spec: str,
    header_rows: list[list[str]],
    body_rows: list[list[str]],
    caption: str,
    label: str,
    wide: bool = False,
    prologue: list[str] | None = None,
    target: str | None = None,
    setup: str | None = None,
) -> str:
    r"""Assemble one table.

    Arguments:
        spec: tabulary column specification, e.g. 'LRRCC'.
        header_rows: one list of already-formatted cells per header line.
        body_rows: the data rows, cells already formatted as strings.
        caption: caption text, LaTeX.
        label: the '\\label' value, e.g. 'tab:morph-curve'.
        wide: use 'table*' (full page width) rather than 'table'.
        prologue: raw lines inserted after the header, e.g. a '\\cmidrule'.
        target: the target module, either tabulary (default) or tabularx.
        setup: raw commands emitted inside the float, before the tabular begins.

    """
    environment = 'table*' if wide else 'table'
    position = 't' if wide else 'H'
    package = 'tabulary' if target is None else target
    width = f'{COL_WIDTH_IN}in' if target == 'tabularx' and not wide else '\\textwidth'
    lines = [
        f'\\begin{{{environment}}}[{position}]',
        ' \\centering',
        ' \\sffamily\\footnotesize',
        *([f' {setup}'] if setup else []),
        f' \\begin{{{package}}}{{{width}}}{{{spec}}}',
        '  \\toprule',
    ]
    for index, row in enumerate(header_rows):
        lines.append('  ' + ' & '.join(row) + ' \\\\')
        if prologue and index == 0:
            lines += ['  ' + line for line in prologue]
    lines.append('  \\midrule')
    for row in body_rows:
        bare_rule = len(row) == 1 and row[0].strip() in {'\\midrule', '\\bottomrule', '\\toprule'}
        lines.append('  ' + ' & '.join(row) + ('' if bare_rule else ' \\\\'))
    lines += [
        '  \\bottomrule',
        f'  \\end{{{package}}}',
        f'  \\caption{{{caption}}}',
        f'  \\label{{{label}}}  % chktex 24',
        f'\\end{{{environment}}}',
    ]
    return '\n'.join(lines) + '\n'


# emitting all ten tables reads these eleven times between them, and corpus_statistics.json is
# 51 MB. caching turns a full regeneration from ~40 seconds into a few
@cache
def _results() -> dict[str, Any]:
    return load_file(EVALUATION_RESULTS_PATH)  # type: ignore[no-any-return]


@cache
def _corpus_stats() -> dict[str, Any]:
    return load_file(CORPUS_STATISTICS_PATH)  # type: ignore[no-any-return]


def treebank_stats() -> str:
    """Table 1: size and composition of the comparison treebanks."""
    stats = _corpus_stats()
    names = [
        'Treebank',
        'Genre',
        'Temporal Coverage',
        'Sentences',
        'Tokens',
        'Words',
        'Unique Lem.',
        'Words/sent.',
    ]
    rows = []
    for key in TREEBANK_ORDER:
        entry = stats[key]
        rows.append(
            [
                TREEBANK_NAMES[key],
                entry['genre'],
                _dates(entry['dates']),
                f'{entry["num_sentences"]:,}',
                f'{entry["num_tokens"]:,}',
                f'{entry["num_words"]:,}',
                f'{entry["num_unique_lemmata"]:,}',
                str(round(entry['avg_words_per_sentence'], 2)),
            ],
        )

    return _latex_table(
        spec='LLLRRRRR',
        header_rows=[[_header(name) for name in names]],
        body_rows=rows,
        caption=(
            'Corpus statistics for existing treebanks. Unique lemmata are counted after '
            'case-folding and stripping digits and punctuation, so that variants of one lemma are '
            'not counted separately. For DALME-Marseille this gives 3,499 types against 3,585 '
            'distinct lemma strings.'
        ),
        label='tab:treebank-stats',
        wide=True,
    )


def pre_trained_performance() -> str:
    """Table 2: every pre-trained model on the gold standard, with the ensemble lemma column."""
    results = _results()['data']
    by_key = {model['key']: model['name'] for model in MODELS}

    body = []
    for key, name in by_key.items():
        if key.endswith('_ens') or key.startswith('marseille_s'):
            continue
        evaluation = results[key]['evaluation']
        ensemble = results[f'{key}_ens']['evaluation']
        cells = [name]
        for column in MORPHO_COLUMNS + SYNTAX_COLUMNS:
            source = ensemble if column == 'Lemmas (ensemble)' else evaluation
            metric = 'Lemmas' if column == 'Lemmas (ensemble)' else column
            cells.append(_bare(f'{source[metric]["f1"]:.2f}'))
        body.append(cells)

    spec = 'X|' + 'c' * len(MORPHO_COLUMNS) + '|' + 'c' * len(SYNTAX_COLUMNS)
    total = 1 + len(MORPHO_COLUMNS) + len(SYNTAX_COLUMNS)
    group = [
        '',
        f'\\multicolumn{{{len(MORPHO_COLUMNS)}}}{{c|}}{{{_header("Morphological")}}}',
        f'\\multicolumn{{{len(SYNTAX_COLUMNS)}}}{{c}}{{{_header("Syntactic")}}}',
    ]
    names = [_header('Model')] + [_header(HEADINGS.get(c, c), expand='~') for c in MORPHO_COLUMNS + SYNTAX_COLUMNS]

    return _latex_table(
        spec=spec,
        header_rows=[group, names],
        prologue=[f'\\cmidrule{{2-{total}}}'],
        body_rows=body,
        caption=(
            'Performance of models trained on existing Latin treebanks, applied to the '
            '200-sentence DALME-Marseille gold standard. Metrics are discussed in Section~\\ref{sec:metrics}.'
        ),
        label='tab:pre-trained-performance',
        wide=True,
        target='tabularx',
    )


def modern_stanza() -> str:
    """Table 3: the same off-the-shelf packages, 1.2.1 versus current Stanza."""
    modern = json.loads(MODERN_RESULTS_PATH.read_text(encoding='utf-8'))
    legacy = _results()['data']
    columns = ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'AllTags', 'UAS', 'LAS', 'MLAS', 'BLEX']

    body = []
    for package, entry in modern['data'].items():
        old = legacy[package]['evaluation']
        new = entry['evaluation']
        body.append(
            [
                TREEBANK_NAMES.get(package, package.upper()),
                '1.2.1',
                *[_bare(f'{old[c]["f1"]:.2f}') for c in columns],
            ],
        )
        body.append(
            ['', f'{_major_minor(modern["stanza_version"])}', *[_bare(f'{new[c]["f1"]:.2f}') for c in columns]],
        )
        body.append(['\\midrule'])

    # remove the last midrule, which is not followed by a row
    body.pop()

    return _latex_table(
        spec='XX' + 'c' * len(columns),
        header_rows=[
            [
                _header('Treebank'),
                _header('Stanza'),
                *[_header(HEADINGS.get(h, h)) for h in columns],
            ],
        ],
        body_rows=body,
        caption=(
            'Current off-the-shelf Stanza Latin models on the same gold standard versus version 1.2.1. '
            'Neither row uses the domain lexicon, so both are directly comparable.'
        ),
        label='tab:modern-stanza',
        wide=True,
        target='tabularx',
    )


def length_correlations() -> str:
    """Table 4: how strongly sentence length predicts tree depth, per corpus."""
    from scipy import stats as scipy_stats  # noqa: PLC0415

    stats = _corpus_stats()
    ranked = []
    for key in TREEBANK_ORDER:
        # sentences of fewer than two non-punctuation words have a degenerate tree
        records = [s for s in stats[key]['sentence_data'] if s['word_count'] >= 2]  # noqa: PLR2004
        lengths = [s['word_count'] for s in records]
        depths = [s['tree_depth'] for s in records]
        r = scipy_stats.pearsonr(lengths, depths).statistic
        ranked.append(
            (
                r,
                [
                    TREEBANK_NAMES[key],
                    f'{len(records):,}',
                    _bare(f'{r:.3f}'),
                    _bare(f'{r**2:.3f}'),
                    # mean length is not a proportion and keeps its leading digit
                    f'{sum(lengths) / len(lengths):.2f}',
                ],
            ),
        )
    # sorted on the correlation itself rather than on its rendering
    # because the leading zero came off
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    body = [row for _, row in ranked]

    return _latex_table(
        spec='LRRRR',
        header_rows=[
            [
                _header('Corpus'),
                _header('Sents.'),
                _header('r'),
                _header('\\small{$R^2$}'),
                _header('Mean len.'),
            ],
        ],
        body_rows=body,
        caption=(
            'Pearson correlation between sentence length in words and dependency tree depth. '
            'All correlations are significant at $p < 10^{-200}$. Sentences of fewer than two '
            'non-punctuation words are excluded, which is why the mean lengths differ slightly '
            'from those in Table~\\ref{tab:treebank-stats}.'
        ),
        label='tab:length-correlations',
        wide=False,
    )


def consistency() -> str:
    """Table 5: intra-annotator consistency on repeated and near-repeated sequences."""
    from bootstrapping import consistency as consistency_module  # noqa: PLC0415

    items = consistency_module.annotated_sentences()

    exact = consistency_module.score_repeats(consistency_module.find_repeats(items))
    body = [
        [
            'Exact repeats',
            f'{exact["sequences"]:,}',
            f'{exact["token_decisions"]:,}',
            *[f'{exact[field]["agreement"]:.2f}' for field in ('upos', 'head', 'deprel', 'las')],
        ],
    ]

    # max substitutions, minimum similarity, whether the row is a result or a boundary check
    relaxations = [(1, 0.7, True), (2, 0.7, True), (2, 0.5, False)]
    for edits, similarity, is_result in relaxations:
        scored = consistency_module.score_near_duplicate_clusters(
            consistency_module.find_near_duplicate_clusters(items, max_edits=edits, min_similarity=similarity),
        )
        plural = '' if edits == 1 else 's'
        row = [
            f'$\\le${edits} substitution{plural}, $\\ge${similarity:.0%} similar'.replace('%', '\\%'),
            f'{scored["clusters"]:,}',
            f'{scored["token_decisions"]:,}',
            *[f'{scored[field]:.2f}' for field in ('upos', 'head', 'deprel', 'las')],
        ]
        # the boundary-check row is separated from the results above it
        if not is_result:
            body.append(['\\midrule'])
        body.append(row)

    return _latex_table(
        spec='Xrrrrrr',
        header_rows=[
            [
                _header('Criterion'),
                _header('Groups'),
                _header('Decisions'),
                _header('UPoS', expand='~~~'),
                _header('Head', expand='~~~'),
                _header('Deprel'),
                _header('LAS', expand='~~~'),
            ],
        ],
        body_rows=body,
        caption=(
            'Intra-annotator consistency, as percentage agreement between analyzes assigned to '
            'the same or nearly the same word sequence in different annotation rounds. The last '
            'row is a boundary check rather than a result since at 50\\% similarity the groups are no '
            'longer duplicates, and head agreement collapses accordingly.'
        ),
        label='tab:consistency',
        wide=True,
        target='tabularx',
    )


def _cumulative_effort() -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Return cumulative sentences, tokens, and annotation hours per reported iteration."""
    results = _results()['data']
    sentences, tokens, hours = {}, {}, {}
    running_s = running_t = 0
    running_h = 0.0
    for key in REPORTED_SEED_KEYS:
        batch = load_file(SEED_TRAINING_PATHS[key])
        running_s += len(batch)
        running_t += sum(1 for sentence in batch for token in sentence if isinstance(token['id'], int))
        running_h += results.get(key, {}).get('annotation_time_minutes', 0) / 60
        sentences[key], tokens[key], hours[key] = running_s, running_t, running_h
    return sentences, tokens, hours


def _learning_curve(metrics: list[str], caption: str, label: str) -> str:
    results = _results()['data']
    sentences, _, hours = _cumulative_effort()

    body = []
    for entry in REPORTED_ITERATIONS:
        key = entry['key']
        evaluation = results[key]['evaluation']
        if key == 'ittb':
            name, count, effort = 'Baseline (ITTB)', EM_DASH, '0.0'
        else:
            name = f'\\normalsize{{{_iteration(key)}}}'
            count, effort = f'{sentences[key]:,}', f'{hours[key]:.1f}'
        body.append([name, count, effort, *[_bare(f'{evaluation[m]["f1"]:.3f}') for m in metrics]])

    return _latex_table(
        spec='XXX' + 'c' * len(metrics),
        header_rows=[
            [
                _header('Iteration'),
                _header('Sentences'),
                _header('Hours'),
                *[_header(HEADINGS.get(m, m), expand='~~~~') for m in metrics],
            ],
        ],
        body_rows=body,
        caption=caption,
        label=label,
        wide=True,
        target='tabularx',
    )


def morph_curve() -> str:
    """Table 6: morphological accuracy across the nine reported iterations."""
    return _learning_curve(
        ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'AllTags'],
        'Morphological performance across the nine reported iterations, F1 on the 200-sentence '
        'gold standard. Sentences are the pooled corrected seed data the model was '
        'trained on, hours are cumulative annotation time. The ITTB baseline is trained on '
        '26,977 sentences of out-of-domain text and required no annotation of this corpus.',
        'tab:morph-curve',
    )


def syn_curve() -> str:
    """Table 7: syntactic accuracy across the nine reported iterations."""
    return _learning_curve(
        SYNTAX_COLUMNS,
        'Syntactic performance across the nine reported iterations, F1 on the 200-sentence gold '
        'standard. Sentences and hours as in Table~\\ref{tab:morph-curve}.',
        'tab:syn-curve',
    )


def cross_treebank_performance() -> str:
    """Table 8: the final reported model against every treebank baseline, on the full metric set."""
    results = _results()['data']
    stats = _corpus_stats()
    sentences, tokens, _ = _cumulative_effort()
    columns = MORPHO_COLUMNS + SYNTAX_COLUMNS
    final = REPORTED_SEED_KEYS[-1]
    headings = ['UPoS', 'XPoS', 'Feats.', 'Lem.', 'Ens.', 'All', 'UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX']

    def scores(key: str) -> list[float]:
        evaluation = results[key]['evaluation']
        ensemble = results.get(f'{key}_ens', {}).get('evaluation', evaluation)
        return [
            (ensemble if column == 'Lemmas (ensemble)' else evaluation)[
                'Lemmas' if column == 'Lemmas (ensemble)' else column
            ]['f1']
            for column in columns
        ]

    baseline = {key: scores(key) for key in BASELINE_ORDER}
    body = [
        [
            TREEBANK_NAMES[key],
            f'{stats[key]["num_sentences"]:,}',
            f'{stats[key]["num_tokens"]:,}',
            *[_bare(f'{value:.3f}') for value in row],
        ]
        for key, row in baseline.items()
    ]

    best = [max(row[index] for row in baseline.values()) for index in range(len(columns))]
    body.append(['\\midrule Best of 5', EM_DASH, EM_DASH, *[_bare(f'{value:.3f}') for value in best]])

    plain = columns.index('Lemmas')
    final_row = [f'{_bare(f"{value:.3f}")}' for value in scores(final)]
    final_row[plain] = EM_DASH
    label = f'\\normalsize{{{_iteration(final)}}}'
    body.append(
        [
            f'\\rowcolor{{black!8}} \\textbf{{{label}}}',
            f'{sentences[final]:,}',
            f'{tokens[final]:,}',
            *final_row,
        ],
    )

    margin = [f'{_bare(f"{value - best[index]:+.3f}")}' for index, value in enumerate(scores(final))]
    margin[plain] = EM_DASH
    body.append(['\\textbf{$\\Delta$}', EM_DASH, EM_DASH, *margin])

    group = [
        '',
        f'\\multicolumn{{2}}{{c|}}{{{_header("Training")}}}',
        f'\\multicolumn{{{len(MORPHO_COLUMNS)}}}{{c|}}{{{_header("Morphological")}}}',
        f'\\multicolumn{{{len(SYNTAX_COLUMNS)}}}{{c}}{{{_header("Syntactic")}}}',
    ]
    names = [
        _header('Model'),
        _header('Sent.'),
        _header('Tokens'),
        *[_header(heading) for heading in headings],
    ]

    return _latex_table(
        spec='X|rr|' + 'c' * len(MORPHO_COLUMNS) + '|' + 'c' * len(SYNTAX_COLUMNS),
        header_rows=[group, names],
        setup='\\setlength{\\tabcolsep}{5pt}',
        prologue=[f'\\cmidrule{{2-{3 + len(columns)}}}'],
        body_rows=body,
        caption=(
            'The final reported model against every treebank baseline on the DALME-Marseille gold '
            'standard, over the same metrics as Table~\\ref{tab:pre-trained-performance}. '
            'Every column uses the plain models except \\emph{Ens.}, which uses the ensemble of the '
            'sequence-to-sequence lemmatizer with domain lexicon lookup for every row alike. '
            f'\\normalsize{label}\\scriptsize~is only ever run in that configuration, so it has no plain '
            'lemmatization figure to report. \\emph{Best of 5} is the highest of the five treebank '
            'models in each column, which is not the same model in every column, and $\\Delta$ is '
            f"the final model's margin over it. Training sentences and tokens for~\\normalsize{label}\\scriptsize~are "
            'the pooled corrected sentences, for the baselines, the size of the treebank the '
            'model was trained on is used.'
        ),
        label='tab:cross-treebank-performance',
        wide=True,
        target='tabularx',
    )


def construction_specific_improvements() -> str:
    """Table 9: baseline versus final model, split by sentence construction type."""
    from bootstrapping.analysis import compute_construction_accuracy  # noqa: PLC0415

    gold = load_file(GOLD_STANDARD_PATH)
    final = REPORTED_SEED_KEYS[-1]
    predictions = {key: load_file(EVALUATION_DATA_PATHS[key]) for key in ('ittb', final)}

    scores = {
        (key, metric): compute_construction_accuracy(gold, sentences, metric)
        for key, sentences in predictions.items()
        for metric in ('upos', 'las')
    }

    labels = {
        'simple_single_object': 'CSO',
        'code_switching': 'CS',
        'complex_single_object': 'CSO',
        'simple_multi_object': 'SMO',
        'complex_multi_object': 'CMO',
        'other': 'Other',
    }

    def row(construction: str) -> list[str]:
        count = scores[('ittb', 'upos')][construction]['sentences']
        cells = [labels.get(construction, construction), str(count)]
        for metric in ('upos', 'las'):
            for key in ('ittb', final):
                cells.append(_bare(f'{scores[(key, metric)][construction]["accuracy"]:.3f}'))
        return cells

    present = [c for c in labels if c in scores[('ittb', 'upos')]]
    common = [c for c in present if scores[('ittb', 'upos')][c]['sentences'] >= INTERPRETABLE_MIN_SENTENCES]
    rare = [c for c in present if c not in common]
    common.sort(key=lambda c: scores[('ittb', 'upos')][c]['sentences'], reverse=True)
    rare.sort(key=lambda c: scores[('ittb', 'upos')][c]['sentences'], reverse=True)

    body = [row(c) for c in common]
    if rare:
        body.append(['\\midrule \\multicolumn{6}{l}{\\scriptsize\\emph{Too few sentences to interpret:}}'])
        body += [row(c) for c in rare]

    final_label = f'\\normalsize{{{_iteration(final)}}}'
    return _latex_table(
        spec='Xrcccc',
        header_rows=[
            ['', '', f'\\multicolumn{{2}}{{c}}{{{_header("UPoS")}}}', f'\\multicolumn{{2}}{{c}}{{{_header("LAS")}}}'],
            [
                _header('Const.'),
                _header('Sents'),
                _header('ITTB'),
                _header(final_label),
                _header('ITTB'),
                _header(final_label),
            ],
        ],
        prologue=['\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}'],
        body_rows=body,
        caption=(
            'Accuracy by construction type on the gold standard, baseline versus final reported '
            'model. The last group is reported for completeness only. Legend: SSO = Simple, single object, '
            'CS = Code-switching, CSO = Complex, single object, SMO = Simple, multiple objects, '
            'CMO = Complex, multiple objects.'
        ),
        label='tab:construction-specific-improvements',
        wide=False,
        target='tabularx',
    )


def annotation_effort_efficiency() -> str:
    """Table 10: annotation cost per round, and how much of the pre-annotation needed correcting."""
    from bootstrapping.analysis import align_tokens  # noqa: PLC0415

    results = _results()['data']
    body: list[list[str]] = []
    totals = {'sentences': 0, 'tokens': 0, 'minutes': 0.0}
    for key in REPORTED_SEED_KEYS:
        corrected = load_file(SEED_TRAINING_PATHS[key])
        sentences = len(corrected)
        tokens = sum(1 for s in corrected for t in s if isinstance(t['id'], int))
        minutes = results.get(key, {}).get('annotation_time_minutes', 0)

        # how much of the model's pre-annotation I had to change
        predicted = {s.metadata['sent_id']: s for s in load_file(SEED_PREDICTED_PATHS[key])}
        changed = compared = 0
        for sentence in corrected:
            match = predicted.get(sentence.metadata.get('sent_id'))
            if match is None:
                continue
            pairs, _, _ = align_tokens(sentence, match)
            compared += len(pairs)
            changed += sum(
                1
                for gold_token, pred_token in pairs
                if (gold_token['head'], gold_token['deprel']) != (pred_token['head'], pred_token['deprel'])
            )

        totals['sentences'] += sentences
        totals['tokens'] += tokens
        totals['minutes'] += minutes
        body.append(
            [
                f'\\normalsize{{{_iteration(key)}}}',
                f'{sentences:,}',
                f'{tokens:,}',
                f'{minutes:.0f}',
                _half_up(minutes / sentences),
                f'{changed / compared * 100:.1f}\\%' if compared else EM_DASH,
            ],
        )

    body.append(
        [
            '\\midrule \\textbf{Total}',
            f'{totals["sentences"]:,}',
            f'{totals["tokens"]:,}',
            f'{totals["minutes"]:.0f}',
            f'{_half_up(totals["minutes"] / totals["sentences"])}',
            '---',
        ],
    )

    return _latex_table(
        spec='CRRRRR',
        header_rows=[
            [
                _header('Round'),
                _header('Sent.'),
                _header('Tokens'),
                _header('Mins.'),
                _header('Min/S'),
                _header('Corrected'),
            ],
        ],
        body_rows=body,
        caption=(
            'Annotation effort per round. \\emph{Corrected} is the share of pre-annotated '
            'tokens whose head or relation the annotator changed, which is the quantity the '
            'bootstrapping loop is meant to drive down.'
        ),
        label='tab:annotation-effort-efficiency',
        wide=False,
    )


EMITTERS = {
    'tab:treebank-stats': treebank_stats,
    'tab:pre-trained-performance': pre_trained_performance,
    'tab:modern-stanza': modern_stanza,
    'tab:length-correlations': length_correlations,
    'tab:consistency': consistency,
    'tab:morph-curve': morph_curve,
    'tab:syn-curve': syn_curve,
    'tab:cross-treebank-performance': cross_treebank_performance,
    'tab:construction-specific-improvements': construction_specific_improvements,
    'tab:annotation-effort-efficiency': annotation_effort_efficiency,
}


def write_all(directory: Path | str = BUILD_DIR) -> list[Path]:
    """Emit every table into 'directory', one file per label."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    written = []
    for label, emitter in EMITTERS.items():
        path = target / f'{label.removeprefix("tab:")}.tex'
        path.write_text(emitter(), encoding='utf-8')
        written.append(path)
    return written


def _stem(label: str) -> str:
    """Remove type prefix from label."""
    return label.removeprefix('tab:')


def install(destination: Path | str = INSTALL_DIR) -> list[Path]:
    """Copy the built tables to where the paper reads them."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for label in EMITTERS:
        source = BUILD_DIR / f'{_stem(label)}.tex'
        if not source.exists():
            msg = f'{source} does not exist. Run without --install first'
            raise FileNotFoundError(msg)
        path = target / source.name
        path.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')
        copied.append(path)
    return copied


def paper_inputs(path: Path | str = PAPER_PATH) -> set[str]:
    """Return the basenames the paper pulls in."""
    return set(TABLE_INPUT.findall(Path(path).read_text(encoding='utf-8')))


def paper_tables(path: Path | str = PAPER_PATH) -> dict[str, str]:
    """Return any literal 'table' block in the paper, keyed by its label."""
    text = Path(path).read_text(encoding='utf-8')
    blocks = {}
    for block in TABLE_BLOCK.finditer(text):
        label = LABEL.search(block.group(0))
        if label:
            blocks[label.group(1)] = block.group(0)
    return blocks


class Drift(NamedTuple):
    """The three ways the paper and this module can disagree about tables."""

    stale: list[str]
    """Installed under tex/tables/, but no longer what the emitter produces."""
    missing: list[str]
    """Emitted here, but the paper never inputs it."""
    ungenerated: list[str]
    """A table written into the paper by hand, with no emitter behind it."""

    def __bool__(self) -> bool:
        """Report whether the paper and this module disagree in any of the three ways."""
        return bool(self.stale or self.missing or self.ungenerated)


def drift(path: Path | str = PAPER_PATH, destination: Path | str = INSTALL_DIR) -> Drift:
    """Compare what the paper reads against what the emitters produce."""
    inputs = paper_inputs(path)
    target = Path(destination)
    stale, missing = [], []
    for label, emitter in EMITTERS.items():
        stem = _stem(label)
        if stem not in inputs:
            missing.append(label)
            continue
        installed = target / f'{stem}.tex'
        if not installed.exists() or installed.read_text(encoding='utf-8') != emitter():
            stale.append(label)
    return Drift(
        stale=stale,
        missing=missing,
        ungenerated=sorted(paper_tables(path)),
    )


def _report(found: Drift, path: Path) -> int:
    """Print what drifted and how to fix it."""
    if not found:
        print(f'OK: all {len(EMITTERS)} tables in {path.relative_to(PROJECT_ROOT)} are current')
        return 0
    if found.stale:
        print('stale in ' + str(INSTALL_DIR.relative_to(PROJECT_ROOT)) + ': ' + ', '.join(found.stale))
    if found.missing:
        print('emitted but never input by the paper: ' + ', '.join(found.missing))
    if found.ungenerated:
        print('written into the paper by hand, with no emitter: ' + ', '.join(found.ungenerated))
    print('run: python -m bootstrapping.report.tables --install')
    return 1


def main(argv: list[str] | None = None) -> int:
    """Regenerate every table, and optionally install it for the paper or verify what is there."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--install',
        nargs='?',
        const=str(INSTALL_DIR),
        metavar='DIR',
        help=f'after building, copy the tables into DIR (default {INSTALL_DIR})',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help="don't rebuild, only report whether the paper's tables are current",
    )
    args = parser.parse_args(argv)

    if args.check:
        return _report(drift(), PAPER_PATH)

    for path in write_all():
        print(f'wrote {path.relative_to(PROJECT_ROOT)}')

    if args.install:
        for path in install(Path(args.install)):
            print(f'installed {path.relative_to(PROJECT_ROOT)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
