r"""Generate the paper's figures from the repository's own data.

Regenerate everything into 'build/figures/':

    python -m bootstrapping.report.paper_figures
    python -m bootstrapping.report.paper_figures --install   # and copy into tex/figures/
    python -m bootstrapping.report.paper_figures --check     # verify the paper is current

Each figure is named after the '\label' it carries in the paper.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bootstrapping.config import PROJECT_ROOT
from bootstrapping.report.figures import apply_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

BUILD_DIR = Path(PROJECT_ROOT) / 'build' / 'figures'
INSTALL_DIR = Path(PROJECT_ROOT) / 'tex' / 'figures'
INSTALLED = frozenset(
    {
        'learning-curve',
        'bootstrapping-loop',
        'cost-benefit',
        'adposition-dip',
        'latin-in-context',
        'pos-distribution',
    },
)
STANZA_BENCHMARK_PATH = Path(PROJECT_ROOT) / 'data' / 'reference' / 'stanza-ud-2.5-performance.csv'

# corpora in the chronological order the paper uses
CORPUS_ORDER = ['perseus', 'proiel', 'llct', 'ittb', 'udante', 'marseille']
CORPUS_LABELS = {
    'perseus': 'Perseus',
    'proiel': 'PROIEL',
    'llct': 'LLCT',
    'ittb': 'ITTB',
    'udante': 'UDante',
    'marseille': 'DALME-Marseille',
}

MORPHO_METRICS = [
    ('UPOS', 'UPoS'),
    ('XPOS', 'XPoS'),
    ('UFeats', 'Features'),
    ('Lemmas', 'Lemmata'),
    ('AllTags', 'All Tags'),
]
SYNTAX_METRICS = [('UAS', 'UAS'), ('LAS', 'LAS'), ('CLAS', 'CLAS'), ('MLAS', 'MLAS'), ('BLEX', 'BLEX')]
DIP_CATEGORIES = ['ADP', 'VERB', 'DET', 'ADJ', 'NOUN', 'NUM']
DPI = 300
TEXT_WIDTH_IN = 6.48
COL_WIDTH_IN = 3.04


def _save(figure: Figure, name: str, pad_inches: float = 0.02) -> Path:
    """Write one figure into 'build/figures/' and return the path."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_DIR / f'{name}.png'
    figure.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=pad_inches)
    return path


def _fit_x(figure: Figure, ax: Any, pad: float = 0.1, passes: int = 2) -> None:
    """Narrow the x limits to what the axes actually draws, leaving 'pad' data units either side."""
    for _ in range(passes):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
        inverse = ax.transData.inverted()
        boxes = [artist.get_window_extent(renderer) for artist in (*ax.patches, *ax.texts, *ax.lines)]
        left = inverse.transform((min(box.x0 for box in boxes), 0))[0]
        right = inverse.transform((max(box.x1 for box in boxes), 0))[0]
        ax.set_xlim(left - pad, right + pad)


def _iteration_axis(ax: Any, labels: list[str]) -> None:
    """Label the x axis with the baseline followed by the nine reported iterations."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Model', fontweight='bold', color='#636363')


def _stanza_benchmarks() -> list[dict[str, Any]]:
    """Read Stanza's published UD 2.5 scores, one row per released model."""
    with STANZA_BENCHMARK_PATH.open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def latin_in_context() -> Path:
    """Figure 1: Latin models on their own test sets against the same models on this corpus."""
    import json  # noqa: PLC0415

    import matplotlib.pyplot as plt  # noqa: PLC0415

    from bootstrapping.config import EVALUATION_RESULTS_PATH  # noqa: PLC0415

    apply_style(profile='print', figsize=(TEXT_WIDTH_IN, 4.0))
    rows = _stanza_benchmarks()

    def score(row: dict[str, Any], column: str) -> float | None:
        """Parse one published score."""
        try:
            return float(row[column])
        except (TypeError, ValueError):
            return None

    def required(row: dict[str, Any], column: str) -> float:
        """As 'score', for the three Latin rows, where a missing value is a broken input file."""
        parsed = score(row, column)
        if parsed is None:
            msg = f'{row["UD 2.5 test"]} has no {column} score in {STANZA_BENCHMARK_PATH.name}'
            raise ValueError(msg)
        return parsed

    published = [(score(r, 'LAS'), score(r, 'MLAS')) for r in rows]
    cloud = [(x, y) for x, y in published if x is not None and y is not None]

    results = json.loads(Path(EVALUATION_RESULTS_PATH).read_text())['data']
    pairs = {'UD_Latin-ITTB': 'ittb', 'UD_Latin-PROIEL': 'proiel', 'UD_Latin-Perseus': 'perseus'}
    label_coords = {
        'UD_Latin-ITTB': (-8, 6),
        'UD_Latin-PROIEL': (-33, 4),
        'UD_Latin-Perseus': (-36, 3),
    }

    figure, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.0))
    ax.scatter(
        [x for x, _ in cloud],
        [y for _, y in cloud],
        s=22,
        color='#5882d6',
        alpha=0.35,
        edgecolors='none',
        label=f'Stanza models (UD set, n={len(cloud)})',
        zorder=2,
    )

    for treebank, key in pairs.items():
        row = next(r for r in rows if r['UD 2.5 test'] == treebank)
        own = (required(row, 'LAS'), required(row, 'MLAS'))
        here = (results[key]['evaluation']['LAS']['f1'] * 100, results[key]['evaluation']['MLAS']['f1'] * 100)
        ax.scatter(*own, s=30, color='#51ae29', edgecolors='#274e13', linewidth=0.8, zorder=6)
        ax.scatter(*here, s=30, color='#e69138', marker='s', edgecolors='#783f04', linewidth=0.8, zorder=6)
        ax.annotate(
            treebank.replace('UD_Latin-', ''),
            xy=own,
            xytext=label_coords[treebank],
            textcoords='offset points',
            fontsize=7,
            fontweight='bold',
            color='#2f4b7c',
            zorder=9,
            bbox={'boxstyle': 'round,pad=-0.2', 'facecolor': '#eeeeee'},
        )
        ax.annotate(
            '',
            xy=(here[0] + 0.5, here[1] + 0.5),
            xytext=(own[0] - 0.5, own[1] - 0.5),
            arrowprops={'arrowstyle': '->', 'color': '#c44e52', 'linewidth': 1.4, 'alpha': 0.85},
            zorder=6,
        )

    ax.scatter([], [], s=30, color='#51ae29', edgecolors='#274e13', label='Latin model (UD set)')
    ax.scatter([], [], s=30, color='#e69138', marker='s', edgecolors='#783f04', label='Latin model (DALME-Marseille)')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xlabel('LAS', fontweight='bold', color='#636363')
    ax.set_ylabel('MLAS', fontweight='bold', color='#636363')
    ax.legend(loc='upper left', borderpad=0.7, handlelength=1.2, columnspacing=1.0, labelspacing=0.8)
    figure.tight_layout()
    return _save(figure, 'latin-in-context')


def pos_distribution() -> Path:
    """Figure 2: part-of-speech group shares per corpus, as horizontal stacked bars."""
    import pandas as pd  # noqa: PLC0415
    from matplotlib.ticker import PercentFormatter  # noqa: PLC0415

    from bootstrapping.config import CORPUS_STATISTICS_PATH, POS_TAG_GROUPS, TREEBANK_NAMES  # noqa: PLC0415
    from bootstrapping.io import load_file  # noqa: PLC0415

    excluded = {'PUNCT', 'X', 'SYM', '_'}
    order = ['marseille', 'ittb', 'llct', 'perseus', 'proiel', 'udante']

    stats = load_file(CORPUS_STATISTICS_PATH)
    rows = []
    for corpus in order:
        counts = stats[corpus]['upos_counts']
        denominator = sum(c for tag, c in counts.items() if tag not in excluded)
        rows.append(
            {
                group: (sum(counts.get(tag, 0) for tag in tags) / denominator * 100 if denominator else 0.0)
                for group, tags in POS_TAG_GROUPS.items()
            },
        )
    frame = pd.DataFrame(rows, index=[TREEBANK_NAMES[c] for c in order]) / 100.0

    apply_style(profile='print', figsize=(TEXT_WIDTH_IN, 3))
    axes = frame.plot(
        kind='barh',
        stacked=True,
        figsize=(TEXT_WIDTH_IN, 3),
        colormap='tab20c',
        edgecolor='#999999',
        width=0.75,
        zorder=3,
    )
    axes.set_yticks(range(len(order)))
    axes.set_yticklabels(frame.index, fontweight='bold')
    axes.invert_yaxis()
    axes.tick_params(axis='x', labelsize=6)
    axes.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes.set_xlim(0, 1)
    axes.set_xlabel('Percentage of total words', fontweight='bold', color='#636363')
    axes.set_ylabel('')
    axes.legend(title='', bbox_to_anchor=(0.5, 1.18), loc='upper center', ncol=4)

    # share labels inside each segment
    # skip slices too narrow to hold one
    minimum_labelled = 0.04
    for row_index, (_, row) in enumerate(frame.iterrows()):
        left = 0.0
        for value in row:
            if value > minimum_labelled:
                axes.text(
                    left + value / 2,
                    row_index,
                    f'{value * 100:.1f}%',
                    va='center',
                    ha='center',
                    fontsize=5,
                    color='#333333',
                    fontweight='bold',
                )
            left += value

    figure = axes.get_figure()
    figure.tight_layout(pad=0.5)
    return _save(figure, 'pos-distribution')


def bootstrapping_loop() -> Path:
    """Figure 3: schematic of one iteration of the procedure."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: PLC0415

    apply_style(profile='print', grid=False, figsize=(COL_WIDTH_IN, 3.5))
    figure, ax = plt.subplots(figsize=(COL_WIDTH_IN, 3.5))
    ax.set_facecolor('white')
    ax.set_xlim(-0.4, 10.0)
    ax.set_ylim(0.2, 10.1)
    ax.axis('off')

    steps = [
        ('1. Draw a batch', '200 sentences, random\nwithin length strata', 9.2, '#4c72b0'),
        ('2. Pre-annotate', 'parse with the model from\niteration $n-1$ (ITTB for $s_1$)', 7.2, '#4c72b0'),
        ('3. Correct by hand', 'morphology, then syntax,\nin brat', 5.2, '#c44e52'),
        ('4. Pool and retrain', 'batches $s_1 \\ldots s_n$ together,\nfrom scratch', 3.2, '#4c72b0'),
        ('5. Evaluate', 'against the fixed\n200-sentence gold standard', 1.2, '#55a868'),
    ]
    for title, detail, y, colour in steps:
        ax.add_patch(
            FancyBboxPatch(
                (2.0, y - 0.62),
                5.0,
                1.24,
                boxstyle='round,pad=0.12',
                facecolor=colour,
                alpha=0.13,
                edgecolor=colour,
                linewidth=1.4,
            ),
        )
        ax.text(4.5, y + 0.26, title, ha='center', va='center', fontsize=9, fontweight='bold', color=colour)
        ax.text(4.5, y - 0.28, detail, ha='center', va='center', fontsize=7, color='#333333')

    for y in (9.2, 7.2, 5.2, 3.2):
        ax.add_patch(
            FancyArrowPatch(
                (4.5, y - 0.76),
                (4.5, y - 1.22),
                arrowstyle='-|>',
                mutation_scale=15,
                color='#c2c2c2',
                linewidth=1.2,
            ),
        )

    # both return paths are routed clear of the boxes
    def _route(x_spur: float, y_from: float, y_to: float, x_edge: float, **style: Any) -> None:
        ax.plot([x_edge, x_spur, x_spur], [y_from, y_from, y_to], solid_capstyle='round', **style)
        ax.add_patch(
            FancyArrowPatch(
                (x_spur, y_to),
                (x_edge, y_to),
                arrowstyle='-|>',
                mutation_scale=14,
                color=style.get('color', '#555555'),
                linewidth=style.get('linewidth', 1.2),
                linestyle=style.get('linestyle', '-'),
            ),
        )

    # training feeds the next round's sampling
    _route(8.35, 3.2, 9.2, 7.2, color='#4c72b0', linewidth=1.6)
    ax.text(
        8.8,
        6.2,
        'next iteration',
        ha='center',
        va='center',
        fontsize=9,
        color='#4c72b0',
        rotation=90,
        fontweight='bold',
    )

    # evaluation doesn't
    _route(1.05, 1.2, 9.2, 1.78, color='#b0b0b0', linewidth=1.2, linestyle='--')
    ax.text(
        1.05,
        5.2,
        'X',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='#c44e52',
        bbox={'boxstyle': 'circle,pad=0.18', 'facecolor': 'white', 'edgecolor': '#c44e52', 'linewidth': 1.2},
    )
    ax.text(
        0.35,
        5.2,
        'evaluation never steers selection',
        ha='center',
        va='center',
        fontsize=9,
        color='#777777',
        style='italic',
        fontweight='bold',
        rotation=90,
    )

    figure.tight_layout()
    _fit_x(figure, ax)
    return _save(figure, 'bootstrapping-loop')


def learning_curve() -> Path:
    """Figure 4: the nine-iteration trajectory, morphological above syntactic."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from bootstrapping import learning  # noqa: PLC0415

    size = (COL_WIDTH_IN, 4.6)
    apply_style(profile='print', figsize=size)
    frame = learning.evaluation_frame(learning.iteration_keys(with_baseline=True))
    labels = ['ITTB', *[f's{i}' for i in range(1, 10)]]
    x = list(range(len(frame)))

    figure, axes = plt.subplots(2, 1, figsize=size, sharex=True)
    for ax, metrics, title in (
        (axes[0], MORPHO_METRICS, 'Morphological'),
        (axes[1], SYNTAX_METRICS, 'Syntactic'),
    ):
        for key, label in metrics:
            ax.plot(x, frame[f'{key}_f1'], marker='o', label=label, markersize=2, linewidth=0.8, alpha=0.85)
        # the baseline is the first point
        ax.axvline(0.5, color='#990000', linewidth=0.8, linestyle=':')
        ax.set_title(title, color='#2f4b7c', fontweight='bold', fontsize=7)
        ax.set_ylabel('F1', fontweight='bold', color='#636363')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(
            loc='lower right',
            ncol=2,
            fontsize=5.5,
            handlelength=1.2,
            columnspacing=0.8,
            borderpad=0.4,
            labelspacing=0.4,
            handletextpad=0.4,
        )

    # only the lower panel carries the iteration labels
    axes[1].tick_params(axis='x', labelsize=6)
    _iteration_axis(axes[1], labels)

    # shade the rounds that score below the out-of-domain baseline on UPoS
    axes[0].axvspan(0.5, 3.5, color='#c44e52', alpha=0.07, zorder=0)
    axes[0].annotate(
        'below baseline',
        xy=(2, 0.06),
        ha='center',
        fontsize=5.5,
        color='#990000',
    )
    figure.tight_layout(pad=0.4)
    return _save(figure, 'learning-curve')


def adposition_dip() -> Path:
    """Figure 5: per-category tagging accuracy, showing the adposition collapse and recovery."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from bootstrapping import learning  # noqa: PLC0415

    apply_style(profile='print', figsize=(TEXT_WIDTH_IN, 3.4))
    keys = learning.iteration_keys(with_baseline=True)
    comparisons = learning.token_comparisons(keys)

    figure, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.4))
    labels = ['ITTB', *[f's{i}' for i in range(1, 10)]]
    for category in DIP_CATEGORIES:
        series = []
        for key in keys:
            frame = comparisons[key]
            rows = frame[frame['gold_upos'] == category]
            series.append(rows['upos_correct'].mean() * 100 if len(rows) else float('nan'))
        emphasis = category in {'ADP', 'VERB'}
        ax.plot(
            range(len(keys)),
            series,
            marker='o',
            markersize=2 if emphasis else 1.5,
            linewidth=1 if emphasis else 0.8,
            alpha=1.0 if emphasis else 0.35,
            label=category,
            zorder=3 if emphasis else 2,
        )

    ax.set_ylabel('UPoS accuracy (%)', fontweight='bold', color='#636363')
    ax.set_ylim(0, 103)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(loc='lower right', ncol=3, handlelength=1.4, columnspacing=1.0, borderpad=0.7, labelspacing=0.8)
    _iteration_axis(ax, labels)
    figure.tight_layout()
    return _save(figure, 'adposition-dip')


def cost_benefit() -> Path:
    """Figure 6: accuracy against cumulative annotation hours."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from bootstrapping import learning  # noqa: PLC0415

    apply_style(profile='print', figsize=(TEXT_WIDTH_IN, 3.2))
    frame = learning.annotation_hours(learning.evaluation_frame())

    figure, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2))
    for key, label in (('LAS', 'LAS'), ('MLAS', 'MLAS'), ('AllTags', 'All Tags')):
        ax.plot(
            frame['cumulative_time_hours'],
            frame[f'{key}_f1'],
            marker='o',
            label=label,
        )

    for _, row in frame.iterrows():
        label_value = f's{int(row["iteration"])}'
        ax.annotate(
            label_value,
            xy=(row['cumulative_time_hours'], row['LAS_f1']),
            xytext=(0, 10) if label_value in ['s6', 's7'] else (0, 6),
            textcoords='offset points',
            ha='center',
            fontsize=6.5,
            color='#2f4b7c',
            fontweight='bold',
        )

    ax.set_xlabel('Cumulative annotation time (hours)', fontweight='bold', color='#636363')
    ax.set_ylabel('F1', fontweight='bold', color='#636363')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(loc='lower right', borderpad=0.7, labelspacing=0.8)
    figure.tight_layout()
    return _save(figure, 'cost-benefit')


FIGURES = (
    latin_in_context,
    pos_distribution,
    bootstrapping_loop,
    learning_curve,
    adposition_dip,
    cost_benefit,
)


def write_all() -> list[Path]:
    """Generate every paper figure and return the paths written."""
    return [figure() for figure in FIGURES]


def install(destination: Path = INSTALL_DIR) -> list[Path]:
    """Copy the figures the paper includes into 'destination', returning what was copied."""
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in sorted(INSTALLED):
        source = BUILD_DIR / f'{name}.png'
        if not source.exists():
            msg = f'{source} does not exist. Run without --install first'
            raise FileNotFoundError(msg)
        target = destination / f'{name}.png'
        shutil.copyfile(source, target)
        copied.append(target)
    return copied


def stale(destination: Path = INSTALL_DIR) -> list[str]:
    """Return the names of installed figures that differ from the built ones, or are absent."""
    out = []
    for name in sorted(INSTALLED):
        source, target = BUILD_DIR / f'{name}.png', destination / f'{name}.png'
        if not target.exists() or not source.exists() or source.read_bytes() != target.read_bytes():
            out.append(name)
    return out


def main(argv: list[str] | None = None) -> int:
    """Regenerate every figure, and optionally install or verify the paper's copies."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--install',
        nargs='?',
        const=str(INSTALL_DIR),
        metavar='DIR',
        help=f'after building, copy the paper figures into DIR (default {INSTALL_DIR})',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help="don't rebuild, just report whether the installed copies match the built ones",
    )
    args = parser.parse_args(argv)

    if args.check:
        outdated = stale()
        if outdated:
            print('stale in tex/figures/: ' + ', '.join(outdated))
            print('run: python -m bootstrapping.report.paper_figures --install')
            return 1
        print(f'OK: all {len(INSTALLED)} paper figures match build/figures/')
        return 0

    for path in write_all():
        print(f'wrote {path.relative_to(Path(PROJECT_ROOT))}')

    if args.install:
        for path in install(Path(args.install)):
            print(f'installed {path.relative_to(Path(PROJECT_ROOT))}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
