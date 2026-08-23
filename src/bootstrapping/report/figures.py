"""Shared matplotlib/seaborn styling."""

from __future__ import annotations

from typing import Any

DEFAULT_FIGSIZE = (12, 6)
DEFAULT_FONT_SIZE = 11
PRINT_WIDTH_IN = 6.45  # \textwidth: letterpaper, 73pt margins
PRINT_FONT_SIZE = 8
PRINT_FIGSIZE = (PRINT_WIDTH_IN, 3.2)
PRINT_LINE_WIDTH = 1.2
PRINT_MARKER_SIZE = 3.0
PRINT_SECONDARY_SIZE = 'small'
PROFILES = ('screen', 'print')
DEFAULT_DPI = 100
DEFAULT_SAVEFIG_DPI = 150
DEFAULT_STYLE = 'bmh'
DEFAULT_PALETTE = 'husl'
DEFAULT_TICK_DIRECTION = 'out'
DEFAULT_TITLE_SIZE = 'large'
DEFAULT_TITLE_PAD = 8.0
DEFAULT_LABEL_SIZE = 'medium'
DEFAULT_AXIS_MARGIN = 0.1
DEFAULT_GRID_COLOR = '#636363'
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_LEGEND_FACE_COLOR = 'white'
DEFAULT_LABEL_PAD = 6.0
DEFAULT_EDGE_COLOR = '#999999'
DEFAULT_FACE_COLOR = '#eeeeee'
DEFAULT_LABEL_COLOR = 'black'
DEFAULT_LABEL_SPACING = 0.5
DEFAULT_CL_PAD = 0.04167
DEFAULT_CL_SPACE = 0.02
DEFAULT_FONT = 'DejaVu Sans'


def _apply_print_overrides() -> None:
    """Shrink the settings that do not follow 'font.size' on their own."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams['lines.linewidth'] = PRINT_LINE_WIDTH
    plt.rcParams['lines.markersize'] = PRINT_MARKER_SIZE
    plt.rcParams['legend.fontsize'] = PRINT_SECONDARY_SIZE
    plt.rcParams['xtick.labelsize'] = PRINT_SECONDARY_SIZE
    plt.rcParams['ytick.labelsize'] = PRINT_SECONDARY_SIZE
    plt.rcParams['axes.labelsize'] = PRINT_SECONDARY_SIZE
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['figure.titlesize'] = 'large'


def apply_style(  # noqa: PLR0913
    palette: str = DEFAULT_PALETTE,
    figsize: tuple[float, float] | None = None,
    font_size: int | None = None,
    style: str = DEFAULT_STYLE,
    *,
    grid: bool = True,
    profile: str = 'screen',
) -> None:
    """Apply the project's figure style to the current matplotlib session.

    Arguments:
        palette: seaborn palette name, e.g. 'husl', 'colorblind', 'deep'.
        figsize: figure size in inches, defaults to the profile's.
        font_size: base font size in points, defaults to the profile's.
        style: matplotlib style sheet supplying the base look, e.g. 'bmh'.
        grid: whether axes carry a grid.
        profile: 'screen' for notebooks (the default), or 'print' for figures authored for the paper.

    Raises:
        ValueError: if 'profile' is not one of 'PROFILES'.

    """
    import matplotlib as mpl  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import seaborn as sns  # noqa: PLC0415

    if profile not in PROFILES:
        msg = f'profile must be one of {PROFILES}, not {profile!r}'
        raise ValueError(msg)

    printing = profile == 'print'
    if figsize is None:
        figsize = PRINT_FIGSIZE if printing else DEFAULT_FIGSIZE
    if font_size is None:
        font_size = PRINT_FONT_SIZE if printing else DEFAULT_FONT_SIZE

    # reset first so repeated calls are idempotent and a previous style cannot leak through
    # then the style sheet, then the palette, then the explicit overrides below
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(style)
    sns.set_palette(palette)

    plt.rcParams['axes.grid'] = grid
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = font_size
    plt.rcParams['figure.dpi'] = DEFAULT_DPI
    plt.rcParams['savefig.dpi'] = DEFAULT_SAVEFIG_DPI
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [DEFAULT_FONT]
    plt.rcParams['xtick.direction'] = DEFAULT_TICK_DIRECTION
    plt.rcParams['ytick.direction'] = DEFAULT_TICK_DIRECTION
    plt.rcParams['axes.titlesize'] = DEFAULT_TITLE_SIZE
    plt.rcParams['axes.titlepad'] = DEFAULT_TITLE_PAD
    plt.rcParams['axes.labelsize'] = DEFAULT_LABEL_SIZE
    plt.rcParams['axes.xmargin'] = DEFAULT_AXIS_MARGIN
    plt.rcParams['axes.ymargin'] = DEFAULT_AXIS_MARGIN
    plt.rcParams['grid.color'] = DEFAULT_GRID_COLOR
    plt.rcParams['grid.alpha'] = DEFAULT_GRID_ALPHA
    plt.rcParams['legend.facecolor'] = DEFAULT_LEGEND_FACE_COLOR
    plt.rcParams['axes.labelpad'] = DEFAULT_LABEL_PAD
    plt.rcParams['axes.edgecolor'] = DEFAULT_EDGE_COLOR
    plt.rcParams['axes.facecolor'] = DEFAULT_FACE_COLOR
    plt.rcParams['axes.labelcolor'] = DEFAULT_LABEL_COLOR
    plt.rcParams['xtick.color'] = DEFAULT_GRID_COLOR
    plt.rcParams['xtick.labelcolor'] = 'inherit'
    plt.rcParams['ytick.color'] = DEFAULT_GRID_COLOR
    plt.rcParams['ytick.labelcolor'] = 'inherit'
    plt.rcParams['legend.edgecolor'] = DEFAULT_EDGE_COLOR
    plt.rcParams['legend.labelcolor'] = None
    plt.rcParams['legend.labelspacing'] = DEFAULT_LABEL_SPACING
    plt.rcParams['figure.constrained_layout.h_pad'] = DEFAULT_CL_PAD
    plt.rcParams['figure.constrained_layout.hspace'] = DEFAULT_CL_SPACE
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.constrained_layout.w_pad'] = DEFAULT_CL_PAD
    plt.rcParams['figure.constrained_layout.wspace'] = DEFAULT_CL_SPACE

    if printing:
        _apply_print_overrides()


def save(figure: Any, filename: str, **kwargs: Any) -> str:
    """Save a figure into 'CHARTS_DIR' and return the path written."""
    from bootstrapping.config import chart_path  # noqa: PLC0415

    path = chart_path(filename)
    figure.savefig(path, bbox_inches='tight', **{'dpi': DEFAULT_SAVEFIG_DPI, **kwargs})
    return path
