"""Fetch the treebanks from Universal Dependencies."""

from __future__ import annotations

from pathlib import Path

import requests

from .config import PROJECT_ROOT

UD_ORG = 'https://github.com/UniversalDependencies'
UD_BRANCH = 'master'
SPLITS = ('dev', 'test', 'train')
HTTP_OK = 200

# UD treebank name -> language code used in this project's filenames
TREEBANKS: dict[str, str] = {
    'UD_Latin-ITTB': 'la_ittb',
    'UD_Latin-LLCT': 'la_llct',
    'UD_Latin-UDante': 'la_udante',
    'UD_Latin-CIRCSE': 'la_circse',
    'UD_Latin-Perseus': 'la_perseus',
    'UD_Latin-PROIEL': 'la_proiel',
    'UD_Occitan-TTB': 'oc_ttb',
}


def corpora_dir() -> Path:
    """Return the directory holding the assembled treebanks."""
    return PROJECT_ROOT / 'data' / 'corpora'


def corpus_path(code: str) -> Path:
    """Return the path to an assembled '-full.conllu' file."""
    return corpora_dir() / f'{code}-ud-full.conllu'


def fetch_treebank(treebank: str, code: str, *, timeout: int = 60) -> str:
    """Download and concatenate the splits of one UD treebank.

    Arguments:
        treebank: UD repository name, e.g. 'UD_Latin-ITTB'.
        code: filename prefix used inside that repository, e.g. 'la_ittb'.
        timeout: per-request timeout in seconds.

    Returns:
        The concatenated CoNLL-U text.

    Raises:
        RuntimeError: if no split could be downloaded.

    """
    parts = []
    for split in SPLITS:
        url = f'{UD_ORG}/{treebank}/raw/refs/heads/{UD_BRANCH}/{code}-ud-{split}.conllu'
        response = requests.get(url, timeout=timeout)
        if response.status_code == HTTP_OK:
            parts.append(response.text)

    if not parts:
        msg = f'no splits downloaded for {treebank}. Check the repository name and branch'
        raise RuntimeError(msg)

    return ''.join(parts)


def ensure_corpora(
    treebanks: dict[str, str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download any missing comparison treebanks and return their paths.

    Arguments:
        treebanks: mapping of UD repository name to filename prefix (defaults to 'TREEBANKS').
        force: re-download even if the assembled file already exists.

    """
    treebanks = treebanks or TREEBANKS
    corpora_dir().mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for treebank, code in treebanks.items():
        path = corpus_path(code)
        if path.exists() and not force:
            paths[code] = path
            continue

        print(f'Downloading {treebank}...')
        path.write_text(fetch_treebank(treebank, code), encoding='utf-8')
        paths[code] = path

    return paths


def missing_corpora(treebanks: dict[str, str] | None = None) -> list[str]:
    """Return the codes of treebanks that are not present locally."""
    treebanks = treebanks or TREEBANKS
    return [code for code in treebanks.values() if not corpus_path(code).exists()]


def main() -> None:
    """Fetch any comparison treebank that is not already present."""
    missing = missing_corpora()
    if not missing:
        print(f'All {len(TREEBANKS)} treebanks already present in {corpora_dir()}')
        return

    print(f'{len(missing)} missing: {", ".join(missing)}')
    for code, path in ensure_corpora().items():
        print(f'  {code:12} {path.name}')
    print(f'\nReady in {corpora_dir()}')


if __name__ == '__main__':
    main()
