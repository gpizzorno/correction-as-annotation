"""Fetch the trained models from Zenodo into the local Stanza resources directory.

    python scripts/fetch_models.py                  # all nine iterations, ~900 MB
    python scripts/fetch_models.py s9               # just the final model, ~110 MB
    python scripts/fetch_models.py s8 s9 --force    # re-download two that are already present
    python scripts/fetch_models.py --list           # what the record holds, download nothing

A Stanza registry has to exist before the packages can be merged into it. If you have never run
Stanza for Latin, create one first with:

    python -c "import stanza; stanza.download('la')"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import requests

from bootstrapping.config import STANZA_RESOURCES_DIR

# the concept DOI, which always resolves to the latest version
# override with --doi to pin a specific version
ZENODO_DOI = '10.5281/zenodo.22048602'
ZENODO_API = 'https://zenodo.org/api/records'
HTTP_OK = 200
CHUNK = 1 << 20

ITERATIONS = [f'marseille_s{i}' for i in range(1, 10)]
PRETRAIN_ARCHIVE = 'pretrain.tar.gz'
REGISTRY_FRAGMENT = 'resources-marseille.json'
CHECKSUMS = 'SHA256SUMS'

RESOURCES_PATH = STANZA_RESOURCES_DIR / 'resources.json'


def record_id(doi: str) -> str:
    """Return the Zenodo record id from a DOI, a record URL, or a bare id.

    Arguments:
        doi: '10.5281/zenodo.1234567', a zenodo.org URL, or '1234567'.

    Returns:
        The numeric record id.

    Raises:
        SystemExit: if no id can be read out of it.

    """
    match = re.search(r'(\d+)\s*$', doi.strip())
    if not match:
        msg = f'could not read a Zenodo record id out of {doi!r}'
        raise SystemExit(msg)
    return match.group(1)


def fetch_manifest(doi: str, *, timeout: int = 30) -> dict[str, dict[str, Any]]:
    """Return the deposit's files, keyed by filename.

    Arguments:
        doi: DOI, record URL, or record id.
        timeout: request timeout in seconds.

    Returns:
        Filename to '{"url": ..., "size": ...}'.

    Raises:
        SystemExit: if the record cannot be read.

    """
    if 'XXXXXXX' in doi:
        msg = (
            'no Zenodo DOI is set yet. Pass --doi once the deposit is published, or fill in\n'
            f'    ZENODO_DOI in {Path(__file__).name}'
        )
        raise SystemExit(msg)

    url = f'{ZENODO_API}/{record_id(doi)}'
    response = requests.get(url, timeout=timeout)
    if response.status_code != HTTP_OK:
        msg = f'{url} returned {response.status_code}. Check the DOI and that the record is public'
        raise SystemExit(msg)

    files = {}
    for entry in response.json().get('files', []):
        links = entry.get('links', {})
        files[entry['key']] = {
            'url': links.get('self') or links.get('download'),
            'size': entry.get('size', 0),
        }
    if not files:
        msg = f'the record at {url} lists no files'
        raise SystemExit(msg)
    return files


def download(url: str, destination: Path, *, size: int = 0, timeout: int = 60) -> Path:
    """Stream one file to disk, reporting progress for anything large.

    Arguments:
        url: direct download link.
        destination: where to write.
        size: expected size in bytes, for the progress line.
        timeout: connection timeout in seconds.

    Returns:
        The path written.

    Raises:
        SystemExit: on a non-200 response.

    """
    response = requests.get(url, stream=True, timeout=timeout)
    if response.status_code != HTTP_OK:
        msg = f'{url} returned {response.status_code}'
        raise SystemExit(msg)

    written = 0
    with destination.open('wb') as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            handle.write(chunk)
            written += len(chunk)
            if size > CHUNK * 8:
                print(f'\r  {destination.name:<28} {written / 1e6:6.1f} / {size / 1e6:.1f} MB', end='', flush=True)
    if size > CHUNK * 8:
        print()
    return destination


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(CHUNK):
            hasher.update(block)
    return hasher.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a 'SHA256SUMS' file into '{filename: digest}'."""
    sums = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        digest, _, name = line.partition('  ')
        if digest and name:
            sums[name.strip()] = digest.strip()
    return sums


def unpack(archive: Path, destination: Path) -> list[str]:
    """Extract one deposit archive into the resources directory.

    Arguments:
        archive: the '.tar.gz' to extract.
        destination: the Stanza resources directory.

    Returns:
        The member names extracted.

    Raises:
        SystemExit: if any member escapes 'la/'.

    """
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != 'la':
                msg = f'{archive.name} contains an unexpected member: {member.name}'
                raise SystemExit(msg)
        tar.extractall(destination)
    return [m.name for m in members if m.isfile()]


def merge_registry(fragment_path: Path, resources_path: Path = RESOURCES_PATH) -> int:
    """Merge the deposit's package entries into Stanza's registry.

    Arguments:
        fragment_path: the deposit's 'resources-marseille.json'.
        resources_path: Stanza's 'resources.json'.

    Returns:
        The number of package entries merged.

    Raises:
        SystemExit: if no registry exists to merge into.

    """
    if not resources_path.exists():
        msg = (
            f'no Stanza registry at {resources_path}.\n'
            'Create one first with:  python -c "import stanza; stanza.download(\'la\')"'
        )
        raise SystemExit(msg)

    resources = json.loads(resources_path.read_text(encoding='utf-8'))
    fragment = json.loads(fragment_path.read_text(encoding='utf-8'))

    merged = 0
    for language, tasks in fragment.items():
        section = resources.setdefault(language, {})
        for task, packages in tasks.items():
            section.setdefault(task, {}).update(packages)
            merged += len(packages)

    resources_path.write_text(json.dumps(resources), encoding='utf-8')
    return merged


def wanted_archives(selection: list[str]) -> list[str]:
    """Return the archive names to fetch for a selection of iterations.

    Arguments:
        selection: iteration names like 's9' or 'marseille_s9'. Empty means all nine.

    Returns:
        Archive filenames, always including the word vectors.

    Raises:
        SystemExit: on an unknown iteration name.

    """
    if not selection:
        chosen = list(ITERATIONS)
    else:
        chosen = []
        for name in selection:
            full = name if name.startswith('marseille_') else f'marseille_{name}'
            if full not in ITERATIONS:
                msg = f'unknown iteration {name!r}. Expected one of: {", ".join(i[10:] for i in ITERATIONS)}'
                raise SystemExit(msg)
            chosen.append(full)

    return [f'{name}.tar.gz' for name in chosen] + [PRETRAIN_ARCHIVE]


def already_present(archive: str, resources_dir: Path) -> bool:
    """Return whether an archive's models are already unpacked."""
    if archive == PRETRAIN_ARCHIVE:
        return (resources_dir / 'la' / 'pretrain' / 'gensim_ftv_v3.pt').exists()
    package = archive.removesuffix('.tar.gz')
    return all(
        (resources_dir / 'la' / task / f'{package}.pt').exists() for task in ('tokenize', 'pos', 'lemma', 'depparse')
    )


def main() -> None:  # noqa: C901
    """Download, verify, unpack, and register the models."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('iterations', nargs='*', help="iterations to fetch, e.g. 's9'. Default is all nine")
    parser.add_argument('--doi', default=ZENODO_DOI, help='Zenodo DOI, record URL, or record id')
    parser.add_argument('--list', action='store_true', help='show what the record holds and exit')
    parser.add_argument('--force', action='store_true', help='re-download even if the models are already unpacked')
    parser.add_argument('--keep-archives', action='store_true', help='do not delete the tarballs after unpacking')
    parser.add_argument('--dir', type=Path, default=STANZA_RESOURCES_DIR, help='Stanza resources directory')
    args = parser.parse_args()

    manifest = fetch_manifest(args.doi)

    if args.list:
        print(f'{len(manifest)} file(s) in the record:\n')
        for name, entry in sorted(manifest.items()):
            print(f'  {name:<28} {entry["size"] / 1e6:8.1f} MB')
        return

    archives = wanted_archives(args.iterations)
    missing = [name for name in [*archives, REGISTRY_FRAGMENT, CHECKSUMS] if name not in manifest]
    if missing:
        msg = f'the record does not contain: {", ".join(missing)}'
        raise SystemExit(msg)

    todo = [a for a in archives if args.force or not already_present(a, args.dir)]
    skipped = [a for a in archives if a not in todo]
    for name in skipped:
        print(f'  {name:<28} already present, skipping')

    args.dir.mkdir(parents=True, exist_ok=True)
    total = sum(manifest[a]['size'] for a in todo)
    if todo:
        print(f'\nfetching {len(todo)} archive(s), {total / 1e6:.0f} MB, into {args.dir}\n')

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)

        checks = download(manifest[CHECKSUMS]['url'], staging / CHECKSUMS)
        expected = read_checksums(checks)
        fragment = download(manifest[REGISTRY_FRAGMENT]['url'], staging / REGISTRY_FRAGMENT)

        for name in [REGISTRY_FRAGMENT, *todo]:
            path = staging / name
            if name != REGISTRY_FRAGMENT:
                download(manifest[name]['url'], path, size=manifest[name]['size'])
            if name in expected and sha256(path) != expected[name]:
                msg = f'{name} failed its SHA-256 check against {CHECKSUMS}. Re-run, or report the record as corrupt'
                raise SystemExit(msg)

        for name in todo:
            files = unpack(staging / name, args.dir)
            print(f'  unpacked {name:<24} {len(files)} file(s)')
            if args.keep_archives:
                shutil.copy2(staging / name, args.dir / name)

        merged = merge_registry(fragment, args.dir / 'resources.json')

    print(f'\nregistered {merged} package entries in {args.dir / "resources.json"}')
    print('\nload a model with:\n')
    print('    import stanza')
    print("    nlp = stanza.Pipeline('la', package='marseille_s9',")
    print("                          processors='tokenize,pos,lemma,depparse', download_method=None)")


if __name__ == '__main__':
    main()
