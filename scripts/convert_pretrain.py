"""Convert a Stanza 1.2.1 pretrain file to the format current Stanza can load.

Nothing about the data changes. 'BaseVocab.state_attrs' is identical between 1.2.1
and 1.14, so this rewrites the vocab as the plain state dict current Stanza expects
and leaves the embedding matrix alone.

Run this under the 1.2.1 venv, since it needs those classes on the path to unpickle
the original:

    .venv/bin/python scripts/convert_pretrain.py \
        ~/stanza_resources/la/pretrain/gensim_ftv_v1.pt \
        ~/stanza_resources_retrain/la/pretrain/gensim_ftv_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# written by both versions. kept so the output matches what current Stanza produces itself
ZIPFILE_SERIALIZATION = False


def convert(source: Path, destination: Path) -> dict[str, object]:
    """Rewrite one pretrain file in the state-dict format and return a summary.

    Arguments:
        source: a Stanza 1.2.1 pretrain '.pt'.
        destination: directory to hold output (parent directories are created)

    """
    data = torch.load(source, map_location='cpu', weights_only=False)

    if not isinstance(data, dict) or 'vocab' not in data or 'emb' not in data:
        msg = f"{source} is not a Stanza pretrain file. Expected a dict with 'vocab' and 'emb'."
        raise ValueError(msg)

    vocab = data['vocab']
    # already converted, or built by a version that stored the state dict directly
    state = vocab if isinstance(vocab, dict) else vocab.state_dict()

    emb = data['emb']
    # weights_only=True rejects numpy arrays as well as classes, so normalise to a tensor
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)

    destination.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {'vocab': state, 'emb': emb},
        destination,
        _use_new_zipfile_serialization=ZIPFILE_SERIALIZATION,
    )

    return {
        'vocab_size': len(state['_id2unit']),
        'emb_shape': tuple(emb.shape),
        'dtype': str(emb.dtype),
        'was_state_dict': isinstance(vocab, dict),
    }


def main() -> int:
    """Convert one pretrain file and report results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='Stanza 1.2.1 pretrain .pt (read only)')
    parser.add_argument('destination', type=Path, help='where to write the converted file')
    args = parser.parse_args()

    if args.source.resolve() == args.destination.resolve():
        print('Cannot overwrite the source in place.', file=sys.stderr)
        return 1

    summary = convert(args.source, args.destination)
    print(f'{args.source} -> {args.destination}')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
