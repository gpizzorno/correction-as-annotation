"""Ordered cleanup steps that produce 'la_marseille-ud-base-fixed.conllu'."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bootstrapping.config import PROJECT_ROOT
from bootstrapping.io import load_file

if TYPE_CHECKING:
    import conllu

BASE_PATH = Path(PROJECT_ROOT) / 'data' / 'corpora' / 'la_marseille-ud-base.conllu'
FIXED_PATH = Path(PROJECT_ROOT) / 'data' / 'corpora' / 'la_marseille-ud-base-fixed.conllu'

ROMAN_CHARACTERS = frozenset('ivxlcdm')

# numeral lemmata that inflect and therefore have to agree with a neighbouring noun
INFLECTING_LEMMATA = frozenset({'a', 'unus', 'duo', 'tres'})

# the head-noun search widens in three passes
# from strict to anything-but-'item'
PREFERRED_UPOS = ('NOUN', 'PROPN')
FALLBACK_UPOS = ('NOUN', 'PROPN', 'ADJ', 'VERB')


def is_roman_numeral(text: str) -> bool:
    """Return True if every character is a roman numeral character."""
    return bool(text) and all(character in ROMAN_CHARACTERS for character in text.lower())


def _nearest_noun(sentence: conllu.TokenList, index: int) -> Any:
    """Find the noun a numeral should agree with: nearest first, widening the criteria."""
    for allowed, skip_item in ((PREFERRED_UPOS, False), (FALLBACK_UPOS, True), (None, True)):
        left, right = index - 1, index + 1
        while left >= 0 or right < len(sentence):
            for position in (left, right):
                if not 0 <= position < len(sentence):
                    continue
                candidate = sentence[position]
                if skip_item and (candidate['form'] or '').lower() == 'item':
                    continue
                if allowed is None or candidate['upos'] in allowed:
                    return candidate
            left -= 1
            right += 1
    return None


def agreeing_form(lemma: str, noun: Any) -> str:  # noqa: C901, PLR0911
    """Return the numeral word form that agrees with 'noun'."""
    form = noun['form'] or ''
    if lemma in ('a', 'unus'):
        if form.endswith('am'):
            return 'unam'
        if form.endswith('a'):
            return 'una'
        if form.endswith('o'):
            return 'uno'
        return 'unum'
    if lemma == 'duo':
        if form.endswith(('as', 'es')):
            return 'duas'
        if form.endswith('os'):
            return 'duos'
        if form.endswith('arum'):
            return 'duarum'
        if form.endswith('orum'):
            return 'duorum'
        return 'duo'
    if form.endswith(('i', 'a')):
        return 'tria'
    return 'tres'


def fix_roman_numerals(sentences: list[conllu.TokenList]) -> dict[str, int]:
    """Expand roman-numeral NUM tokens into Latin words, in place."""
    report = {'forms': 0, 'lemmata': 0}

    for sentence in sentences:
        for token in sentence:
            if token['upos'] != 'NUM' or not is_roman_numeral(token['form'] or ''):
                continue

            original_form, original_lemma = token['form'], token['lemma']

            if token['lemma'] in INFLECTING_LEMMATA:
                noun = _nearest_noun(sentence, token['id'] - 1)
                token['form'] = agreeing_form(token['lemma'], noun) if noun else token['lemma']
                if token['lemma'] == 'a':
                    token['lemma'] = 'unus'
            elif token['lemma'] == 'quadraginta' and (token['form'] or '').lower() == 'iiii':
                token['form'], token['lemma'] = 'quatuor', 'quattuor'
            else:
                token['form'] = token['lemma']  # indeclinable

            report['forms'] += token['form'] != original_form
            report['lemmata'] += token['lemma'] != original_lemma

    return report


def fix_numform(sentences: list[conllu.TokenList]) -> dict[str, int]:
    """Mark every digit-form numeral as written out in words, in place."""
    changed = 0
    for sentence in sentences:
        for token in sentence:
            if token['upos'] == 'NUM' and token['feats'] and token['feats'].get('NumForm') == 'Digit':
                token['feats']['NumForm'] = 'Word'
                changed += 1
    return {'numform': changed}


def fix_quondam(sentences: list[conllu.TokenList]) -> dict[str, int]:
    """Normalise the lemma 'condam' to 'quondam', in place."""
    changed = 0
    for sentence in sentences:
        for token in sentence:
            if token['lemma'] == 'condam':
                token['lemma'] = 'quondam'
                changed += 1
    return {'lemmata': changed}


# cleanup steps in the order they must be run
STEPS = [
    ('roman numerals', fix_roman_numerals),
    ('numform', fix_numform),
    ('quondam', fix_quondam),
]


def build(source: Path | str = BASE_PATH) -> tuple[list[conllu.TokenList], dict[str, dict[str, int]]]:
    """Apply every cleanup step to the base corpus and return the sentences and a per-step report."""
    sentences = load_file(str(source))
    return sentences, {name: step(sentences) for name, step in STEPS}


def serialise(sentences: list[conllu.TokenList]) -> str:
    """Render sentences as CoNLL-U text."""
    return ''.join(sentence.serialize() for sentence in sentences)


def main() -> None:
    """Rebuild the fixed corpus."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=BASE_PATH)
    parser.add_argument('--output', type=Path, default=FIXED_PATH)
    args = parser.parse_args()

    sentences, reports = build(args.source)
    for name, report in reports.items():
        print(f'  {name:16} ' + ', '.join(f'{k}={v:,}' for k, v in report.items()))

    rebuilt = serialise(sentences)

    args.output.write_text(rebuilt, encoding='utf-8')
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
