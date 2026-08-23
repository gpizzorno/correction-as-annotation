"""Utility functions for reading and writing data."""

from __future__ import annotations

import json
from typing import Any

import conllu
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL


def load_file(filepath: str) -> Any:
    """Load a file and return its content."""
    if filepath.endswith('.conllu'):
        return load_conllu_file(filepath)
    if filepath.endswith('.json'):
        return load_json_file(filepath)
    return load_text_file(filepath)


def load_conllu_file(filepath: str) -> list[conllu.TokenList]:
    """Load a CoNLL-U file and return list of sentences."""
    with open(filepath, encoding='utf-8') as file:
        return conllu.parse(file.read())  # type:ignore[no-any-return]


def load_json_file(filepath: str) -> Any:
    """Load a JSON file and return the data as a dictionary."""
    with open(filepath, encoding='utf-8') as file:
        return json.load(file)


def load_text_file(filepath: str) -> str:
    """Load a text file and return its content as a string."""
    with open(filepath, encoding='utf-8') as file:
        return file.read()


def write_stanza_document(doc: Document, sent_metadata: list[dict[str, str]], filepath: str) -> None:
    """Write Stanza document to a file in CoNLL-U format.

    Arguments:
        doc: Stanza Document object containing the parsed sentences.
        sent_metadata: List of metadata dictionaries for each sentence.
            {'sent_id': str, 'text': str}
        filepath: Path to the output file.

    """
    sentences = CoNLL.doc2conll(doc)

    with open(filepath, 'w', encoding='utf-8') as file:
        # strict: a count mismatch means the metadata belongs to different sentences, which would
        # silently mislabel every sentence from that point on
        for sent, metadata in zip(sentences, sent_metadata, strict=True):
            file.write(f'# sent_id = {metadata["sent_id"]}\n')
            file.write(f'# text = {metadata["text"]}\n')
            for token_line in sent:
                file.write(token_line)
                file.write('\n')
            file.write('\n')
