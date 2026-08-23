# Correction as Annotation

This repository contains the corpus, code, and evaluation data accompanying the paper *Correction as Annotation: Bootstrapping a Dependency Parser for Documentary Medieval Latin*.

Documentary sources from the medieval archive remain poorly served by existing NLP tools, and not solely by virtue of their age. On a corpus of 160 inventories compiled in Marseille between 1258 and 1446, none of the five available Latin treebank models is suitable: the highest labelled attachment score is 0.62. This repository provides the corpus, the gold standard, and the code for a human-in-the-loop procedure that produces the required in-domain training data as a by-product of using those inadequate models. The process consisted of nine rounds in which a model pre-annotates 200 sentences, an expert corrects them, and the corrections are used to train the subsequent model.

|  |  |
|---|---|
| Corpus | DALME-Marseille: 160 inventories from Marseille, 1258–1446. |
| Annotation | 1,804 hand-corrected sentences, 14,516 tokens, 33.2 hours. |
| Final model | 0.977 UPoS · 0.917 LAS · 0.822 MLAS on a 200-sentence gold standard. |
| Against the baseline | 0.80 → 0.98 UPoS and 0.48 → 0.92 LAS, on 97% less training data. |
| Trained models | [10.5281/zenodo.22048602](https://doi.org/10.5281/zenodo.22048602) — 917 MB, archived separately. |

## The Corpus

If you require only the annotated corpus, it may be used directly (no installation is necessary).

| File | Size | |
|---|---|---|
| [`data/corpora/la_marseille-ud-full.conllu`](data/corpora/la_marseille-ud-full.conllu) | 6,903 sentences, 54,294 tokens. | The entire corpus, parsed by the final model. |
| [`data/gold/la_marseille-ud-gold.conllu`](data/gold/la_marseille-ud-gold.conllu) | 200 sentences, 1,441 tokens. | Hand-corrected throughout. Every score reported in the paper is computed against this. |
| [`data/seeds/corrected/`](data/seeds/corrected/) | 1,804 sentences. | The nine hand-corrected batches, presented in the order they were annotated. |

Annotations are in CoNLL-U format throughout, following the Universal Dependencies guidelines. Two fields in the MISC column merit particular attention, both present on every one of the 54,294 tokens:

```conllu
# sent_id = 089AD9DF-38D8-468D-A5E0-01F47EF9CDDC
# text = Item unum alium smaragdum et aliud saphirum cum virgis aureis.
1  Item       item       ADV   d--------  _                     4  advmod  4:advmod  Gloss=next|start_char=30|end_char=34
2  unum       unus       NUM   m-s---nn-  Case=Nom|Gender=Neut  4  nummod  4:nummod  Gloss=one|start_char=35|end_char=39
3  alium      alius      DET   p-s---ma-  Case=Acc|Gender=Masc  4  det     4:det     Gloss=other|start_char=40|end_char=45
4  smaragdum  smaragdus  NOUN  n-s---nn-  Case=Nom|Gender=Neut  0  root    0:root    Gloss=emerald|start_char=46|end_char=55
```

`Gloss=` provides an English gloss per token, and the `start_char`/`end_char` offsets map annotations back to the source transcription, so every annotation can be traced to its location in the document.

The corpus derives from the [Documentary Archaeology of Late Medieval Europe](https://dalme.org)
(DALME) project and is licenced under CC BY 4.0. See the per-file statements in [`data/LICENSE`](data/LICENSE).

## Installation

Python 3.11 is required.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The prior command installs the package in editable mode with the `notebooks` and `vectors` extras. Add the `dev` extra to obtain `ruff`, `mypy` and `pytest`.

## Regenerating the Paper's Outputs

This process requires no models. The scored evaluation output is included as data and every table and figure is generated from files in this repository together with the comparison treebanks, which are fetched rather than included.

```bash
python -m bootstrapping.corpora      # fetch the 7 UD treebanks    ~98 MB, download-bound
python -m bootstrapping.lemmata      # build the lemma lists       ~40 s
python -m bootstrapping.statistics   # compile corpus statistics   ~25 s, writes 53 MB
```

Execute these commands in the listed order, since each step reads outputs produced by the preceding one. Then either open [`pipeline.ipynb`](pipeline.ipynb), which reproduces the paper section by section, or proceed directly to the artefacts:

```bash
python -m bootstrapping.report.tables         # the 10 tables, into build/tables/   ~40 s
python -m bootstrapping.report.paper_figures  # the 6 figures, into build/figures/   ~5 s
```

## Using the Models

The nine trained iterations are hosted on Zenodo because their aggregate size is 917 MB. Each iteration is distributed as a complete Stanza pipeline (tokenizer, tagger, lemmatizer, and dependency parser) trained from scratch on the pool of corrected sentences accumulated up to that round.

```bash
python -c "import stanza; stanza.download('la')"   # once, to create Stanza's registry
python scripts/fetch_models.py s9                  # the final model, ~110 MB
```

Invoking `fetch_models.py` without arguments downloads all nine iterations (~900 MB). The script verifies the archive using the deposit's `SHA256SUMS`, unpacks into `STANZA_RESOURCES_DIR`, and merges the package registry, so that Stanza can resolve the packages by name.

```python
import stanza

nlp = stanza.Pipeline('la', package='marseille_s9',
                      processors='tokenize,pos,lemma,depparse',
                      download_method=None, tokenize_pretokenized=True)
doc = nlp('item unum morterium lapidis')
```

The distributed lemmatizers are the trained models without additional patches. The paper's *Ensemble* results are obtained by injecting the domain lexicon into the lemmatizer dictionaries. The command `python scripts/build_ensemble_models.py` performs that operation using the lexicon contained in this repository.

## Retraining

All downstream artefacts reproduce precisely, including evaluation output, tables, figures, and the numerical results reported in the paper. The trained models themselves are the sole element in the chain that this repository cannot reproduce bit-for-bit.

The command `scripts/train_iteration.py 9` constructs iteration 9's UD treebank from the corrected batches and prints the three Stanza training commands (supplying `--execute` will run them). The training-set construction is exact: `training.py` pools batches `s1…s9`, shuffles with a fixed seed, performs a 90/10 split, and Stanza again splits the 90% set into an 80/20 split for its internal development set, so the models are trained on approximately 72% of the pool. That process is reproducible.

## Project Structure

```
src/bootstrapping/    the package: everything that reads this repository's own data
scripts/              entry points that reach outside it, *e.g.* the model store, another venv, a GPU
tests/                pins the lexicon derivation, the lemma lists, and the model fetcher
data/                 corpus, gold standard, seed batches, evaluation output, lexicon
pipeline.ipynb        the paper, section by section, regenerating as it goes
```

The repository follows a single organizing rule: the package operates on this repository's data while the scripts access resources external to the repository (for example the model store, a separate virtual environment, or a GPU).

### The Package

▶ denotes a module with a command-line entry point, *i.e.*, it can be run as: `python -m bootstrapping.<name>`.

**Foundations**

| Module | |
|---|---|
| `config.py` | Every path used by the project, plus the registry of iterations reported in the paper. |
| `io.py` | CoNLL-U, JSON, and plain-text readers, dispatched on file extension. |
| `extractors.py` | Lemma cleaning, and extraction of unique-lemma and PoS counts from parsed text. |
| `parsers.py` | Latin/Occitan classification via fuzzy lookup against references. These are loaded lazily so that dependent modules may operate before they are built. |

**Building the Corpus and Other Resources**

| Module | |
|---|---|
| `corpora.py` ▶ | Fetches the seven comparison treebanks from Universal Dependencies. |
| `corpus/build.py` ▶ | Implements the three mechanical corrections of §5.2 (Roman numerals, the `NumForm` feature, and the `condam` lemma) |
| `sampling.py` | Length-stratified batch selection (§5.3). |
| `lexicon.py` ▶ | Constructs the domain lexicon and produces the gold-free master by subtraction (§4.4). |
| `lemmata.py` ▶ | Builds lemma inventories per treebank and the two language references consumed by `parsers.py`. |
| `statistics.py` ▶ | Compiles `corpus_statistics.json`, which underpins Tables 1, 4, and 8, as well as the Figure 2. |

**Training, Evaluation, and Analysis**

| Module | |
|---|---|
| `training.py` | Training-set construction and the Stanza training procedure (§5.7). |
| `evaluation.py` | Executes a model configuration over the gold standard and computes CoNLL 2018 metrics. |
| `consistency.py` | Measures intra-annotator consistency over repeated constructions (Table 5, §5.6). |
| `learning.py` | Shared loading and frame construction for the learning-dynamics analyses of §6.3. |
| `analysis/comparison.py` | Prediction-against-gold alignment, bootstrap, and paired *t*-tests, as well as error tracking and construction classification. |
| `sentence_metrics/` | Per-sentence measures used in §4.7: `token_count`, `tree_depth`, `distance` (mean dependency distance), `gbsc`. |

**Reporting**

| Module | |
|---|---|
| `report/tables.py` ▶ | Generates the paper's 10 tables as LaTeX. |
| `report/paper_figures.py` ▶ | Produces the paper's 6 figures. |
| `report/figures.py` | Shared matplotlib styling. |

### The Scripts

The external resources required by each script are summarized in the second column.

| Script | Reaches for |
|---|---|
| `fetch_models.py` | Zenodo to download, verify, and install the models into `STANZA_RESOURCES_DIR`. |
| `build_ensemble_models.py` | The model store, to patch the domain lexicon into each lemmatizer model. |
| `evaluate_models.py` | The trained models to parse the gold standard with each configuration and write results to `data/evaluation/`. |
| `train_iteration.py` | A GPU to construct a single iteration's UD treebank and print or execute the Stanza training commands.|
| `convert_pretrain.py` | The word vectors to convert them into the format current Stanza can consume. |
| `zero_shot_baseline.py` | A second virtual environment using current Stanza to run the §4.5 comparison reported in Table 3. |

## Licences

The repository uses multiple licences.

| | |
|---|---|
| Code (`src/`, `scripts/`) | MIT, see [`LICENSE`](LICENSE). |
| Datasets (`data/`) | Documented file-by-file, with provenance and licensing information. See [`data/LICENSE`](data/LICENSE). |
| Trained models | CC BY 4.0, as stated in the [Zenodo deposit](https://doi.org/10.5281/zenodo.22048602). |

## Citation

```bibtex
@article{pizzorno_correction_as_annotation,
  author = {Pizzorno, Gabriel H.},
  title  = {Correction as Annotation: Bootstrapping a Dependency Parser
            for Documentary Medieval Latin},
  year   = {2026}
}

@dataset{pizzorno_marseille_models,
  author    = {Pizzorno, Gabriel H.},
  title     = {Bootstrapped dependency models for documentary medieval Latin
                (DALME-Marseille), iterations s1--s9},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22048602}
}
```
