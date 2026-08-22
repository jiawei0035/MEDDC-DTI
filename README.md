# MEDDC-DTI

Core implementation of **MEDDC-DTI: multi-route evidence decomposition and
disagreement-aware calibration for drug-target interaction prediction**.

This pre-publication repository contains source code only. Datasets, derived
embeddings, residue-contact graphs, trained checkpoints, experiment outputs,
manuscript files, and figures are intentionally excluded. The implementation
supports Davis, KIBA, and DrugBank under S1 (random-pair), S2 (cold-target), and
S3 (cold-drug) settings after the user supplies the corresponding private data.

## Repository contents

```text
MEDDC-DTI/
|-- src/                       # model, data construction, training, evaluation
|-- scripts/
|   |-- build_biochemical_prior_cache.py
|   |-- export_node_features.py
|   `-- normalize_dataset.py
|-- docs/REPRODUCIBILITY.md    # protocol and environment record
|-- run.py                     # command-line entry point
|-- environment.yml            # verified Conda environment
|-- requirements.txt           # pinned Python dependencies
|-- CITATION.cff
`-- LICENSE
```

The `.gitignore` excludes `data/`, `dataset/`, `checkpoints/`, model weights,
outputs, logs, and common local-development files.

## Installation

The reference environment uses Python 3.8.20, PyTorch 2.0.1+cu118, PyTorch
Geometric 2.3.1, and RDKit 2022.09.5:

```bash
conda env create -f environment.yml
conda activate meddc-dti
```

Alternatively, use a Python 3.8 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Private data contract

No data are distributed in this repository. The loader accepts either a legacy
file for each dataset:

```text
dataset/Davis.txt
dataset/KIBA.txt
dataset/DrugBank.txt
```

Each legacy row must end with these five whitespace-separated fields:

```text
drug_id protein_id canonical_smiles protein_sequence binary_label
```

Alternatively, the non-redundant normalized representation is supported:

```text
dataset/<dataset_name>/drugs.tsv
dataset/<dataset_name>/proteins.tsv
dataset/<dataset_name>/interactions.tsv
```

Use `scripts/normalize_dataset.py` to convert a legacy table. Required private
feature files use this layout:

```text
data/<dataset_name>/esm2/<protein_id>.pt
data/<dataset_name>/chemberta2_embeddings.pt
data/<dataset_name>/esm2_embeddings.pt
```

Each residue-level ESM2 file must contain `residue_emb` and `contact_map`.
Pooled embedding files must contain an `embeddings` tensor or an ID-keyed
mapping. The cache-building scripts document how these derived inputs are
constructed. Keep all of these files outside Git history.

## Training and evaluation

Examples for Davis are shown below; replace the dataset and setting for KIBA or
DrugBank.

```bash
python run.py --mode train --dataset Davis --setting 2 \
  --seed 3 --data_seed 3 --split_seed 3 \
  --epochs 2000 --patience 200
```

Evaluate a private checkpoint by explicitly passing its path:

```bash
python run.py --mode eval --dataset Davis --setting 2 \
  --checkpoint /private/path/best_model.pt
```

By default, generated runs are written under `outputs/`, which is ignored by
Git.

> **Protocol disclosure:** the archived research implementation selects a
> checkpoint using test AUC and does not create a separate validation split.
> This conflicts with any manuscript statement that model selection used only
> validation AUC. Before making a leakage-free validation-selection claim,
> define a train/validation/test protocol and retrain MEDDC-DTI and all compared
> baselines under the same protocol.

## Pre-publication confidentiality

Keep the GitHub repository **private** until the manuscript or preprint is
public. Before changing it to public, review the complete Git history—not only
the latest files—to ensure that no dataset, embedding, checkpoint, token,
credential, or unpublished manuscript asset was ever committed.

## Citation and license

The anticipated repository URL is recorded in `CITATION.cff`. Add the article
DOI and final journal citation after publication. The current source license is
MIT; the authors should confirm this choice before public release. External
datasets and pretrained models remain subject to their original licenses.
