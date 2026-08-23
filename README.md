# MEDDC-DTI

**Multi-route Evidence Decomposition and Disagreement-aware Calibration for Drug--Target Interaction Prediction**

MEDDC-DTI is an evidence-aware multimodal learning framework for drug--target interaction (DTI) prediction. The model integrates complementary biochemical, topological, and interaction evidence, then performs reliability-calibrated fusion to obtain robust interaction scores. This repository contains the reference implementation, data-processing utilities, and reproducible training and evaluation entry points used in the accompanying study.

## Highlights

- **Multi-route evidence decomposition**: separate encoders model biochemical features, graph topology, and observed interaction structure.
- **Multi-source biochemical priors**: optional ChemBERTa-2, RDKit, ESM-2, and protein-sequence features enrich node representations.
- **Reliability-aware evidence fusion**: adaptive softmax fusion weights evidence according to its estimated reliability.
- **Disagreement-aware calibration**: expert consensus and pair-level calibration improve prediction quality when evidence sources disagree.
- **Flexible evaluation**: supports Davis, KIBA, and DrugBank with cold-drug (S1), cold-target(S2) and  Joint new-drug/new-target prediction (S3) settings.
- **Reproducible experiments**: deterministic seeds, configurable ablations, cached feature support, and a single command-line entry point.


## Installation

The reference environment is Python 3.8.20 with PyTorch 2.0.1 (CUDA 11.8), PyTorch Geometric 2.3.1, RDKit 2022.09.5, and Transformers 4.38.2.

```bash
conda env create -f environment.yml
conda activate meddc-dti
```

Alternatively, install the pinned dependencies in a Python 3.8 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data and Features

The loader accepts the benchmark tables in `dataset/` (`Davis.txt`, `KIBA.txt`, and `DrugBank.txt`). Each row follows the format:

```text
drug_id protein_id canonical_smiles protein_sequence binary_label
```

For larger experiments, the normalized representation is also supported:

```text
dataset/<dataset_name>/drugs.tsv
dataset/<dataset_name>/proteins.tsv
dataset/<dataset_name>/interactions.tsv
```

Optional derived features can be prepared with the scripts in `scripts/`. Residue-level ESM-2 files contain `residue_emb` and `contact_map`; pooled ChemBERTa-2 and ESM-2 files contain an `embeddings` tensor or an ID-keyed mapping.

## Training

Run a training experiment with the unified entry point. The example below uses the Davis dataset under the S2 cold-target setting:

```bash
python run.py --mode train --dataset Davis --setting 2 \
  --seed 3 --data_seed 3 --split_seed 3 \
  --epochs 2000 --patience 200
```

Replace `Davis` and `2` with `KIBA` or `DrugBank` and the desired evaluation setting. Results, checkpoints, and logs are written to `outputs/`.

## Evaluation

Evaluate a trained checkpoint as follows:

```bash
python run.py --mode eval --dataset Davis --setting 2 \
  --checkpoint /path/to/best_model.pt
```

The command reports standard binary-classification metrics, including ROC-AUC, AUPR, F1, and accuracy.

## Citation

If you use MEDDC-DTI in your research, please cite the accompanying article. Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

The source code is released under the MIT License. Please consult the original licenses of external datasets and pretrained models before redistribution.
