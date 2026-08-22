# Reproducibility record

## Supported experiment matrix

The implementation accepts Davis, KIBA, and DrugBank with the following split
settings:

- S1: random interaction-pair split;
- S2: cold-target split;
- S3: cold-drug split.

The default reference configuration uses model seed, data seed, and split seed
3. Dataset-specific epoch and patience defaults are defined in `src/main.py`.

## Verified software environment

- Python 3.8.20
- PyTorch 2.0.1+cu118
- PyTorch Geometric 2.3.1
- NumPy 1.24.3
- SciPy 1.10.1
- scikit-learn 1.3.0
- NetworkX 3.1
- RDKit 2022.09.5
- Transformers 4.38.2

Use `environment.yml` for the closest reproduction of the research runtime.
Newer PyTorch releases can introduce small numerical changes. PyTorch also
warns that the CUDA implementation of one `scatter_reduce` operation is not
fully deterministic; this limitation should be reported with final results.

## Private artifacts

Datasets, derived embeddings, residue-contact graphs, trained weights, and
experiment outputs are deliberately absent from this pre-publication code-only
repository. Store each artifact in a versioned private location and record its
SHA-256 checksum. When journal policy requires public reproducibility, publish
the permitted artifacts through a persistent repository such as Zenodo and add
the DOI and checksums to this document.

## Model-selection disclosure

The archived training implementation selects its best checkpoint using test
AUC. It does not currently create a separate validation set. This may yield an
optimistic estimate and must not be described as validation-only model
selection. A journal-facing leakage-free experiment requires a predefined
train/validation/test split, validation-based early stopping, and one final
evaluation on the untouched test set for every method.

## Final release record

Before public release, record the repository commit, artifact version, dataset
provenance, preprocessing commands, GPU model, CUDA version, operating system,
wall time, peak GPU memory, and all random seeds.
