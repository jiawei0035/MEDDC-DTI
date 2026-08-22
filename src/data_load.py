import numpy as np
from pathlib import Path


def shuffle_dataset(dataset, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(dataset)
    return dataset


def load_interactions(dataset_name='Davis', dataset_dir='dataset', seed=3):
    dataset_dir = Path(dataset_dir)
    normalized_dir = dataset_dir / dataset_name.lower()
    interactions_path = normalized_dir / 'interactions.tsv'
    drug_ids, protein_ids, interactions = [], [], []
    if interactions_path.exists():
        with (normalized_dir / 'drugs.tsv').open('r', encoding='utf-8') as handle:
            drugs = dict(line.rstrip('\n').split('\t', 1) for line in list(handle)[1:])
        with (normalized_dir / 'proteins.tsv').open('r', encoding='utf-8') as handle:
            proteins = dict(line.rstrip('\n').split('\t', 1) for line in list(handle)[1:])
        with interactions_path.open('r', encoding='utf-8') as handle:
            rows = [line.rstrip('\n').split('\t') for line in list(handle)[1:]]
        rows = shuffle_dataset(rows, seed)
        parsed_rows = ((row[0], row[1], drugs[row[0]], proteins[row[1]], int(row[2])) for row in rows)
    else:
        dataset_path = dataset_dir / f'{dataset_name}.txt'
        with dataset_path.open('r', encoding='utf-8') as handle:
            rows = shuffle_dataset(handle.read().strip().split('\n'), seed)
        parsed_rows = ((fields[-5], fields[-4], fields[-3], fields[-2], int(fields[-1])) for fields in (row.strip().split() for row in rows))
    for drug_id, protein_id, smiles, sequence, label in parsed_rows:
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        interactions.append((drug_id, protein_id, label, smiles, sequence))
    return interactions, len(set(drug_ids)), len(set(protein_ids))


def load_davis_interactions(dataset_path='dataset/Davis.txt', seed=3):
    dataset_name = dataset_path.replace('\\', '/').split('/')[-1].replace('.txt', '')
    dataset_dir = '/'.join(dataset_path.replace('\\', '/').split('/')[:-1]) or 'dataset'
    return load_interactions(dataset_name, dataset_dir, seed)
