import random
import os
import json
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from scipy.sparse import coo_matrix

from config import MODEL_CONFIG
from graph_data import GraphDataset, ProteinGraphDataset, collate_graphs
from protein_graph import build_esm2_contact_graph


def one_of_k_encoding(value, allowable_set):
    if value not in allowable_set:
        raise ValueError(f'{value} is not in allowable set {allowable_set}')
    return [value == item for item in allowable_set]


def one_of_k_encoding_unk(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
         'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
         'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
         'Pt', 'Hg', 'Pb', 'X']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()], dtype=np.float32)


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    node_features = []
    for atom in molecule.GetAtoms():
        feature = atom_features(atom)
        node_features.append(feature / sum(feature))
    edges = []
    for bond in molecule.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    graph = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((molecule.GetNumAtoms(), molecule.GetNumAtoms()))
    for src, dst in graph.edges:
        mol_adj[src, dst] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    row_index, col_index = np.where(mol_adj >= 0.5)
    for row, col in zip(row_index, col_index):
        edge_index.append([row, col])
    return molecule.GetNumAtoms(), np.asarray(node_features, dtype=np.float32), np.asarray(edge_index, dtype=np.int64)


def mask_negatives(num_edges, ratio, seed):
    mask = np.ones(num_edges, dtype=bool)
    mask[:int(ratio * num_edges)] = False
    rng = np.random.RandomState(seed)
    rng.shuffle(mask)
    return mask


def normalize_adjacency(matrix):
    row_sum = matrix.sum(1)
    inv_sqrt = np.power(row_sum, -0.5).flatten()
    inv_sqrt[np.isinf(inv_sqrt)] = 0.0
    inv_sqrt_matrix = np.diag(inv_sqrt)
    return inv_sqrt_matrix.dot(matrix).dot(inv_sqrt_matrix)


def positive_common_neighbor_graph(node_count, adjacency, common_neighbor_threshold=5):
    adjacency_np = adjacency.numpy().astype(np.float32)
    common = adjacency_np @ adjacency_np.T
    return torch.from_numpy((common > common_neighbor_threshold).astype(np.float32))


def compute_drug_morgan_prior(smiles_list):
    features = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            features.append(np.zeros(1032, dtype=np.float32))
            continue
        descriptors = np.array([
            Descriptors.MolWt(molecule) / 500.0,
            Descriptors.MolLogP(molecule) / 5.0,
            Descriptors.TPSA(molecule) / 140.0,
            Descriptors.NumHDonors(molecule) / 10.0,
            Descriptors.NumHAcceptors(molecule) / 10.0,
            Descriptors.NumRotatableBonds(molecule) / 10.0,
            Descriptors.NumAromaticRings(molecule) / 5.0,
            Descriptors.qed(molecule),
        ], dtype=np.float32)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        fingerprint_array = np.array(fingerprint, dtype=np.float32)
        features.append(np.concatenate([descriptors, fingerprint_array]))
    return np.array(features, dtype=np.float32)


def compute_drug_auxiliary_features(smiles_list):
    features = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            features.append(np.zeros(1202, dtype=np.float32))
            continue
        descriptors = np.array([
            Descriptors.MolWt(molecule) / 500.0,
            Descriptors.MolLogP(molecule) / 5.0,
            Descriptors.TPSA(molecule) / 140.0,
            Descriptors.NumHDonors(molecule) / 10.0,
            Descriptors.NumHAcceptors(molecule) / 10.0,
            Descriptors.NumRotatableBonds(molecule) / 10.0,
            Descriptors.NumAromaticRings(molecule) / 5.0,
            Descriptors.RingCount(molecule) / 8.0,
            Descriptors.HeavyAtomCount(molecule) / 80.0,
            Descriptors.FractionCSP3(molecule),
            Descriptors.qed(molecule),
        ], dtype=np.float32)
        morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024), dtype=np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(molecule), dtype=np.float32)
        features.append(np.concatenate([descriptors, morgan, maccs]))
    features = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - mean) / std).astype(np.float32)


def compute_protein_sequence_prior(sequences):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    hydrophobic = set('AILMFWYV')
    charged = set('DEKRH')
    polar = set('STNQCY')
    features = []
    for sequence in sequences:
        seq = sequence or ''
        length = max(len(seq), 1)
        counts = np.array([seq.count(aa) / length for aa in amino_acids], dtype=np.float32)
        summary = np.array([
            np.log1p(length) / 8.0,
            sum(1 for aa in seq if aa in hydrophobic) / length,
            sum(1 for aa in seq if aa in charged) / length,
            sum(1 for aa in seq if aa in polar) / length,
            seq.count('C') / length,
        ], dtype=np.float32)
        features.append(np.concatenate([summary, counts]))
    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def standardize_prior_features(features):
    features = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - mean) / std).astype(np.float32)


def project_prior_features(features, out_dim, seed):
    features = standardize_prior_features(features)
    rng = np.random.RandomState(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(max(features.shape[1], 1)), size=(features.shape[1], out_dim)).astype(np.float32)
    return standardize_prior_features(features @ projection)


def build_drug_drug_edges(fingerprints, top_k=5, threshold=0.5):
    n_nodes = fingerprints.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if n_nodes < 2:
        return adjacency
    dot = fingerprints @ fingerprints.T
    sums = fingerprints.sum(axis=1)
    union = sums[:, None] + sums[None, :] - dot + 1e-8
    similarity = (dot / union).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    k = min(top_k, n_nodes - 1)
    top_index = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_nodes), k)
    cols = top_index.reshape(-1)
    values = similarity[rows, cols]
    mask = values > threshold
    adjacency[rows[mask], cols[mask]] = values[mask]
    return np.maximum(adjacency, adjacency.T)


def build_drug_auxiliary_edges(features, top_k=8, threshold=0.25):
    n_nodes = features.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if n_nodes < 2:
        return adjacency
    features = features.astype(np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normalized = features / norm
    similarity = normalized @ normalized.T
    similarity = np.nan_to_num(similarity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)
    k = min(top_k, n_nodes - 1)
    top_index = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_nodes), k)
    cols = top_index.reshape(-1)
    values = similarity[rows, cols]
    mask = values > threshold
    adjacency[rows[mask], cols[mask]] = values[mask]
    return np.maximum(adjacency, adjacency.T)


def load_cached_embeddings(dataset_name, names, node_ids):
    data_root = os.environ.get('MEDDC_DTI_DATA_DIR', 'data')
    candidate_dirs = [
        os.path.join(data_root, dataset_name.lower()),
        os.path.join(data_root, dataset_name),
        os.path.join(data_root, 'shared'),
        os.path.join(data_root, 'features'),
        data_root,
    ]
    seen_dirs = []
    for directory in candidate_dirs:
        if directory not in seen_dirs:
            seen_dirs.append(directory)
    for directory in seen_dirs:
        for name in names:
            for suffix in ('.npy', '.npz', '.pt'):
                path = os.path.join(directory, name + suffix)
                if not os.path.exists(path):
                    continue
                if suffix == '.pt':
                    payload = torch.load(path, map_location='cpu', weights_only=False)
                    if isinstance(payload, torch.Tensor):
                        return payload.float().numpy()
                    if isinstance(payload, dict):
                        values = payload.get('embeddings', payload)
                        if isinstance(values, torch.Tensor):
                            return values.float().numpy()
                        if isinstance(values, np.ndarray):
                            return values.astype(np.float32)
                        return np.stack([np.asarray(values[node_id], dtype=np.float32) for node_id in node_ids])
                if suffix == '.npz':
                    payload = np.load(path, allow_pickle=True)
                    if 'embeddings' in payload:
                        return payload['embeddings'].astype(np.float32)
                    first_key = payload.files[0]
                    return payload[first_key].astype(np.float32)
                return np.load(path, allow_pickle=True).astype(np.float32)
            path = os.path.join(directory, dataset_name.lower() + '_' + name + '.pt')
            if not os.path.exists(path):
                continue
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(payload, torch.Tensor):
                return payload.float().numpy()
            if isinstance(payload, dict):
                values = payload.get('embeddings', payload)
                if isinstance(values, torch.Tensor):
                    return values.float().numpy()
                if isinstance(values, np.ndarray):
                    return values.astype(np.float32)
                return np.stack([np.asarray(values[node_id], dtype=np.float32) for node_id in node_ids])
    return None


def load_biochemical_prior_audit(dataset_name):
    for directory in (os.path.join('data', dataset_name.lower()), os.path.join('data', dataset_name), 'data'):
        path = os.path.join(directory, 'biochemical_prior_audit.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
    return {}


def sample_unobserved_negative_edges(nb_drugs, nb_proteins, observed_edges, n_sample, seed, excluded_drugs=None, excluded_proteins=None, excluded_pairs=None):
    if n_sample <= 0:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.RandomState(seed)
    observed = set((int(d), int(p)) for d, p in observed_edges)
    excluded_drugs = set() if excluded_drugs is None else set(int(item) for item in excluded_drugs)
    excluded_proteins = set() if excluded_proteins is None else set(int(item) for item in excluded_proteins)
    excluded_pairs = set() if excluded_pairs is None else set((int(d), int(p)) for d, p in excluded_pairs)
    negatives = []
    max_attempts = max(n_sample * 20, 1000)
    attempts = 0
    while len(negatives) < n_sample and attempts < max_attempts:
        drug = int(rng.randint(0, nb_drugs))
        protein = int(rng.randint(0, nb_proteins))
        pair = (drug, protein)
        if drug not in excluded_drugs and protein not in excluded_proteins and pair not in excluded_pairs and pair not in observed:
            observed.add(pair)
            negatives.append(pair)
        attempts += 1
    return np.asarray(negatives, dtype=np.int64)


def split_exclusion_sets(nb_drugs, nb_proteins, setting, split_seed):
    if setting == 1:
        return set(get_random_folds(nb_drugs, 5, split_seed)[4]), set(), set()
    if setting == 2:
        return set(), set(get_random_folds(nb_proteins, 5, split_seed)[4]), set()
    if setting == 3:
        test_drugs = set(get_random_folds(nb_drugs, 5, split_seed)[4])
        test_proteins = set(get_random_folds(nb_proteins, 5, None if split_seed is None else split_seed + 100000)[4])
        return set(), set(), set((drug, protein) for drug in test_drugs for protein in test_proteins)
    return set(), set(), set()


def build_protein_protein_edges(sequences, k=3, top_k=5, threshold=0.3):
    vocabulary = {}
    bags = []
    for sequence in sequences:
        bag = {}
        for start in range(len(sequence) - k + 1):
            kmer = sequence[start:start + k]
            if kmer not in vocabulary:
                vocabulary[kmer] = len(vocabulary)
            index = vocabulary[kmer]
            bag[index] = bag.get(index, 0) + 1
        bags.append(bag)
    n_nodes = len(sequences)
    vocab_size = len(vocabulary)
    if vocab_size == 0 or n_nodes < 2:
        return np.zeros((n_nodes, n_nodes), dtype=np.float32)
    matrix = np.zeros((n_nodes, vocab_size), dtype=np.float32)
    for row, bag in enumerate(bags):
        for col, count in bag.items():
            matrix[row, col] = count
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    similarity = (matrix @ matrix.T).astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    adjacency = np.zeros_like(similarity)
    kk = min(top_k, n_nodes - 1)
    top_index = np.argpartition(-similarity, kk - 1, axis=1)[:, :kk]
    rows = np.repeat(np.arange(n_nodes), kk)
    cols = top_index.reshape(-1)
    values = similarity[rows, cols]
    mask = values > threshold
    adjacency[rows[mask], cols[mask]] = values[mask]
    return np.maximum(adjacency, adjacency.T)


def dense_to_sparse_tensor(matrix):
    sparse_matrix = sp.coo_matrix(matrix)
    indices = torch.LongTensor(np.vstack((sparse_matrix.row, sparse_matrix.col)))
    values = torch.FloatTensor(sparse_matrix.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_matrix.shape))


def get_random_folds(total_size, fold_count, seed):
    folds = []
    rng = random.Random(seed)
    remaining = set(range(total_size))
    fold_size = total_size / fold_count
    leftover = total_size % fold_count
    for _ in range(fold_count):
        current_size = fold_size
        if leftover > 0:
            current_size += 1
            leftover -= 1
        fold = rng.sample(sorted(remaining), int(current_size))
        remaining = remaining.difference(fold)
        folds.append(fold)
    fold_union = set()
    for fold in folds:
        assert len(set(fold) & fold_union) == 0
        fold_union |= set(fold)
    assert len(fold_union & set(range(total_size))) == total_size
    return folds


def negative_sampling_ratio(dataset_name):
    if dataset_name == 'Davis':
        return 0.6032
    if dataset_name == 'KIBA':
        return 0.7648
    if dataset_name == 'DrugBank':
        return 0.0
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def split_interaction_edges(all_edges, nb_drugs, nb_proteins, setting, split_seed):
    if setting == 1:
        folds = get_random_folds(nb_drugs, 5, split_seed)
        test_set = set(folds[4])
        drug_indices = all_edges[:, 0]
        is_test = np.isin(drug_indices, list(test_set))
    elif setting == 2:
        folds = get_random_folds(nb_proteins, 5, split_seed)
        test_set = set(folds[4])
        protein_indices = all_edges[:, 1] - nb_drugs
        is_test = np.isin(protein_indices, list(test_set))
    elif setting == 3:
        drug_folds = get_random_folds(nb_drugs, 5, split_seed)
        protein_folds = get_random_folds(nb_proteins, 5, None if split_seed is None else split_seed + 100000)
        test_drugs = set(drug_folds[4])
        test_proteins = set(protein_folds[4])
        drug_indices = all_edges[:, 0]
        protein_indices = all_edges[:, 1] - nb_drugs
        is_test = np.isin(drug_indices, list(test_drugs)) & np.isin(protein_indices, list(test_proteins))
    else:
        raise ValueError(f'Unsupported setting: {setting}')
    train_edges = all_edges[~is_test].copy()
    test_edges = all_edges[is_test].copy()
    train_edges[:, 1] -= nb_drugs
    test_edges[:, 1] -= nb_drugs
    return train_edges, test_edges


def build_interaction_dataset(interactions, nb_drugs, nb_proteins, dataset_name='Davis', setting=2, split_seed=3, config=MODEL_CONFIG, build_graph_loaders=True):
    drug_ids = sorted(set(item[0] for item in interactions))
    protein_ids = sorted(set(item[1] for item in interactions))
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    protein_to_idx = {protein_id: idx + len(drug_ids) for idx, protein_id in enumerate(protein_ids)}

    drug_num = np.array([drug_to_idx[item[0]] for item in interactions])
    protein_num = np.array([protein_to_idx[item[1]] for item in interactions])
    labels = np.array([item[2] for item in interactions])
    all_pairs = np.vstack((drug_num, protein_num, labels)).T
    all_pairs = all_pairs[all_pairs[:, 2].argsort()]

    positive_edges = all_pairs[all_pairs[:, 2] == 1, 0:2]
    negative_edges_original = all_pairs[all_pairs[:, 2] == 0, 0:2]
    negative_mask = mask_negatives(len(negative_edges_original), negative_sampling_ratio(dataset_name), 666)
    negative_edges = negative_edges_original[negative_mask][:, 0:2]
    all_edges = np.concatenate([positive_edges, negative_edges], axis=0)

    protein_dict = {}
    drug_to_smiles = {}
    for drug_id, protein_id, _, smiles, sequence in interactions:
        protein_dict.setdefault(protein_id, sequence)
        drug_to_smiles.setdefault(drug_id, smiles)

    compound_smiles = [drug_to_smiles[drug_id] for drug_id in drug_ids]
    drug_loader = None
    protein_loader = None
    if build_graph_loaders:
        drug_graphs = [smiles_to_graph(smiles) for smiles in compound_smiles]
        data_root = os.environ.get('MEDDC_DTI_DATA_DIR', 'data')
        esm2_dir = os.path.join(data_root, dataset_name.lower(), 'esm2')
        target_graphs = {
            protein_id: build_esm2_contact_graph(protein_id, protein_dict[protein_id], esm2_dir, config.esm2_contact_thresh)
            for protein_id in protein_ids
        }
        drug_loader = Data.DataLoader(GraphDataset(drug_graphs), collate_fn=collate_graphs, batch_size=nb_drugs, shuffle=False)
        protein_loader = Data.DataLoader(ProteinGraphDataset(protein_ids, target_graphs), collate_fn=collate_graphs, batch_size=min(nb_proteins, 512), shuffle=False)

    train_edges, test_edges = split_interaction_edges(all_edges, nb_drugs, nb_proteins, setting, split_seed)
    if dataset_name == 'DrugBank' and config.drugbank_unobserved_neg_ratio > 0:
        observed_edges = all_pairs[:, 0:2].copy()
        observed_edges[:, 1] -= nb_drugs
        excluded_drugs, excluded_proteins, excluded_pairs = split_exclusion_sets(nb_drugs, nb_proteins, setting, split_seed)
        n_unobserved = int(max(1, positive_edges.shape[0]) * config.drugbank_unobserved_neg_ratio)
        sampled_negatives = sample_unobserved_negative_edges(
            nb_drugs,
            nb_proteins,
            observed_edges,
            n_unobserved,
            config.drugbank_unobserved_neg_seed + int(setting),
            excluded_drugs,
            excluded_proteins,
            excluded_pairs,
        )
        if sampled_negatives.size > 0:
            train_edges = np.concatenate([train_edges, sampled_negatives], axis=0)

    train_mask = coo_matrix((np.ones(train_edges.shape[0], dtype=bool), (train_edges[:, 0], train_edges[:, 1])), shape=(nb_drugs, nb_proteins)).toarray()
    test_mask = coo_matrix((np.ones(test_edges.shape[0], dtype=bool), (test_edges[:, 0], test_edges[:, 1])), shape=(nb_drugs, nb_proteins)).toarray()
    train_mask = torch.from_numpy(train_mask).view(-1)
    test_mask = torch.from_numpy(test_mask).view(-1)

    positive_label_edges = all_pairs[all_pairs[:, 2] == 1, 0:2].copy()
    positive_label_edges[:, 1] -= nb_drugs
    label_matrix = coo_matrix((np.ones(positive_label_edges.shape[0]), (positive_label_edges[:, 0], positive_label_edges[:, 1])), shape=(nb_drugs, nb_proteins)).toarray()
    label_matrix = np.clip(label_matrix, 0, 1)
    label_vector = torch.from_numpy(label_matrix).float().view(-1)
    train_edge_labels = label_matrix[train_edges[:, 0], train_edges[:, 1]].astype(np.float32)
    test_edge_labels = label_matrix[test_edges[:, 0], test_edges[:, 1]].astype(np.float32)

    positive_train_edges = all_pairs[all_pairs[:, 2] == 1, 0:2]
    graph_edges = np.vstack((positive_train_edges, positive_train_edges[:, [1, 0]]))
    positive_adjacency = torch.zeros((nb_drugs + nb_proteins, nb_drugs + nb_proteins))
    for drug_node, protein_node in graph_edges:
        positive_adjacency[int(drug_node)][int(protein_node)] = 1
    if dataset_name in ('KIBA', 'DrugBank'):
        topology_adjacency = positive_adjacency.numpy().astype(np.float32)
    else:
        common_neighbor_graph = positive_common_neighbor_graph(nb_drugs + nb_proteins, positive_adjacency, common_neighbor_threshold=5)
        topology_adjacency = (positive_adjacency + common_neighbor_graph).numpy().astype(np.float32)

    drug_prior = compute_drug_morgan_prior(compound_smiles)
    drug_aux_features = compute_drug_auxiliary_features(compound_smiles)
    ddi_adjacency_base = build_drug_drug_edges(drug_prior[:, 8:], top_k=5, threshold=0.5)
    ddi_adjacency = ddi_adjacency_base * 0.1
    if config.use_drug_aux_similarity_graph:
        aux_ddi_adjacency = build_drug_auxiliary_edges(
            drug_aux_features,
            top_k=config.drug_aux_similarity_top_k,
            threshold=config.drug_aux_similarity_threshold,
        )
        ddi_adjacency = ddi_adjacency + aux_ddi_adjacency * config.drug_aux_similarity_weight
    drug_embeddings = load_cached_embeddings(dataset_name, config.chemberta2_cache_names, drug_ids) if config.use_cached_drug_embeddings else None
    if drug_embeddings is not None and drug_embeddings.shape[0] == nb_drugs:
        ddi_adjacency = ddi_adjacency + build_drug_auxiliary_edges(
            drug_embeddings,
            top_k=config.drug_embedding_top_k,
            threshold=config.drug_embedding_threshold,
        ) * config.drug_embedding_graph_weight
    protein_sequences = [protein_dict[protein_id] for protein_id in protein_ids]
    ppi_adjacency_base = build_protein_protein_edges(protein_sequences, k=3, top_k=5, threshold=0.3)
    ppi_adjacency = ppi_adjacency_base * 0.1
    protein_embeddings = load_cached_embeddings(dataset_name, config.esm2_cache_names, protein_ids) if config.use_cached_protein_embeddings else None
    if protein_embeddings is not None and protein_embeddings.shape[0] == nb_proteins:
        ppi_adjacency = ppi_adjacency + build_drug_auxiliary_edges(
            protein_embeddings,
            top_k=config.protein_embedding_top_k,
            threshold=config.protein_embedding_threshold,
        ) * config.protein_embedding_graph_weight
    topology_adjacency[:nb_drugs, :nb_drugs] += ddi_adjacency
    topology_adjacency[nb_drugs:, nb_drugs:] += ppi_adjacency
    topology_adjacency = normalize_adjacency(topology_adjacency + np.eye(topology_adjacency.shape[0]))
    topology_adjacency = dense_to_sparse_tensor(topology_adjacency)

    sim_adj_raw = np.zeros((nb_drugs + nb_proteins, nb_drugs + nb_proteins), dtype=np.float32)
    sim_adj_raw[:nb_drugs, :nb_drugs] = ddi_adjacency_base
    sim_adj_raw[nb_drugs:, nb_drugs:] = ppi_adjacency_base
    sim_adj_normalized = normalize_adjacency(sim_adj_raw + np.eye(sim_adj_raw.shape[0]))
    similarity_adjacency = dense_to_sparse_tensor(sim_adj_normalized)

    dti_adjacency = np.zeros((nb_drugs + nb_proteins, nb_drugs + nb_proteins), dtype=np.float32)
    for drug_node, protein_node in graph_edges:
        dti_adjacency[int(drug_node)][int(protein_node)] = 1.0
    dti_adjacency = normalize_adjacency(dti_adjacency + np.eye(dti_adjacency.shape[0]))
    dti_adjacency = torch.FloatTensor(dti_adjacency)
    biochemical_sources = {'rdkit_morgan_maccs': True, 'chemberta2_cached': False, 'esm2_cached': False, 'protein_sequence_prior': True}
    drug_prior_blocks = [drug_aux_features, drug_prior]
    if drug_embeddings is not None and drug_embeddings.shape[0] == nb_drugs:
        drug_prior_blocks.append(drug_embeddings.astype(np.float32))
        biochemical_sources['chemberta2_cached'] = True
    protein_prior_blocks = [compute_protein_sequence_prior(protein_sequences)]
    if protein_embeddings is not None and protein_embeddings.shape[0] == nb_proteins:
        protein_prior_blocks.append(protein_embeddings.astype(np.float32))
        biochemical_sources['esm2_cached'] = True
    protein_sequence_prior = protein_prior_blocks[0]
    drug_biochemical_prior = project_prior_features(np.concatenate(drug_prior_blocks, axis=1), config.n_input, config.biochemical_prior_projection_seed)
    protein_biochemical_prior = project_prior_features(np.concatenate(protein_prior_blocks, axis=1), config.n_input, config.biochemical_prior_projection_seed + 1)
    biochemical_prior_features = np.concatenate([drug_biochemical_prior, protein_biochemical_prior], axis=0)
    source_prior_features = {}
    if config.use_rdkit_prior:
        source = np.zeros((nb_drugs + nb_proteins, drug_prior.shape[1]), dtype=np.float32)
        source[:nb_drugs] = standardize_prior_features(drug_prior)
        source_prior_features['rdkit'] = torch.from_numpy(source).float()
    if config.use_chemberta2_prior and drug_embeddings is not None and drug_embeddings.shape[0] == nb_drugs:
        source = np.zeros((nb_drugs + nb_proteins, drug_embeddings.shape[1]), dtype=np.float32)
        source[:nb_drugs] = standardize_prior_features(drug_embeddings)
        source_prior_features['chemberta2'] = torch.from_numpy(source).float()
    if config.use_esm2_prior and protein_embeddings is not None and protein_embeddings.shape[0] == nb_proteins:
        source = np.zeros((nb_drugs + nb_proteins, protein_embeddings.shape[1]), dtype=np.float32)
        source[nb_drugs:] = standardize_prior_features(protein_embeddings)
        source_prior_features['esm2'] = torch.from_numpy(source).float()
    if config.use_protein_sequence_prior:
        source = np.zeros((nb_drugs + nb_proteins, protein_sequence_prior.shape[1]), dtype=np.float32)
        source[nb_drugs:] = standardize_prior_features(protein_sequence_prior)
        source_prior_features['protein_sequence'] = torch.from_numpy(source).float()
    source_dims = {name: int(value.size(1)) for name, value in source_prior_features.items()}
    return {
        'drug_loader': drug_loader,
        'protein_loader': protein_loader,
        'adjacency': topology_adjacency,
        'similarity_adjacency': similarity_adjacency,
        'dti_adjacency': dti_adjacency,
        'labels': label_vector,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'edge': graph_edges,
        'nb_drugs': nb_drugs,
        'nb_proteins': nb_proteins,
        'nb_all': nb_drugs + nb_proteins,
        'drug_aux_features': torch.from_numpy(drug_aux_features).float(),
        'biochemical_prior_features': torch.from_numpy(biochemical_prior_features).float(),
        'biochemical_source_features': source_prior_features,
        'biochemical_source_dims': source_dims,
        'biochemical_prior_sources': biochemical_sources,
        'biochemical_prior_audit': load_biochemical_prior_audit(dataset_name),
        'train_edges': torch.from_numpy(train_edges[:, 0:2]).long(),
        'test_edges': torch.from_numpy(test_edges[:, 0:2]).long(),
        'train_edge_labels': torch.from_numpy(train_edge_labels).float(),
        'test_edge_labels': torch.from_numpy(test_edge_labels).float(),
    }


def build_davis_s2_dataset(interactions, nb_drugs, nb_proteins, split_seed=3):
    return build_interaction_dataset(interactions, nb_drugs, nb_proteins, dataset_name='Davis', setting=2, split_seed=split_seed)
