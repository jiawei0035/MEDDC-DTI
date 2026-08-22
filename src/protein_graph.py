import os
import numpy as np
import torch


def build_esm2_contact_graph(target_key, target_sequence, esm2_dir, contact_thresh=0.2):
    esm2_file = os.path.join(esm2_dir, target_key + '.pt')
    if not os.path.exists(esm2_file):
        raise FileNotFoundError(f'ESM2 residue graph file not found: {esm2_file}')
    data = torch.load(esm2_file, map_location='cpu', weights_only=False)
    contact_map = data['contact_map'].float().numpy()
    residue_embedding = data['residue_emb'].float().numpy()
    target_size = len(target_sequence)
    contact_map = contact_map + np.eye(contact_map.shape[0])
    row_index, col_index = np.where(contact_map >= contact_thresh)
    edge_index = np.array([[i, j] for i, j in zip(row_index, col_index)], dtype=np.int64)
    if residue_embedding.shape[0] != target_size:
        aligned = np.zeros((target_size, residue_embedding.shape[1]), dtype=np.float32)
        copy_len = min(residue_embedding.shape[0], target_size)
        aligned[:copy_len] = residue_embedding[:copy_len]
        residue_embedding = aligned
    return target_size, residue_embedding.astype(np.float32), edge_index
