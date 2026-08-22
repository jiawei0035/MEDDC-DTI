import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool as gmp, global_mean_pool as gep


class MolecularProteinGraphEncoder(nn.Module):
    def __init__(self, num_features_pro=1280, num_features_mol=78, output_dim=256, dropout=0.2, gnn_pool='mean'):
        super().__init__()
        self.gnn_pool = gnn_pool
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=4, concat=False, dropout=0.1)
        self.mol_conv2 = GATConv(num_features_mol, num_features_mol * 2, heads=4, concat=False, dropout=0.1)
        self.mol_conv3 = GATConv(num_features_mol * 2, num_features_mol * 4, heads=4, concat=False, dropout=0.1)
        mol_pool_dim = num_features_mol * 4 * (2 if gnn_pool == 'mean_max' else 1)
        self.mol_fc_g1 = nn.Linear(mol_pool_dim, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        protein_gnn_dim = 128 if num_features_pro > 256 else num_features_pro
        self.pro_proj = nn.Sequential(nn.Linear(num_features_pro, protein_gnn_dim), nn.LayerNorm(protein_gnn_dim), nn.ReLU()) if num_features_pro > 256 else None
        self.pro_conv1 = GATConv(protein_gnn_dim, protein_gnn_dim, heads=4, concat=False, dropout=0.1)
        self.pro_conv2 = GATConv(protein_gnn_dim, protein_gnn_dim * 2, heads=4, concat=False, dropout=0.1)
        self.pro_conv3 = GATConv(protein_gnn_dim * 2, protein_gnn_dim * 4, heads=4, concat=False, dropout=0.1)
        pro_pool_dim = protein_gnn_dim * 4 * (2 if gnn_pool == 'mean_max' else 1)
        self.pro_fc_g1 = nn.Linear(pro_pool_dim, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _pool(self, node_embeddings, batch):
        if self.gnn_pool == 'mean_max':
            return torch.cat([gep(node_embeddings, batch), gmp(node_embeddings, batch)], dim=-1)
        if self.gnn_pool == 'max':
            return gmp(node_embeddings, batch)
        return gep(node_embeddings, batch)

    def mol_forward(self, mol_x, mol_edge_index, mol_batch):
        x = self.relu(self.mol_conv1(mol_x, mol_edge_index))
        x = self.relu(self.mol_conv2(x, mol_edge_index))
        x = self.relu(self.mol_conv3(x, mol_edge_index))
        x = self._pool(x, mol_batch)
        x = self.dropout(self.relu(self.mol_fc_g1(x)))
        return self.dropout(self.mol_fc_g2(x))

    def pro_forward(self, protein_x, protein_edge_index, protein_batch):
        if self.pro_proj is not None:
            protein_x = self.pro_proj(protein_x)
        x = self.relu(self.pro_conv1(protein_x, protein_edge_index))
        x = self.relu(self.pro_conv2(x, protein_edge_index))
        x = self.relu(self.pro_conv3(x, protein_edge_index))
        x = self._pool(x, protein_batch)
        x = self.dropout(self.relu(self.pro_fc_g1(x)))
        return self.dropout(self.pro_fc_g2(x))


GNNNet = MolecularProteinGraphEncoder
