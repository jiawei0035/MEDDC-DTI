import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch, InMemoryDataset


class GraphDataset(InMemoryDataset):
    def __init__(self, graphs):
        super().__init__(root='.')
        self.graphs = []
        for graph in graphs:
            features, edge_index = graph[1], graph[2]
            self.graphs.append(DATA.Data(x=torch.tensor(features, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).t()))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, target_keys, target_graphs):
        super().__init__(root='.')
        self.graphs = []
        for key in target_keys:
            _, features, edge_index = target_graphs[key]
            self.graphs.append(DATA.Data(x=torch.tensor(features, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).t()))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def collate_graphs(data_list):
    return Batch.from_data_list(data_list)
