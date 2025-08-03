import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch_geometric


class GraphDataset(torch.utils.data.Dataset):


    def __init__(self, embeddings, domains, targets):
        self.embeddings = embeddings
        self.domains = domains
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        target = self.targets[idx]
        domain = self.domains[idx]

        x = embedding.view(-1, embedding.size(-1))
        num_nodes = x.size(0)

        edge_index = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])

        for i in range(num_nodes):
            edge_index.append([i, i])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index, y=target), domain


def custom_collate_fn(batch):

    graph_data_list = [item[0] for item in batch]
    domain_list = [item[1] for item in batch]

    batch_graph = torch_geometric.data.Batch.from_data_list(graph_data_list)

    batch_domain = torch.stack(domain_list, dim=0)

    return batch_graph, batch_domain



class DL_GATv2(nn.Module):


    def __init__(self, input_dim, domain_dim, hidden_dim, feat_dropout,
                 attn_dropout, fnn_dropout, num_heads):
        super(DL_GATv2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.feat_dropout = nn.Dropout(feat_dropout)

        self.gatv2_1 = GATv2Conv(
            input_dim, hidden_dim, heads=num_heads,
            dropout=attn_dropout, add_self_loops=False
        )
        self.gatv2_2 = GATv2Conv(
            hidden_dim * num_heads, hidden_dim, heads=num_heads,
            dropout=attn_dropout, add_self_loops=False
        )

        self.domain_mlp = nn.Sequential(
            nn.Linear(domain_dim, 64),
            nn.ReLU(),
            nn.Dropout(fnn_dropout)
        )

        self.residual_transform = nn.Linear(input_dim, hidden_dim * num_heads)

        self.graph_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads),
            nn.ReLU(),
            nn.Dropout(fnn_dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_heads + 64, 128),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(fnn_dropout),
            nn.Linear(64, 1)
        )

        self.relu = nn.ReLU()

    def forward(self, data, domain):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.size(0) > 0:
            residual = self.residual_transform(x)
            x = self.gatv2_1(x, edge_index)
            x = self.relu(x)
            x = self.feat_dropout(x)
            x += residual

            residual = x
            x = self.gatv2_2(x, edge_index)
            x = self.relu(x)
            x = self.feat_dropout(x)
            x += residual

            x = global_mean_pool(x, batch)
        else:
            batch_size = domain.size(0)
            x = torch.zeros((batch_size, self.hidden_dim * self.num_heads), device=x.device)

        x = self.graph_fc(x)

        domain_processed = self.domain_mlp(domain)

        x_cat = torch.cat([x, domain_processed], dim=1)
        return self.fc(x_cat).squeeze()
