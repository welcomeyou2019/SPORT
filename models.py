import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_max_pool, GlobalAttention
import numpy as np
from conv import GNN_node
from typing import Optional
from torch_scatter import scatter

def subgraph_pool(h_node, batched_data, pool, dim=0):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs#16
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    # print(torch.cumsum(num_subgraphs, dim=0))
    # print(tmp,tmp.shape)
    # print(batched_data)
    graph_offset = tmp[batched_data.batch]
    # print(graph_offset)
    # print(batched_data.subgraph_batch)

    # subgraph_idx = graph_offset + batched_data.subgraph_batch
    subgraph_idx = batched_data.subgraph_batch
    # print(subgraph_idx)
    h_graph = pool(h_node, subgraph_idx, dim=dim)
    # print(h_graph)
    # print(h_graph.shape)
    # print('subgraph idx',subgraph_idx)
    # # print(h_graph.shape)
    # a = list(subgraph_idx.cpu().numpy())
    # # print(torch.from_numpy(np.array(list(set(a)))).long())
    # print(a)
    # print(set(a))
    # print(np.array(set(a)))

    # return h_graph[torch.from_numpy(np.array(list(set(a)))).long().to(h_graph.device)]
    return h_graph

def global_mean_pool(x, batch, dim=0, size: Optional[int] = None):
    size = int(batch.max().item() + 1) if size is None else size
    # size = len(set(list(batch.numpy()))) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='mean')

def global_add_pool(x, batch, dim=0, size: Optional[int] = None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='add')

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = self.emb_dim if self.JK == 'last' else self.emb_dim * self.num_layer + in_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual,
                                 gnn_type=gnn_type, num_random_features=num_random_features,
                                 feature_encoder=feature_encoder)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        # print(batched_data)
        # print(np.array(batched_data.subgraph_batch))
        # print(set(np.array(batched_data.subgraph_batch)),len(set(np.array(batched_data.subgraph_batch))))
        h_node = self.gnn_node(batched_data)
        # print(h_node.shape)

        return subgraph_pool(h_node, batched_data, self.pool)


class GNNComplete(GNN):
    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', num_random_features=0, residual=False, drop_ratio=0.5, JK="last", graph_pooling="mean",
                 feature_encoder=lambda x: x):

        super(GNNComplete, self).__init__(num_tasks, num_layer, in_dim, emb_dim, gnn_type, num_random_features,
                                          residual, drop_ratio, JK, graph_pooling, feature_encoder)

        if gnn_type == 'graphconv':
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim, out_features=self.out_dim // 2),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim // 2, out_features=num_tasks)
            )
        else:
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim, out_features=num_tasks),
            )

    def forward(self, batched_data):
        subgraph_list = []
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        # print(h_node.shape,h_graph.shape)
        h_subgraph = subgraph_pool(h_node, batched_data, global_mean_pool)
        # print(batched_data.num_subgraphs)
        # print(h_subgraph.shape)
        subgraph_index = torch.cumsum(batched_data.num_subgraphs, dim=0)
        # print(subgraph_index)

        for i in range(len(subgraph_index)):
            if i == 0:
                subgraph_list.append(self.final_layers(h_subgraph[:subgraph_index[i]]))
            else:
                subgraph_list.append(self.final_layers(h_subgraph[subgraph_index[i - 1]:subgraph_index[i]]))

        return self.final_layers(h_graph), subgraph_list


class DSnetwork(torch.nn.Module):
    def __init__(self, subgraph_gnn, channels, num_tasks, invariant):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant

        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                           out_features=channels[i]))
            if self.invariant:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i],
                                                   out_features=channels[i]))
            else:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn.out_dim,
                                                   out_features=channels[i]))

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=channels[-1], out_features=2 * channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * channels[-1], out_features=num_tasks)
        )

    def forward(self, batched_data):
        # print(batched_data)
        # print(self.subgraph_gnn)
        h_subgraph = self.subgraph_gnn(batched_data)
        subgraph_list = []
        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)
                # print(fc_sum)
                # print(h_subgraph.shape)
                # print(batched_data.subgraph_idx_batch.shape)
                x2 = fc_sum(
                    torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx_batch])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
            subgraph_index = torch.cumsum(batched_data.num_subgraphs, dim=0)
            # print(subgraph_index)

            for i in range(len(subgraph_index)):
                if i == 0:
                    subgraph_list.append(F.softmax(self.final_layers(h_subgraph[:subgraph_index[i]])))
                else:
                    subgraph_list.append(F.softmax(self.final_layers(h_subgraph[subgraph_index[i - 1]:subgraph_index[i]])))

        return self.final_layers(h_graph), subgraph_list


class DSSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv, sample):
        super(DSSnetwork, self).__init__()

        self.emb_dim = emb_dim
        self.sample = sample

        self.feature_encoder = feature_encoder

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim if i != 0 else in_dim, emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )



    def forward(self, batched_data):
        # print(batched_data)
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # x = self.feature_encoder(x)
        out_preturb = None
        subgraph_list = []
        for i in range(len(self.gnn_list)):
            gnn, bn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.gnn_sum_list[i], self.bn_sum_list[i]

            h1 = bn(gnn(x, edge_index, edge_attr))

            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            # num_nodes_per_subgraph = batched_data.num_nodes
            # print(num_nodes_per_subgraph)
            # print(torch.cumsum(num_nodes_per_subgraph, dim=0))
            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                             torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx
            # print('offset',graph_offset.shape)
            #
            # print(x.shape, node_idx.shape)
            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            h2 = bn_sum(gnn_sum(x_sum, batched_data.original_edge_index,
                                batched_data.original_edge_attr if edge_attr is not None else edge_attr))

            x = F.relu(h1 + h2[node_idx])
        # print('x:',x.shape)
        # print(batched_data)
        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # print(batched_data.num_subgraphs)
        # print(h_subgraph.shape)
        # print(batched_data.num_subgraphs)
        subgraph_index = torch.cumsum(batched_data.num_subgraphs, dim=0)

        # num_subgraphs = batched_data.num_subgraphs
        # tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
        #                  subgraph_index])
        # graph_offset = tmp[batched_data.batch]
        # subgraph_idx = graph_offset + batched_data.subgraph_idx_batch
        # print(subgraph_idx)


        for i in range(len(subgraph_index)):
            if i == 0:
                subgraph_list.append(self.final_layers(h_subgraph[:subgraph_index[i]]))
            else:
                subgraph_list.append(self.final_layers(h_subgraph[subgraph_index[i-1]:subgraph_index[i]]))

        subgraph_list_sec = []
        if self.sample != None:
            for subgraph in subgraph_list:
                if subgraph.shape[0] <= self.sample:
                    subgraph_list_sec.append(subgraph)
                else:
                    subgraph_list_sec.append(subgraph[:self.sample])
            subgraph_list = subgraph_list_sec

        h_subgraph = h_subgraph[h_subgraph.sum(1)!=0,:]
        # aggregate to obtain a representation of the graph given the representations of the subgraphs
        # print('h_subgraph',h_subgraph.shape, batched_data.subgraph_idx_batch.shape)
        # print(h_subgraph.shape)
        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        # h_graph = torch_scatter.scatter(src=h_subgraph, index=subgraph_idx, dim=0, reduce="mean")
        # print(h_graph.shape)
        # print(len(subgraph_list))
        return self.final_layers(h_graph), subgraph_list


class EgoEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
