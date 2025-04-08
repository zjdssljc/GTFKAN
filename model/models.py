# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from .layers import TransformerEncoderLayer
from einops import repeat
import numpy as np

'''
class SATdecoder(nn.Module):
    def __init__(self, microbe_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(SATdecoder, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.drug_num = Drug_num
        self.microbe_num = microbe_num
        self.decode = nn.ModuleList([
            nn.Linear(Nodefeat_size if l == 0 else Nodefeat_size, Nodefeat_size) for l in
            range(nlayers)])

        self.linear = nn.Linear(Nodefeat_size, 1)
        self.drug_linear = nn.Linear(Nodefeat_size, Nodefeat_size)
        self.microbe_linear = nn.Linear(Nodefeat_size, Nodefeat_size)

    def forward(self, nodes_features, drug_index, microbe_index):

        microbe_features = nodes_features[microbe_index]
        drug_features = nodes_features[drug_index]

        microbe_features = self.microbe_linear(microbe_features)
        drug_features = self.drug_linear(drug_features)
        pair_nodes_features = drug_features*microbe_features
        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))

        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)
'''


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index_=None,
            subgraph_edge_attr=None, subgraph_indicatorindex=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
               # subgraph_edge_index=subgraph_edge_index,
               # subgraph_indicator_index=subgraph_indicator_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )

        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=2,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=1,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):   #1373+173
                self.embedding = nn.Embedding(in_size, d_model) 
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,  ##################     d_model
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 1546)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se


        encoder_layer = TransformerEncoderLayer(   #d_model隐藏层的维度
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        #self.SATdecoder = SATdecoder(173,1373,d_model,d_model,1,dropout)


        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.fc = nn.Sequential(
            nn.Linear(d_model, 1546, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),   #dmodel=128
                nn.ReLU(True),
                nn.Linear(d_model, num_class)   #
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))


    def forward(self, data,return_attn=False):
        x = data
        adj = np.loadtxt("./data/MDAD/adj21.txt")
        ptr = np.loadtxt('./data/MDAD/ptr.txt')
        adj = torch.tensor(adj, dtype=torch.int64)
        ptr = torch.tensor(ptr, dtype=torch.int64)
        edge_index = adj
        batch = []  #索引
        for i in range(0,1546):
            batch.append(i)
        batch = torch.tensor(batch,dtype=torch.int64)

        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))

        edge_attr = None
        subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output, 
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr, 
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            #subgraph_edge_index=subgraph_edge_index,
            #subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, batch)  #######batch只有0和31
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list

        #output = self.SATdecoder(output, adj[0], adj[1])
        #output = output.view(-1)

        #output(1546,128)
        return self.fc(output)
        #return self.classifier(output)