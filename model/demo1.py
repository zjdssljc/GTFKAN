# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import time
import torch
from torch import nn, optim
import torch_geometric.utils as utils
from model.models import GraphTransformer
from model.utils import count_parameters
from model.position_encoding import POSENCODINGS
from model.gnn_layers import GNN_TYPES


def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on MDAD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="MDAD",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=1, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=256, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=10, help="number of epochs for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=128, help='edge features hidden dim')  ##
    parser.add_argument('--gnn-type', type=str, default='gcn',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=4, help="number of layers for GNNs default:2")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="gnn",
                        help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--not_extract_node_feature', action='store_true')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, A, criterion, optimizer, lr_scheduler, use_cuda=False):
    print('Graph Transformer')

    model.train()
    if args.abs_pe == 'lap':
        # sign flip as in Bresson et al. for laplacian PE
        sign_flip = torch.rand(A)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        A.abs_pe = A.abs_pe * sign_flip.unsqueeze(0)

    for epoch in range(args.epochs):
        time_epoch_start = time.time()
        out = model(A)

        attr_embedding = out.detach().numpy()
        loss = criterion(out, A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('==>>> epoch: {}, train loss: {:.6f},Time:{:.2f}'.format(epoch+1, loss, time.time() - time_epoch_start))
    return attr_embedding


def train(A):   #A构建的异构网络
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    A = torch.FloatTensor(A)
    emb = np.size(A, axis=1)  # emb:4638

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(A)

    if 'pna' in args.gnn_type or args.gnn_type == 'mpnn':
        deg = torch.cat([
            utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in A])
    else:
        deg = None
    print(deg)

    model = GraphTransformer(in_size=emb,
                             num_class=2,
                             d_model=args.dim_hidden,   #128
                             dim_feedforward=2 * args.dim_hidden,     #0.5 1 2
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             k_hop=args.k_hop,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=None,
                             edge_dim=args.edge_dim,
                             se=args.se,
                             deg=deg,
                             in_embed=False,
                             edge_embed=False,
                             global_pool=args.global_pool)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #设置学习率
    lr_steps = args.lr / (args.warmup * len(A))
    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(A)

    print("Training...")
    F = train_epoch(model, A,criterion, optimizer, warmup_lr_scheduler, args.use_cuda)
    return F
