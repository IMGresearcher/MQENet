from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, LayerNorm, LeakyReLU
from torch_geometric.nn import GATv2Conv,SAGPooling, \
    JumpingKnowledge, global_mean_pool, global_max_pool


class DeepGCNLayer(torch.nn.Module):

    def __init__(self, conv=None, norm=None, act=None, block='res+',
                 dropout=0.5, ckpt_grad=False):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index):

        if self.block == 'res+':
            h = x
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv(h, edge_index)

            return x + h

        else:
            h = self.conv(x, edge_index)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'


class DeeperGCN(torch.nn.Module):
    def __init__(self, input_channels, num_layers, hidden_channels, num_classes, ratio=.7, dropout=0., conv_dropout=0.):
        super(DeeperGCN, self).__init__()
        self.dropout = dropout
        self.node_encoder = Linear(input_channels, hidden_channels).to("cuda:0")
        self.DeepGCNs = torch.nn.ModuleList().to("cuda:0")
        self.Pool = torch.nn.ModuleList().to("cuda:0")
        self.ratio = ratio
        for i in range(1, num_layers + 1):

            conv = GATv2Conv(hidden_channels, hidden_channels, dropout=conv_dropout).jittable()

            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = LeakyReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=conv_dropout,
                                 ckpt_grad=False)
            self.DeepGCNs.append(layer).to("cuda:0")
            if i != num_layers:
                pool = SAGPooling(hidden_channels, ratio=self.ratio)
                pool.gnn = pool.gnn.jittable()
                self.Pool.append(pool).to("cuda:0")


        self.jump = JumpingKnowledge(mode='cat').to("cuda:0")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * num_layers * 2, hidden_channels * num_layers // 2),
            nn.BatchNorm1d(hidden_channels * num_layers // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels * num_layers // 2, 3),
            nn.BatchNorm1d(3),
            nn.LeakyReLU(),
            nn.Linear(3, num_classes),
            #nn.Softmax()
            nn.Sigmoid()
        ).to("cuda:0")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch


        x = self.node_encoder(x)
        x = self.DeepGCNs[0].conv(x, edge_index)


        xs = [torch.cat([global_mean_pool(x, batch),
                         global_max_pool(x, batch)], dim=1)]

        for conv, pool in zip(self.DeepGCNs[1:], self.Pool):
            x = conv(x, edge_index)
            x, edge_index, _, batch, *_ = pool(x=x, edge_index=edge_index, batch=batch)
            xs += [torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=1)]

        x = self.jump(xs)
        output = self.mlp(x)
        output = output.view(-1)
        return output


GraphData = namedtuple('GraphData', ['x', 'edge_index', 'batch'])


class TraceModel(DeeperGCN):
    def forward(self, x, edge_index, batch):
        data = GraphData(x, edge_index, batch)
        return super().forward(data)
