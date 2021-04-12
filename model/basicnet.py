import torch
from torch import nn
import torch.nn.functional as F

import cv2
import os
import h5py, math
import numpy as np


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ConcatNet(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d):
        super(ConcatNet, self).__init__()

        self.w = 60
        self.h = 60

        c1, c2, c3, c4 = 256, 512, 1024, 2048

        self.conv1 = nn.Sequential(BasicConv2d(c1, c1, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c1, c1, BatchNorm, kernel_size=1, padding=0))
        self.conv2 = nn.Sequential(BasicConv2d(c2, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))
        self.conv3 = nn.Sequential(BasicConv2d(c3, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))
        self.conv4 = nn.Sequential(BasicConv2d(c4, c2, BatchNorm, kernel_size=3, padding=1), BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))

        # (256+512+1024+2048=3840)
        c = c1 + c2 + c2 + c2
        self.conv5 = nn.Sequential(BasicConv2d(c, c2, BatchNorm, kernel_size=3, padding=1),
                                   BasicConv2d(c2, c2, BatchNorm, kernel_size=1, padding=0))

    def forward(self, x1, x2, x3, x4):
        x1 = F.interpolate(x1, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)

        x2 = F.interpolate(x2, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)

        x3 = F.interpolate(x3, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)

        x4 = F.interpolate(x4, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x4 = self.conv4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1) # c=256 x 4 = 1024
        x = self.conv5(x)

        return x

 
class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous() # b x k x c
        support = torch.matmul(x_t, self.weight) # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous() # b x c x k
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous() # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x)) # b x c x k
        x = self.relu(x)
        return x

#nips-18
class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) #across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

class MutualModule0(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule0, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    #graph0: edge, graph1/2: region, assign:edge
    def forward(self, edge_graph, region_graph1, region_graph2, assign):
        m = self.corr_matrix(edge_graph, region_graph1, region_graph2)
        edge_graph = edge_graph + m

        edge_graph = self.gcn(edge_graph)
        edge_x = edge_graph.bmm(assign) # reprojection
        edge_x = self.conv(edge_x.unsqueeze(3)).squeeze(3)
        return edge_x

    def corr_matrix(self, edge, region1, region2):
        assign = edge.permute(0, 2, 1).contiguous().bmm(region1)
        assign = F.softmax(assign, dim=-1) #normalize region-node
        m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m

class ECGraphNet(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(ECGraphNet, self).__init__()
        self.dim = dim
        self.conv0 =  nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.node_num = 32
        self.proj0 = GraphNet(self.node_num, self.dim, False)
        self.conv1 = nn.Sequential(BasicConv2d(2*self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
 
    def forward(self, x, edge):
        b, c, h, w = x.shape
        device = x.device

        '''
        _, _, h1, w1 = edge.shape
        if h1 != h or w1 != w:
            edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=True)
        '''

        x1 = torch.sigmoid(edge).mul(x) #elementwise-mutiply
        x1 = self.conv0(x1)
        nodes, _ = self.proj0(x1) # b x c x k or b x k x c

        residual_x = x.view(b, c, -1).permute(2, 0, 1)[:, None] - nodes.permute(2, 0, 1)
        residual_x = residual_x.permute(2, 3, 1, 0).view(b, c, self.node_num, h, w).contiguous()

        '''
        residual_x = torch.zeros([b, c, self.node_num, h, w], device=device, dtype=x.dtype, layout=x.layout)
        for i in range(self.node_num):
            residual_x[:, :, i:i+1, :, :] = (x - nodes[:, :, i:i+1].unsqueeze(3)).unsqueeze(2)
        '''

        dists = torch.norm(residual_x, dim=1, p=2) # b x k x h x w

        k = 5
        _, idx = torch.topk(dists, k=k, dim=1, largest=False) # b x 5 x h x w

        num_points = h * w
        idx_base = torch.arange(0, b, device=device).view(-1, 1, 1, 1)*self.node_num
        idx = (idx + idx_base).view(-1)

        nodes = nodes.transpose(2, 1).contiguous()
        x1 = nodes.view(b*self.node_num, -1)[idx, :]
        x1 = x1.view(b, num_points, k, c)

        x2 = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(2).repeat(1, 1, k, 1)

        x1 = torch.cat((x1-x2, x2), dim=3).permute(0, 3, 1, 2).contiguous() # b x n x 5 x (2c)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x = x + x1.view(b, c, h, w)

        return x

class MutualModule1(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule1, self).__init__()
        self.dim = dim

        self.gcn = CascadeGCNet(dim, loop=3)

        self.pred0 = nn.Conv2d(self.dim, 1, kernel_size=1) # predicted edge is used for edge-region mutual sub-module

        self.pred1_ = nn.Conv2d(self.dim, 1, kernel_size=1) # region prediction

        # conv region feature afger reproj
        self.conv0 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        self.ecg = ECGraphNet(self.dim, BatchNorm, dropout)

    def forward(self, region_x, region_graph, assign, edge_x):
        b, c, h, w = edge_x.shape

        edge = self.pred0(edge_x)
       
        region_graph = self.gcn(region_graph)
        n_region_x = region_graph.bmm(assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))

        region_x = region_x + n_region_x # raw-feature with residual

        region_x = region_x + edge_x
        region_x = self.conv1(region_x) 

        # enhance
        region_x = self.ecg(region_x, edge)

        region = self.pred1_(region_x)

        return region_x, edge, region


class MutualNet(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=512, num_clusters=8, dropout=0.1):
        super(MutualNet, self).__init__()

        self.dim = dim

        self.edge_proj0   = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.region_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.edge_conv = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
                                       #BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=3, padding=1)
        self.edge_conv[0].reset_params()

        self.region_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv1[0].reset_params()

        self.region_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv2[0].reset_params()

        self.r2e = MutualModule0(self.dim, BatchNorm, dropout) 
        self.e2r = MutualModule1(self.dim, BatchNorm, dropout)

    def forward(self, edge_x, region_x):
        # project region/edge fature to graph
        region_graph, region_assign = self.region_proj0(region_x)
        edge_graph, edge_assign = self.edge_proj0(edge_x)

        edge_graph = self.edge_conv(edge_graph.unsqueeze(3)).squeeze(3)

        # region-edge mutual learning
        region_graph1 = self.region_conv1(region_graph.unsqueeze(3)).squeeze(3)
        region_graph2 = self.region_conv2(region_graph.unsqueeze(3)).squeeze(3)

        # CGI
        n_edge_x = self.r2e(edge_graph, region_graph1, region_graph2, edge_assign)
        edge_x = edge_x + n_edge_x.view(edge_x.size()).contiguous()

        # edge-region mutual learning
        region_x, edge, region = self.e2r(region_x, region_graph, region_assign, edge_x)

        return edge_x, edge, region_x, region

