from models.encoder import GCN_Encoder
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from utils.register import register
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, GCNConv, GCN2Conv
from torch_geometric.utils import add_self_loops, degree
import numpy as np

class TraceMLP(nn.Module):
    def __init__(self, trace_dim, L, hidden_dim, mvc_dim):
        super(TraceMLP, self).__init__()
        self.input_dim = trace_dim * L
        self.lin = Linear(self.input_dim, hidden_dim, bias=False)
        self.lin2 = Linear(hidden_dim, mvc_dim, bias=False)

    def forward(self, data):
        x = data.trace_all
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(-1, self.input_dim)

        x = self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.normalize(x, p=2, dim=1)
        data.mvc = x

        return data

class TrajectoryMLPMixer(nn.Module):
    def __init__(self, trace_dim, L, mvc_dim):

        super(TrajectoryMLPMixer, self).__init__()
        self.trace_dim = trace_dim
        self.L = L
        self.out_dim = mvc_dim
        
        # Trajectory mixing weights
        self.w_traj = nn.Parameter(torch.randn(1, trace_dim))
        # Channel mixing weights
        self.W_channel = nn.Parameter(torch.randn(self.trace_dim, self.out_dim))

    def forward(self, data):
        """
            T (torch.Tensor): Input tensor of shape (L, N, d).
            L is the number of hops, N is the batch size (or number of nodes), 
            and out is the mvc dimension.
        Returns:
            torch.Tensor: Output tensor of shape (N, out).
        """
        x = data.trace_all
        T_weighted = F.relu((self.w_traj * x).sum(dim=2))  # Element-wise multiplication and normalization L*N
        T_weighted = F.normalize(T_weighted,dim=0)
        T_weighted = T_weighted.unsqueeze(2)
        T_weighted = T_weighted * x                        # Element-wise multiplication
        T_mixed = T_weighted.sum(dim=0)                    # Summation over L dimension to get shape (N, d)
        T_out = torch.matmul(T_mixed, self.W_channel)      # Matrix multiplication for channel mixing
        data.mvc = T_out

        return data

class EdgeScoringNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(EdgeScoringNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        start_features = data.mvc[data.edge_index[:,~data.slow_edge_mask][0, :], :]
        end_features = data.mvc[data.edge_index[:,~data.slow_edge_mask][1, :], :]
        combined_features = torch.cat((start_features, end_features), dim=1)
        x = F.relu(self.fc1(combined_features))        
        data.edge_scores = self.fc2(x).squeeze()
        return data

@register.model_register
class ADDNODE_GNN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, activation="relu", dropout=0.5, norm='id', **kargs):
        super(ADDNODE_GNN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden_size, activation, dropout, use_bn)
        if kargs['mvc']=='mlp':
            print("mvc_encoder:mlp")
            self.trace_mlp = TraceMLP(hidden_size, layer_num, kargs['trace_hidden'], kargs['mvc_dim'])
        elif kargs['mvc']=='tmm':
            print("mvc_encoder:tmm")
            self.trace_mlp = TrajectoryMLPMixer(hidden_size, layer_num, kargs['mvc_dim'])   

        self.edge_scorer = EdgeScoringNet(kargs['mvc_dim'], kargs['mvc_hidden'])
        self.encoder = register.encoders[kargs['encoder']](input_dim, layer_num, hidden_size, hidden_size, activation, dropout, norm, kargs['last_activation'], kargs['alpha'], kargs['theta'])
        self.classifier = torch.nn.Linear(hidden_size, output_dim)
        # self.classifier = GCNConv(hidden_size, output_dim)
        self.linear_classifier = torch.nn.Linear(hidden_size*2, output_dim)
    
    def forward(self, x=None, edge_index=None, edge_weight=None, frozen=False, **kwargs):

        ##################################### ADDNODE ###############################################
        data = kwargs['data']
        data = self.trace_mlp(data)
        data = self.edge_scorer(data)
        active_edge = torch.nn.functional.gumbel_softmax(data.edge_scores, tau=0.5, hard=True, eps=1e-10, dim=-1)[:,0]
        '''
        y_soft = data.edge_scores.softmax(dim=-1)
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(data.edge_scores, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        active_edge = ret[:,0]
        '''
        num_edge = data.edge_index[:,~data.slow_edge_mask].size(1)
        
        train_mask = torch.ones(data.edge_index.size(1))
        train_mask[:num_edge] = active_edge
        for i in range(num_edge):
            train_mask[num_edge+i] = 1-train_mask[i]
            train_mask[2*num_edge+i] = 1-train_mask[i]
        train_mask = train_mask.bool()
        x,edge_index = data.x,data.edge_index[:,train_mask]
        #print('now active edges:',edge_index.size(1))

        ##################################### DOWNSTREAM ############################################        
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x,trace = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
            trace = trace.permute(1, 0, 2)
            trace_all = trace[~data.insert_node_mask]
            x = x[~data.insert_node_mask]
            trace_all = trace_all.permute(1, 0, 2)         
        # x = self.classifier(x, edge_index)
        x = self.classifier(x)
        return x,trace_all
    
    def forward_subgraph(self, x, edge_index, batch, root_n_id, edge_weight=None, **kwargs):
        # x = torch.rand(x.shape, device=x.device)
        # x = torch.ones(x.shape, device=x.device)
        x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.linear_classifier(x) # use linear classifier
        # x = x[root_n_id]
        # x = self.classifier(x)
        return x
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight.data)
        torch.nn.init.constant_(self.linear_classifier.bias.data, 0)
        
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)


@register.model_register
class GNN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, activation="relu", dropout=0.5, norm='id', **kargs):
        super(GNN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden_size
        self.input_dim = input_dim
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden_size, activation, dropout, use_bn)
        self.encoder = register.encoders[kargs['encoder']](input_dim, layer_num, hidden_size, hidden_size, activation, dropout, norm, kargs['last_activation'], kargs['alpha'], kargs['theta'])
        self.classifier = torch.nn.Linear(hidden_size, output_dim)
        # self.classifier = GCNConv(hidden_size, output_dim)
        self.linear_classifier = torch.nn.Linear(hidden_size*2, output_dim)#subgraph use this
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        data = kwargs['data']
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x,_ = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.classifier(x, edge_index)
        x = self.classifier(x)
        return x,data
    
    def forward_subgraph(self, x, edge_index, batch, root_n_id, edge_weight=None, **kwargs):
        # x = torch.rand(x.shape, device=x.device)
        # x = torch.ones(x.shape, device=x.device)
        x= self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.linear_classifier(x) # use linear classifier
        # x = x[root_n_id]
        # x = self.classifier(x)
        return x
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight.data)
        torch.nn.init.constant_(self.linear_classifier.bias.data, 0)
        
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        