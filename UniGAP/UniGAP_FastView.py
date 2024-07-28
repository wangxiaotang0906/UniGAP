import torch
from torch_geometric.loader import DataLoader
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GCN2Conv
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear

def ADD_ALL(data):
    x, edge_index = data.x, data.edge_index
    device = data.x.device

    # isolate self loops 
    self_loop_mask =  edge_index[0] == edge_index[1]
    edge_index_self_loop = edge_index[:, self_loop_mask]
    edge_index = edge_index[:, ~self_loop_mask]
    edge_index_to_addnode = edge_index

    # add new adjust—nodes, and use linear interpolation to initialize their features
    insert_node_ids = torch.arange(edge_index_to_addnode.size(1), device=device) + data.num_nodes
    x_insert_node = x[edge_index_to_addnode[0]]
    x_insert_node.mul_(0.5).add_(x[edge_index_to_addnode[1]], alpha=0.5)
    new_x = torch.cat([x, x_insert_node], dim=0)

    # add new edges between slow nodes and the original nodes that replace the original edges
    edge_index_slow = [
        torch.stack([edge_index_to_addnode[0], insert_node_ids]),
        torch.stack([insert_node_ids, edge_index_to_addnode[1]]),
        ]
    
    #new_edge_index = torch.cat([edge_index,edge_index_self_loop, *edge_index_slow], dim=1)
    new_edge_index = torch.cat([edge_index, *edge_index_slow], dim=1)

    # prepare a mask that distinguishes between original nodes and slow nodes
    insert_node_mask = torch.cat([
        torch.zeros(x.size(0), device=device),
        torch.ones(insert_node_ids.size(0), device=device)
    ], dim=0).bool()

    slow_edge_mask = torch.cat([
        torch.zeros(edge_index.size(1), device=device),
        torch.ones(torch.cat(edge_index_slow,dim=1).size(1), device=device)
    ], dim=0).bool()

    data.x, data.edge_index, data.insert_node_mask, data.slow_edge_mask = new_x, new_edge_index, insert_node_mask, slow_edge_mask
    return data

class GCNConv_wo_lin(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 3: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
class Pre_Trace_Model(MessagePassing):
    def __init__(self, node_features_dim, trace_dim, K, w, pre_trace = 'mpnn'):
        super(Pre_Trace_Model, self).__init__()
        self.conv1 = GCNConv(node_features_dim, trace_dim)
        for i in range(1,K):
            setattr(self, f'conv{i+1}', GCNConv_wo_lin())
        self.K = K
        self.w = w
        self.convs = [getattr(self, f'conv{i+1}') for i in range(K)]
        self.preprocess = pre_trace

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        trace_all = []
        for i in range(self.K):
            x = self.convs[i](x,edge_index)
            if (i+1) % self.w == 0:
                x = F.normalize(x, p=2, dim=1)
            trace_all.append(x)
        trace_all = torch.stack(trace_all,dim=0)
        if self.preprocess == 'mpnn':
            print("trace_pre_compute : pure_mpnn")
            data.trace_all = trace_all
        elif self.preprocess == 'zero':
            print("trace_pre_compute : zero")
            data.trace_all = torch.zeros_like(trace_all) 
        return data


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


class UniGAP_GNN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden_size=128, output_dim=70, activation="relu", dropout=0.5, norm='id', **kargs):
        super(UniGAP_GNN, self).__init__()
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
        self.linear_classifier = torch.nn.Linear(hidden_size*2, output_dim)
    
    def forward(self, x=None, edge_index=None, edge_weight=None, frozen=False, **kwargs):

        ###################################### UniGAP ###############################################
        data = kwargs['data']
        data = self.trace_mlp(data)
        data = self.edge_scorer(data)

        # gumbel—softmax:
        active_edge = torch.nn.functional.gumbel_softmax(data.edge_scores, tau=0.5, hard=True, eps=1e-10, dim=-1)[:,0]

        # no gumbel:
        # y_soft = data.edge_scores.softmax(dim=-1)
        # index = y_soft.max(dim=-1, keepdim=True)[1]
        # y_hard = torch.zeros_like(data.edge_scores, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        # ret = y_hard - y_soft.detach() + y_soft
        # active_edge = ret[:,0]

        num_edge = data.edge_index[:,~data.slow_edge_mask].size(1)
        
        train_mask = torch.ones(data.edge_index.size(1))
        train_mask[:num_edge] = active_edge
        for i in range(num_edge):
            train_mask[num_edge+i] = 1-train_mask[i]
            train_mask[2*num_edge+i] = 1-train_mask[i]
        train_mask = train_mask.bool()
        x,edge_index = data.x,data.edge_index[:,train_mask]

        #print('now active edges:',edge_index.size(1))

        ##################################### DOWNSTREAM #############################################
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x,trace = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight) # you can diy your downstream model here.
            trace = trace.permute(1, 0, 2)
            trace_all = trace[~data.insert_node_mask]
            x = x[~data.insert_node_mask]
            trace_all = trace_all.permute(1, 0, 2)         
        x = self.classifier(x)
        return x,trace_all
    

    def reset_classifier(self):
        torch.nn.init.xavier_uniform_(self.linear_classifier.weight.data)
        torch.nn.init.constant_(self.linear_classifier.bias.data, 0)
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    acc_list = []
    for i in range(20):
        # load data
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=i)
        data.y = data.y.squeeze()
        ################################### Pre_Trace_compute #######################################
        if 'UniGAP' in config.model:
            pre_mpnn = Pre_Trace_Model(data.x.shape[1],config.hidden_size,config.layer_num,config.w,config.pre_trace)
            data = pre_mpnn(data)
            data = ADD_ALL(data)

        model = load_model(data.x.shape[1], num_classes, config).to(device)
        params = model.parameters()
        # coompute params nums
        num_params = 0
        for param in params:
            num_params += torch.prod(torch.tensor(param.size()))
        print('num_params：', num_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, val_loader, test_loader = None, None, None
        if config.subsampling:
            train_loader, val_loader, test_loader = subsampling(data, config, sampler=config.sampler)
        test_acc = train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device)
        print(i, test_acc)
        acc_list.append(test_acc)
    
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc*100:.2f}±{final_acc_std*100:.2f}")
        

