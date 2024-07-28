import torch
from torch_geometric.loader import DataLoader
import torch_geometric
from tqdm import tqdm
import numpy as np
import yaml 
from yaml import SafeLoader

from utils.args import Arguments
from utils.sampling import subsampling
from data.load import load_data
from models import load_model

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GCN2Conv
from torch_geometric.utils import add_self_loops, degree


def ADD_ALL(data):
    x, edge_index = data.x, data.edge_index
    device = data.x.device

    # isolate self loops 
    self_loop_mask =  edge_index[0] == edge_index[1]
    edge_index_self_loop = edge_index[:, self_loop_mask]
    edge_index = edge_index[:, ~self_loop_mask]

    edge_index_to_halfhop = edge_index

    # add new adjust—nodes, and use linear interpolation to initialize their features
    insert_node_ids = torch.arange(edge_index_to_halfhop.size(1), device=device) + data.num_nodes
    x_insert_node = x[edge_index_to_halfhop[0]]
    x_insert_node.mul_(0.5).add_(x[edge_index_to_halfhop[1]], alpha=0.5)
    new_x = torch.cat([x, x_insert_node], dim=0)

    # add new edges between slow nodes and the original nodes that replace the original edges
    edge_index_slow = [
        torch.stack([edge_index_to_halfhop[0], insert_node_ids]),
        torch.stack([insert_node_ids, edge_index_to_halfhop[1]]),
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
#       torch.zeros(edge_index_self_loop.size(1), device=device),


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
    
class Pre_MPNN_Model(MessagePassing):
    def __init__(self, node_features_dim, trace_dim, K, w, pre_trace = 'mpnn'):
        super(Pre_MPNN_Model, self).__init__()
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
            print("trace_pre_compute:pure_mpnn")
            data.trace_all = trace_all
        elif self.preprocess == 'zero':
            print("trace_pre_compute:zero")
            data.trace_all = torch.zeros_like(trace_all) 
        return data


    
def train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = 0
        
    for epoch in tqdm(range(config.epochs)):
        model.train()
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        if config.earlystop:
            val_acc = eval_subgraph(model, val_loader, device)
            if val_acc > best_val:
                cnt = 0
                best_test_fromval = eval_subgraph(model, test_loader, device)
                best_val = val_acc
            else:
                cnt += 1
                if cnt >= patience:
                    print(f'early stop at epoch {epoch}')
                    break
    if not config.earlystop:
        best_test_fromval = eval_subgraph(model, test_loader, device)
    return best_test_fromval

def eval_subgraph(model, data_loader, device):
    model.eval()
    
    correct = 0
    total_num = 0
    for batch in data_loader:
        batch = batch.to(device)
        preds = model.forward_subgraph(batch.x, batch.edge_index, batch.batch, batch.root_n_index).argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total_num += batch.y.shape[0]
    acc = correct / total_num
    return acc

def train_fullgraph(model, optimizer, criterion, config, data, device):
    if config.earlystop:
        cnt = 0
        patience = config.patience
        best_val = 0
        best_test_fromval = 0
    model.train()
    data = data.to(device)
    for epoch in tqdm(range(config.epochs)):
        optimizer.zero_grad()
        out,data.trace_all = model(data.x, data.edge_index, data = data)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if config.earlystop:
            val_acc = eval_fullgraph(model, data, device, config)
            if val_acc > best_val:
                cnt = 0
                best_test_fromval = eval_fullgraph(model, data, device, config)
                best_val = val_acc
            else:
                cnt += 1
                if cnt >= patience:
                    print(f'early stop at epoch {epoch}')
                    break
    if not config.earlystop:
        best_test_fromval = eval_fullgraph(model, data, device, config)
    return best_test_fromval


def eval_fullgraph(model, data, device, config):
    model.eval()
    data = data.to(device)
    pred,_ = model(data.x, data.edge_index, data = data)
    pred= pred.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc    

def train_eval(model, optimizer, criterion, config, data, train_loader, val_loader, test_loader, device):
    if config.subsampling:
        test_acc = train_subgraph(model, optimizer, criterion, config, train_loader, val_loader, test_loader, device)
    else:
        test_acc = train_fullgraph(model, optimizer, criterion, config, data, device)
    return test_acc

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    acc_list = []
    for i in range(10):
        # load data
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=i)
        data.y = data.y.squeeze()
        #####
        if 'ADDNODE' in config.model:
            pre_mpnn = Pre_MPNN_Model(data.x.shape[1],config.hidden_size,config.layer_num,config.w,config.pre_trace)
            data = pre_mpnn(data)
            data = ADD_ALL(data)
            #print(data)
        #####
        model = load_model(data.x.shape[1], num_classes, config).to(device)
        params = model.parameters()
        # 计算参数量
        num_params = 0
        for param in params:
            num_params += torch.prod(torch.tensor(param.size()))
        print('模型参数量：', num_params)

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
        

if __name__ == '__main__':
    args = Arguments().parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)
    # combine args and config
    for k, v in config.items():
        args.__setattr__(k, v)
    print(args)
    main(args)