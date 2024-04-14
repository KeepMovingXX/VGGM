import numpy as np
import scipy.io as scio
from torch import nn
import torch
from torch_geometric.data import Data

def load_graph(filename, device, dim):
    cooWhole = scio.mmread(filename)
    cooarray = cooWhole.toarray()
    x = torch.eye(int(np.sqrt(cooarray.shape[1])),device=device)
    data_list = []
    for i in range(cooarray.shape[0]):
        edge_indexs = []
        edge_source = []
        edge_target = []
        for j in range(cooarray.shape[1]):
            if cooarray[i][j] != 0:
                i1 = int(j / np.sqrt(cooarray.shape[1]))
                i2 = int(j % np.sqrt(cooarray.shape[1]))
                edge_source.append(i1)
                edge_target.append(i2)
        edge_indexs.append(edge_source)
        edge_indexs.append(edge_target)
        edge_indexs = torch.tensor(edge_indexs, dtype=torch.long).to(device)
        data = Data(x=x, edge_index=edge_indexs)
        data_list.append(data)
    return data_list

def load_trad(filename, device, nodes):
    x = torch.eye(nodes, device=device)
    edge_sorce = []
    edge_target = []
    data_list = []
    edge_all = []
    guard = '1948'
    with open(filename, 'r') as f:
        for line in f:
            walk = line.split()
            if walk[4] == guard:
                edge_sorce.append(int(walk[1]))
                edge_target.append(int(walk[3]))
                edge_sorce.append(int(walk[3]))
                edge_target.append(int(walk[1]))
            else:
                edge_all.append(edge_target)
                edge_all.append(edge_sorce)
                edge_indexs = torch.tensor(edge_all, dtype=torch.long).to(device)
                data = Data(x=x, edge_index=edge_indexs)
                data_list.append(data)
                guard = walk[4]
                edge_sorce = []
                edge_target = []
                edge_all = []
    return data_list

def load_sbm(filename, device, nodes):
    x = torch.eye(nodes, device=device)
    edge_sorce = []
    edge_target = []
    data_list = []
    edge_all = []
    guard = '0'
    with open(filename, 'r') as f:
        for line in f:
            walk = line.strip().split(',')
            if walk[0] == guard:
                edge_sorce.append(int(walk[1]))
                edge_target.append(int(walk[2]))
                edge_sorce.append(int(walk[2]))
                edge_target.append(int(walk[1]))
            else:
                edge_all.append(edge_target)
                edge_all.append(edge_sorce)
                edge_indexs = torch.tensor(edge_all, dtype=torch.long).to(device)
                data = Data(x=x, edge_index=edge_indexs)
                data_list.append(data)
                guard = walk[0]
                edge_sorce = []
                edge_target = []
                edge_all = []
    return data_list


def predictResult(assignments):
    """Get the potential changePoints
    This function takes the resultant assignments(type:list) as input,
    such as [1,1,1,3,3,3,2,2,2,1,1,2,3,3,1,3,3,2...]，then using the
    specific spatial information inside the data to find possible changePoint.
    Parameters
    ----------
    assignments: the label of each point which
    given by clustering method
    Returns
    ----------
    p_c: potential changePoints
    """
    # print ("assignments is ", assignments)
    predicts = []
    temp = assignments[0]
    for i in range(1, len(assignments)):
        if temp == assignments[i]:
            continue
        predicts.append(i)
        temp = assignments[i]
    p_c = set(predicts)
    return p_c


