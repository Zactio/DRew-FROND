import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
import torch_geometric
from torch_geometric.data import Data

from graphgps.stage.stage_inits import init_DRewGCN
sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))
from param_calcs import get_k_neighbourhoods

class DRewGNNStage(nn.Module):
    """
    Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        # Initialize parameters using the provided numeric values from cfg
        init_DRewGCN(self, dim_in, dim_out, num_layers)
  
    def set_graph_data(self, edge_index, edge_attr=None):
        """Store graph connectivity so that it is available during integration."""
        self.edge_index = edge_index
        self.edge_attr = edge_attr  # edge attributes if available


    def forward(self, batch):
        """
        x_{t+1} = x_t + f(x_t, x_{t-1})
        Executes only one step (t = 0)
        """
        A = lambda k: batch.edge_index[:, batch.edge_attr[:, 0] == k]
        W = lambda k, t: self.W_kt["k=%d, t=%d" % (k, t)]

        t = 0
        x = []  # store history if needed

        x.append(batch.x)
        batch.x = torch.zeros_like(x[t])
        k_neighbourhoods = get_k_neighbourhoods(t)
        if cfg.agg_weights.use: # Compute alpha weights
            alpha = self.alpha_t[t]
        else:
            alpha = torch.ones(len(k_neighbourhoods))
        alpha = F.softmax(alpha, dim=0)
        if not cfg.agg_weights.convex_combo:
            alpha = alpha * len(k_neighbourhoods)
        for i, k in enumerate(k_neighbourhoods):# Aggregate messages from k-hop neighborhoods
            if A(k).shape[1] > 0:  # there are edges of type k
                delay = max(k - self.nu, 0)
                batch.x = batch.x + alpha[i] * W(k, t)(batch, x[t - delay], A(k)).x
        batch.x = x[t] + nn.ReLU()(batch.x)
        return batch.x
register_stage('drew_gnn', DRewGNNStage)

import numpy as np

def get_laplacian(edge_index):
    L = pyg.utils.get_laplacian(edge_index, normalization='sym')[0]
    L = pyg.utils.to_dense_adj(L).squeeze() # from index format to matrix
    return tonp(L)

def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr
