import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool 
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch_geometric.graphgym.register import register_network
import torch_geometric
from torch_geometric.data import Data

class DRewFrondModel(BaseGNN):
    def __init__(self, cfg, dataset, device=torch.device('cpu')):
        super(DRewFrondModel, self).__init__(cfg, dataset, device)
        self.cfg = cfg.model 
        # FROND settings
        self.f = set_function(cfg)
        block = set_block(cfg)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, cfg, device, t=time_tensor).to(device)

        # DRew settings
        self.alpha_t = nn.Parameter(torch.ones(cfg.model.layers_mp))  
        self.nu = getattr(cfg, 'nu', 1)

    def forward(self, batch):
        cfg = self.cfg 
        x = batch.x
        x = x.float()
        batch_index = batch.batch 

        # ------------------ FROND ODE Layers ------------------
        if cfg.use_labels:
            y = x[:, -self.num_classes:]
            x = x[:, :-self.num_classes]

        x = F.dropout(x, cfg.input_dropout, training=self.training)
        x = self.m1(x)

        if cfg.use_mlp:
            x = F.dropout(x, cfg.dropout, training=self.training)
            x = F.dropout(x + self.m11(F.relu(x)), cfg.dropout, training=self.training)
            x = F.dropout(x + self.m12(F.relu(x)), cfg.dropout, training=self.training)

        if cfg.use_labels:
            x = torch.cat([x, y], dim=-1)

        if cfg.batch_norm:
            x = self.bn_in(x)

        # Solve ODE block (FROND)
        x_init = x.clone()

        if 'graphcon' in cfg.function:
            x = torch.cat([x, x_init], dim=-1)
            self.odeblock.set_x0(x)
            z = self.odeblock(x)
            z = z[:, cfg.hidden_dim:]
        else:
            self.odeblock.set_x0(x)
            z = self.odeblock(x, batch)

        z = F.relu(z)

        if cfg.fc_out:
            z = self.fc(z)
            z = F.relu(z)

        z_out = z.clone()

        alpha = F.softmax(self.alpha_t, dim=0)

        if getattr(cfg, 'l2norm', False):
            z_out = F.normalize(z_out, p=2, dim=-1)

        # Pool nodes into graph embeddings
        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(z_out, batch_index)  # [num_graphs, hidden_dim]

        # Final decoding for graph classification
        out = self.m2(graph_emb)  
        return out, batch.y

register_network('drew-frond', DRewFrondModel)