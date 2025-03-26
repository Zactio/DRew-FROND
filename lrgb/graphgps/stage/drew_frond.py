import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool # added to aggregate the node representations into graph-level representations (Node classification to graph classificatoin)
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch_geometric.graphgym.register import register_network


# Updated codes from node classification to graph classification
# Change from OPT to CFG
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
        self.alpha_t = nn.Parameter(torch.ones(cfg.model.layers_mp))  # Replaces opt['num_layers']
        self.nu = getattr(cfg, 'nu', 1)

    def forward(self, batch):
        cfg = self.cfg  # convenience

        x = batch.x
# ------------------ Added The error occurs because the node features (i.e. batch.x) are stored as a Long tensor instead of a floating-point tensor. Dropout (and most neural network operations) expects a floating-point type. To fix the error, convert the input tensor to float before applying dropout. ------------------        
        x = x.float()
# ------------------------------------------------------------------------------------------------------------
        batch_index = batch.batch  # graph assignment for each node

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
            z = self.odeblock(x)

        z = F.relu(z)

        if cfg.fc_out:
            z = self.fc(z)
            z = F.relu(z)

        # ------------------ DRew Layer (final stage) ------------------
        z_out = z.clone()

        alpha = F.softmax(self.alpha_t, dim=0)

        if getattr(cfg, 'l2norm', False):
            z_out = F.normalize(z_out, p=2, dim=-1)

        # Pool nodes into graph embeddings
        from torch_geometric.nn import global_mean_pool
        graph_emb = global_mean_pool(z_out, batch_index)  # [num_graphs, hidden_dim]

        # Final decoding for graph classification
        out = self.m2(graph_emb)  # [num_graphs, num_classes]

        print("success!!!!")
        return out


##### Node classification code
# # Change from OPT to CFG
# class DRewFrondModel(BaseGNN):
#     def __init__(self, cfg, dataset, device=torch.device('cpu')):
#         super(DRewFrondModel, self).__init__(cfg, dataset, device)

#         self.cfg = cfg.model ## [VERIFY]  # save cfg for use in forward()

#         # FROND settings
#         self.f = set_function(cfg)
#         block = set_block(cfg)
#         time_tensor = torch.tensor([0, self.T]).to(device)
#         self.odeblock = block(self.f, cfg, device, t=time_tensor).to(device)

#         # DRew settings
#         self.alpha_t = nn.Parameter(torch.ones(cfg.layers_mp))  # Replaces opt['num_layers']
#         self.nu = getattr(cfg, 'nu', 1)

#     def forward(self, x):
#         cfg = self.cfg  # convenience

#         # ------------------ FROND ODE Layers ------------------
#         if cfg.use_labels:
#             y = x[:, -self.num_classes:]
#             x = x[:, :-self.num_classes]

#         x = F.dropout(x, cfg.input_dropout, training=self.training)
#         x = self.m1(x)

#         if cfg.use_mlp:
#             x = F.dropout(x, cfg.dropout, training=self.training)
#             x = F.dropout(x + self.m11(F.relu(x)), cfg.dropout, training=self.training)
#             x = F.dropout(x + self.m12(F.relu(x)), cfg.dropout, training=self.training)

#         if cfg.use_labels:
#             x = torch.cat([x, y], dim=-1)

#         if cfg.batch_norm:
#             x = self.bn_in(x)

#         # Solve ODE block (FROND)
#         x_init = x.clone()

#         if 'graphcon' in cfg.function:
#             x = torch.cat([x, x_init], dim=-1)
#             self.odeblock.set_x0(x)
#             z = self.odeblock(x)
#             z = z[:, cfg.hidden_dim:]
#         else:
#             self.odeblock.set_x0(x)
#             z = self.odeblock(x)

#         z = F.relu(z)

#         if cfg.fc_out:
#             z = self.fc(z)
#             z = F.relu(z)

#         # z = F.dropout(z, cfg.dropout, training=self.training)

#         # ------------------ DRew Layer (final stage) ------------------
#         # Example simplified DRew layer (needs actual propagation logic)
#         z_out = z.clone()

#         alpha = F.softmax(self.alpha_t, dim=0)

#         # (Optional) Example: Apply rewiring propagation
#         # (You'd need actual adjacency/graph edge_index logic here)
#         # z_out = rewired_propagation(z, alpha)  # Placeholder

#         # L2 normalize (optional)
#         if getattr(cfg, 'l2norm', False):
#             z_out = F.normalize(z_out, p=2, dim=-1)

#         # Final decoding
#         z_out = self.m2(z_out)

#         print("success!!!!")
#         return z_out


# # Register the model to GraphGym
# register_network('drew-frond', DRewFrondModel)

# ----------------------------------------------------- Original Frond code ------------------------------------------------------

    # # Activation.
    # z = F.relu(z)

    # if self.opt['fc_out']:
    #   z = self.fc(z)
    #   z = F.relu(z)

    # # Dropout.
    # z = F.dropout(z, self.opt['dropout'], training=self.training)

    # # Decode each node embedding to get node label.
    # z = self.m2(z)
    # return z

# ----------------------------------------------------- Original Frond code END --------------------------------------------------



# class DRewFrondModel(BaseGNN):
#     def __init__(self, opt, dataset, device=torch.device('cpu')):
#         super(DRewFrondModel, self).__init__(opt, dataset, device)

#         # FROND settings
#         self.f = set_function(opt)
#         block = set_block(opt)
#         time_tensor = torch.tensor([0, self.T]).to(device)
#         self.odeblock = block(self.f, opt, device, t=time_tensor).to(device)

#         # DRew settings
#         self.alpha_t = nn.Parameter(torch.ones(opt['num_layers']))  # Example, adapt to your DRew logic
#         self.nu = opt.get('nu', 1)

#     def forward(self, x):
#         # ------------------ FROND ODE Layers ------------------
#         if self.opt['use_labels']:
#             y = x[:, -self.num_classes:]
#             x = x[:, :-self.num_classes]

#         x = F.dropout(x, self.opt['input_dropout'], training=self.training)
#         x = self.m1(x)

#         if self.opt['use_mlp']:
#             x = F.dropout(x, self.opt['dropout'], training=self.training)
#             x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
#             x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

#         if self.opt['use_labels']:
#             x = torch.cat([x, y], dim=-1)

#         if self.opt['batch_norm']:
#             x = self.bn_in(x)

#         # Solve ODE block (FROND)
#         x_init = x.clone()
#         if 'graphcon' in self.opt['function']:
#             x = torch.cat([x, x_init], dim=-1)
#             self.odeblock.set_x0(x)
#             z = self.odeblock(x)
#             z = z[:, self.opt['hidden_dim']:]
#         else:
#             self.odeblock.set_x0(x)
#             z = self.odeblock(x)

#         z = F.relu(z)

#         if self.opt['fc_out']:
#             z = self.fc(z)
#             z = F.relu(z)

#         z = F.dropout(z, self.opt['dropout'], training=self.training)

#         # ------------------ DRew Layer (final stage) ------------------
#         # (Example simplified DRew layer)
#         z_out = z.clone()
#         alpha = F.softmax(self.alpha_t, dim=0)
#         # Assuming you have edge_index and adjacency defined elsewhere:
#         # z_out = rewired propagation on z + z

#         # L2 normalize (optional)
#         if self.opt.get('l2norm', False):
#             z_out = F.normalize(z_out, p=2, dim=-1)

#         # Final decoding
#         z_out = self.m2(z_out)
#         print("success!!!!")
#         return z_out


# # Register the model
# register_network('drew-frond', DRewFrondModel)



