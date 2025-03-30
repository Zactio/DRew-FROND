import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from utils import Meter

# Attempt 3; updated to remove automatic instantiation of odefunc so that subclasses can instantiate it correctly.
# ---------------------------------------------------------------------------------------------------------------------------------------


class ODEblock(nn.Module):
    def __init__(self, odefunc, cfg, device, t):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.t = t
        # REMOVED:
        # self.odefunc = odefunc(cfg, device, cfg.model.layers_mp)
        # The subclass is now responsible for instantiating odefunc with the proper numeric arguments.
    
    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def set_time(self, time):
        self.t = torch.tensor([0, time]).to(self.device)

    def __repr__(self):
        return f"{self.__class__.__name__}( Time Interval {self.t[0].item()} -> {self.t[1].item()} )"

class ODEFunc(MessagePassing):
    def __init__(self, cfg, device):
        super(ODEFunc, self).__init__()
        self.cfg = cfg
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None

        # ODE-specific parameters
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train2 = nn.Parameter(torch.tensor(0.0))

        # Parameters for scalar alpha/beta
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

        self.x0 = None
        self.nfe = 0  # Number of function evaluations

    def __repr__(self):
        return self.__class__.__name__


class BaseGNN(MessagePassing):
    def __init__(self, cfg, dataset, device=torch.device('cpu')):
        super(BaseGNN, self).__init__()
        self.cfg = cfg
        self.device = device

        # Dataset information
        self.num_classes = dataset.num_classes
        self.num_features = dataset.data.num_features
        
        # Time from cfg.model
        self.T = cfg.model.time

        # Initialize meters
        self.fm = Meter()
        self.bm = Meter()

        hidden_dim = cfg.model.hidden_dim

        # Initial Linear layer
        self.m1 = nn.Linear(self.num_features, hidden_dim)

        # Optional MLP layers
        if cfg.model.use_mlp:
            self.m11 = nn.Linear(hidden_dim, hidden_dim)
            self.m12 = nn.Linear(hidden_dim, hidden_dim)

        # Adjust hidden_dim if using labels
        if cfg.model.use_labels:
            hidden_dim += self.num_classes  # expand hidden dim to account for labels

        self.hidden_dim = hidden_dim

        # Fully connected output layer (optional)
        if cfg.model.fc_out:
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        # Final classification layer
        self.m2 = nn.Linear(hidden_dim, self.num_classes)

        # BatchNorm (optional)
        if cfg.model.batch_norm:
            self.bn_in = nn.BatchNorm1d(hidden_dim)
            self.bn_out = nn.BatchNorm1d(hidden_dim)

    def getNFE(self):
        """Get number of function evaluations."""
        return self.odeblock.odefunc.nfe

    def resetNFE(self):
        """Reset number of function evaluations."""
        self.odeblock.odefunc.nfe = 0

    def reset(self):
        """Reset parameters of linear layers."""
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__



# Attempt 2; with opt format
# ---------------------------------------------------------------------------------------------------------------------------------------

# # changed class to suit cfg instead of opt; removed data from ODE block 

# # class ODEblock(nn.Module):
# #     def __init__(self, odefunc, cfg, device, t):
# #         super(ODEblock, self).__init__()
# #         self.cfg = cfg
# #         self.t = t
# #         self.device = device

# #         hidden_dim = cfg.model.hidden_dim
# #         self.odefunc = odefunc(hidden_dim, hidden_dim, cfg, device)

# class ODEblock(nn.Module):
#     def __init__(self, odefunc, cfg, device, t):
#         super().__init__()
#         self.cfg = cfg
#         self.device = device
#         self.t = t

#         # hidden_dim = cfg.model.hidden_dim
#         # self.odefunc = odefunc(hidden_dim, hidden_dim, cfg, device)
#         self.odefunc = odefunc(cfg, device, cfg.model.layers_mp)


#     def set_x0(self, x0):
#         self.odefunc.x0 = x0.clone().detach()

#     def reset_tol(self):
#         self.atol = 1e-7
#         self.rtol = 1e-9
#         self.atol_adjoint = 1e-7
#         self.rtol_adjoint = 1e-9

#     def set_time(self, time):
#         self.t = torch.tensor([0, time]).to(self.device)

#     def __repr__(self):
#         return f"{self.__class__.__name__}( Time Interval {self.t[0].item()} -> {self.t[1].item()} )"


# class ODEFunc(MessagePassing):
#     # currently requires in_features = out_features; removed data from __init__
#     # def __init__(self, in_dim, out_dim, cfg, device): ORIGINAL
#     def __init__(self, cfg, device):
#         super(ODEFunc, self).__init__()
#         self.cfg = cfg
#         self.device = device
#         self.edge_index = None
#         self.edge_weight = None
#         self.attention_weights = None

#         # ODE-specific parameters
#         self.alpha_train = nn.Parameter(torch.tensor(0.0))
#         self.beta_train = nn.Parameter(torch.tensor(0.0))
#         self.beta_train2 = nn.Parameter(torch.tensor(0.0))

#         # Parameters for scalar alpha/beta
#         self.alpha_sc = nn.Parameter(torch.ones(1))
#         self.beta_sc = nn.Parameter(torch.ones(1))

#         self.x0 = None
#         self.nfe = 0  # Number of function evaluations

#     def __repr__(self):
#         return self.__class__.__name__


# class BaseGNN(MessagePassing):
#     def __init__(self, cfg, dataset, device=torch.device('cpu')):
#         super(BaseGNN, self).__init__()

#         self.cfg = cfg
#         self.device = device

#         # Dataset information
#         self.num_classes = dataset.num_classes
#         self.num_features = dataset.data.num_features
#         # self.num_nodes = dataset.data.num_nodes  # Uncomment if needed

#         # Time from cfg.model
#         self.T = cfg.model.time

#         # Initialize meters
#         self.fm = Meter()
#         self.bm = Meter()

#         hidden_dim = cfg.model.hidden_dim

#         # Initial Linear layer
#         self.m1 = nn.Linear(self.num_features, hidden_dim)

#         # Optional MLP layers
#         if cfg.model.use_mlp:
#             self.m11 = nn.Linear(hidden_dim, hidden_dim)
#             self.m12 = nn.Linear(hidden_dim, hidden_dim)

#         # Adjust hidden_dim if using labels
#         if cfg.model.use_labels:
#             hidden_dim += self.num_classes  # expand hidden dim to account for labels

#         self.hidden_dim = hidden_dim

#         # Fully connected output layer (optional)
#         if cfg.model.fc_out:
#             self.fc = nn.Linear(hidden_dim, hidden_dim)

#         # Final classification layer
#         self.m2 = nn.Linear(hidden_dim, self.num_classes)

#         # BatchNorm (optional)
#         if cfg.model.batch_norm:
#             self.bn_in = nn.BatchNorm1d(hidden_dim)
#             self.bn_out = nn.BatchNorm1d(hidden_dim)

#     def getNFE(self):
#         """Get number of function evaluations."""
#         return self.odeblock.odefunc.nfe

#     def resetNFE(self):
#         """Reset number of function evaluations."""
#         self.odeblock.odefunc.nfe = 0

#     def reset(self):
#         """Reset parameters of linear layers."""
#         self.m1.reset_parameters()
#         self.m2.reset_parameters()

#     def __repr__(self):
#         return self.__class__.__name__



# Original; with opt format
# ---------------------------------------------------------------------------------------------------------------------------------------

# class ODEblock(nn.Module):
#   def __init__(self, odefunc, opt,data, device, t):
#     super(ODEblock, self).__init__()
#     self.opt = opt
#     self.t = t
    

#     self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt,data, device)




#   def set_x0(self, x0):
#     self.odefunc.x0 = x0.clone().detach()

#   def reset_tol(self):
#     self.atol = 1e-7
#     self.rtol = 1e-9
#     self.atol_adjoint = 1e-7
#     self.rtol_adjoint = 1e-9

#   def set_time(self, time):
#     self.t = torch.tensor([0, time]).to(self.device)

#   def __repr__(self):
#     return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
#            + ")"


# class ODEFunc(MessagePassing):

#   # currently requires in_features = out_features
#   def __init__(self, opt,data, device):
#     super(ODEFunc, self).__init__()
#     self.opt = opt
#     self.device = device
#     self.edge_index = None
#     self.edge_weight = None
#     self.attention_weights = None
#     self.alpha_train = nn.Parameter(torch.tensor(0.0))
#     self.beta_train = nn.Parameter(torch.tensor(0.0))
#     self.x0 = None
#     self.nfe = 0
#     self.alpha_sc = nn.Parameter(torch.ones(1))
#     self.beta_sc = nn.Parameter(torch.ones(1))
#     self.beta_train2 = nn.Parameter(torch.tensor(0.0))

#   def __repr__(self):
#     return self.__class__.__name__


# class BaseGNN(MessagePassing):
#   def __init__(self, opt, dataset, device=torch.device('cpu')):
#     super(BaseGNN, self).__init__()
#     self.opt = opt
#     self.T = opt['time']
#     self.num_classes = dataset.num_classes
#     self.num_features = dataset.data.num_features
#     # self.num_nodes = dataset.data.num_nodes
#     self.device = device
#     self.fm = Meter()
#     self.bm = Meter()


#     self.m1 = nn.Linear(self.num_features, opt['hidden_dim'])

#     if self.opt['use_mlp']:
#       self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
#       self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
#     if opt['use_labels']:
#       # todo - fastest way to propagate this everywhere, but error prone - refactor later
#       opt['hidden_dim'] = opt['hidden_dim'] + self.num_classes
#     else:
#       self.hidden_dim = opt['hidden_dim']
#     if opt['fc_out']:
#       self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
#     self.m2 = nn.Linear(opt['hidden_dim'], self.num_classes)
#     if self.opt['batch_norm']:
#       self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'])
#       self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])


#   def getNFE(self):
#     # return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe
#     return self.odeblock.odefunc.nfe

#   def resetNFE(self):
#     self.odeblock.odefunc.nfe = 0
#     # self.odeblock.reg_odefunc.odefunc.nfe = 0

#   def reset(self):
#     self.m1.reset_parameters()
#     self.m2.reset_parameters()

#   def __repr__(self):
#     return self.__class__.__name__
