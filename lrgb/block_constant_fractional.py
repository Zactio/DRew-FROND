from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint


# Attempt 3; updated to instantiate the ODE function with the correct numeric dimensions (hidden_dim and layers_mp from cfg) rather than passing the entire cfg
# ---------------------------------------------------------------------------------------------------------------------------------------

class ConstantODEblock_FRAC(ODEblock):
    def __init__(self, odefunc, cfg, device, t=torch.tensor([0, 1])):
        # Call the base class constructor
        super().__init__(odefunc, cfg, device, t)
        self.device = device
        self.cfg = cfg
        self.model_cfg = cfg.model

        # Instantiate odefunc with the expected numerical arguments
        self.odefunc = odefunc(
            self.model_cfg.hidden_dim,  # dim_in
            self.model_cfg.hidden_dim,  # dim_out
            self.model_cfg.layers_mp    # num_layers
        )
# change 4: using getattr (As no adjoint in YAML) #### [VERIFY] IF NEED ADJOINT in ODE
#---------------------------------------------------------------------------------------------------------------------------------------        
        if getattr(self.model_cfg, 'adjoint', False):
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
# ---------------------------------------------------------------------------------------------------------------------------------------            
## part of attempt 3 below
# ---------------------------------------------------------------------------------------------------------------------------------------
        # Select the appropriate ODE integrator
        # if self.model_cfg.adjoint:
        #     from torchdiffeq import odeint_adjoint as odeint
        # else:
        #     from torchdiffeq import odeint
# ---------------------------------------------------------------------------------------------------------------------------------------
        self.train_integrator = odeint
        self.test_integrator = odeint

    def forward(self, x):
        t = self.t.type_as(x)
        integrator = self.train_integrator if self.training else self.test_integrator
        func = self.odefunc
        state = x

        alpha = torch.tensor(self.model_cfg.alpha_ode)

        if alpha > 1 or alpha <= 0:
            raise ValueError("alpha_ode must be in (0, 1)")

        # Use fractional ODE integration via fdeint
        z = fdeint(func, state, alpha,
                   t=self.model_cfg.time,
                   step_size=self.model_cfg.step_size,
                   method=self.model_cfg.method)

        return z



# # Attempt 2
# # ---------------------------------------------------------------------------------------------------------------------------------------



# # old 
# # Changed structure from Opt to Cfg

# # class ConstantODEblock_FRAC(ODEblock):
# # from base_classes import ODEblock
# # import torch
# # from torchfde import fdeint

# # class ConstantODEblock_FRAC(ODEblock):
# #     def __init__(self, odefunc, cfg, device, t=torch.tensor([0, 1])):
# #         # Initialize the parent class (ODEblock)
# #         super(ConstantODEblock_FRAC, self).__init__(odefunc, cfg, device, t)

# ## New code now passes 4 arguments instead of 5; doesn't need data
# class ConstantODEblock_FRAC(ODEblock):
#     def __init__(self, odefunc, cfg, device, t=torch.tensor([0, 1])):
#         super().__init__(odefunc, cfg, device, t)

#         self.device = device
#         self.cfg = cfg
#         self.model_cfg = cfg.model

#         # Initialize odefunc
#         self.odefunc = odefunc(
#             self.model_cfg.hidden_dim,
#             self.model_cfg.hidden_dim,
#             self.model_cfg.layers_mp
#         )
        

#         # ODE integrator selection
#         if self.model_cfg.adjoint:
#             from torchdiffeq import odeint_adjoint as odeint
#         else:
#             from torchdiffeq import odeint

#         self.train_integrator = odeint
#         self.test_integrator = odeint

#     def forward(self, x):
#         t = self.t.type_as(x)
#         integrator = self.train_integrator if self.training else self.test_integrator
#         func = self.odefunc
#         state = x

#         alpha = torch.tensor(self.model_cfg.alpha_ode)

#         if alpha > 1 or alpha <= 0:
#             raise ValueError("alpha_ode must be in (0, 1)")

#         # fdeint fractional ODE integration
#         z = fdeint(func, state, alpha,
#                    t=self.model_cfg.time,
#                    step_size=self.model_cfg.step_size,
#                    method=self.model_cfg.method)

#         return z



# class ConstantODEblock_FRAC(ODEblock):
#   def __init__(self, odefunc,  opt,  device, t=torch.tensor([0, 1])):
#     super(ConstantODEblock_FRAC, self).__init__(odefunc,  opt, data,   device, t)

#     self.odefunc = odefunc(dim_in, dim_out, num_layers)
#     # if opt['data_norm'] == 'rw':
#     #   edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
#     #                                                                fill_value=opt['self_loop_weight'],
#     #                                                                num_nodes=data.num_nodes,
#     #                                                                dtype=data.x.dtype)
#     # else:
#     #   edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
#     #                                        fill_value=opt['self_loop_weight'],
#     #                                        num_nodes=data.num_nodes,
#     #                                        dtype=data.x.dtype)
#     # self.odefunc.edge_index = edge_index.to(device)
#     # self.odefunc.edge_weight = edge_weight.to(device)
#     # self.reg_odefunc = None
#     # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

#     if opt['adjoint']:
#       from torchdiffeq import odeint_adjoint as odeint
#     else:
#       from torchdiffeq import odeint

#     self.train_integrator = odeint
#     self.test_integrator = odeint
#     # self.set_tol()
#     self.device = device
#     self.opt = opt
#   def forward(self, x):
#     t = self.t.type_as(x)

#     integrator = self.train_integrator if self.training else self.test_integrator
    
#     # reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

#     # func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
#     # state = (x,) + reg_states if self.training and self.nreg > 0 else x

#     func = self.odefunc
#     state = x


#     alpha = torch.tensor(self.opt['alpha_ode'])

#     if alpha > 1:
#         raise ValueError("alpha_ode must be in (0,1)")

#     z = fdeint(func, state, alpha, t=self.opt['time'], step_size=self.opt['step_size'], method=self.opt['method'])
#     return z

#   def __repr__(self):
#     return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
#            + ")"
