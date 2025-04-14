from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint

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
        if getattr(self.model_cfg, 'adjoint', False):
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.train_integrator = odeint
        self.test_integrator = odeint

    def forward(self, x, batch):
        t = self.t.type_as(x)
        integrator = self.train_integrator if self.training else self.test_integrator
        func = self.odefunc
        state = x

        alpha = torch.tensor(self.model_cfg.alpha_ode)

        if alpha > 1 or alpha <= 0:
            raise ValueError("alpha_ode must be in (0, 1)")

        # Use fractional ODE integration via fdeint
        z = fdeint(func, state, alpha, # func here is function_drew_gnn
                   t=self.model_cfg.time,
                   step_size=self.model_cfg.step_size,
                   method=self.model_cfg.method, batch=batch)

        return z


