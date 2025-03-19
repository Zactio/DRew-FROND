import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch_geometric.graphgym.register import register_network


class DRewFrondModel(BaseGNN):
    def __init__(self, opt, dataset, device=torch.device('cpu')):
        super(DRewFrondModel, self).__init__(opt, dataset, device)

        # FROND settings
        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, device, t=time_tensor).to(device)

        # DRew settings
        self.alpha_t = nn.Parameter(torch.ones(opt['num_layers']))  # Example, adapt to your DRew logic
        self.nu = opt.get('nu', 1)

    def forward(self, x):
        # ------------------ FROND ODE Layers ------------------
        if self.opt['use_labels']:
            y = x[:, -self.num_classes:]
            x = x[:, :-self.num_classes]

        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        if self.opt['use_mlp']:
            x = F.dropout(x, self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
            x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

        if self.opt['use_labels']:
            x = torch.cat([x, y], dim=-1)

        if self.opt['batch_norm']:
            x = self.bn_in(x)

        # Solve ODE block (FROND)
        x_init = x.clone()
        if 'graphcon' in self.opt['function']:
            x = torch.cat([x, x_init], dim=-1)
            self.odeblock.set_x0(x)
            z = self.odeblock(x)
            z = z[:, self.opt['hidden_dim']:]
        else:
            self.odeblock.set_x0(x)
            z = self.odeblock(x)

        z = F.relu(z)

        if self.opt['fc_out']:
            z = self.fc(z)
            z = F.relu(z)

        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # ------------------ DRew Layer (final stage) ------------------
        # (Example simplified DRew layer)
        z_out = z.clone()
        alpha = F.softmax(self.alpha_t, dim=0)
        # Assuming you have edge_index and adjacency defined elsewhere:
        # z_out = rewired propagation on z + z

        # L2 normalize (optional)
        if self.opt.get('l2norm', False):
            z_out = F.normalize(z_out, p=2, dim=-1)

        # Final decoding
        z_out = self.m2(z_out)
        print("success!!!!")
        return z_out


# Register the model
register_network('drew-frond', DRewFrondModel)



