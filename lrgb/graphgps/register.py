from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN

def add_custom_model_cfg():
    # Add any keys that are NOT part of default GraphGym's model config
    cfg.model.hidden_dim = 256
    cfg.model.dropout = 0.2
    cfg.model.input_dropout = 0.6
    cfg.model.use_mlp = True
    cfg.model.use_labels = False
    cfg.model.batch_norm = True
    cfg.model.fc_out = True
    cfg.model.function = 'DREW'
    cfg.model.nu = 1
    cfg.model.l2norm = True
    cfg.model.alpha_ode = 0.85
    cfg.model.time = 40
    cfg.model.block = 'constant_graph'
    cfg.model.method = 'predictor'
    cfg.model.step_size = 1.0
    cfg.model.layers_mp = 23
