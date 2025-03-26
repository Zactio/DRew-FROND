# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage

# from torch_geometric.graphgym.models.layer import LayerConfig, new_layer_config
# def GNNLayer(dim_in, dim_out, has_act=True):
#     """
#     Wrapper for a GNN layer

#     Args:
#         dim_in (int): Input dimension
#         dim_out (int): Output dimension
#         has_act (bool): Whether has activation function after the layer

#     """
#     return GeneralLayer(
#         cfg.gnn.layer_type,
#         layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
#                                       has_bias=False, cfg=cfg))
