import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class SumExpModel(nn.Module):

    def __init__(self, args, cfg):
        super().__init__()
        if "dino" in args.version:
            cfg.MODEL.input_dim = 768

        if args.projector == "LP":
            self.prj = nn.Linear(cfg.MODEL.input_dim, cfg.MODEL.output_dim)
        else:
            self.prj = self.projection(cfg.MODEL.input_dim, cfg.MODEL.output_dim)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax()

        self.prop = cfg.MODEL.prop
        if self.prop:
            self.cls_head = nn.Linear(cfg.MODEL.input_dim, cfg.MODEL.prop_cls)
        initialize_weights(self)

        self.version = args.version

    def projection(self, input_dim, out_dim, num_layer=2, hidden_dims=[1024, 512]):
        projection_layers = []
        last_dim = input_dim
        for i in range(num_layer):
            projection_layers.append(nn.Linear(last_dim, hidden_dims[i]))
            projection_layers.append(nn.GELU())
            last_dim = hidden_dims[i]

        projection_layers.append(nn.Linear(last_dim, out_dim))
        return nn.Sequential(*projection_layers)

    def forward(self, patches, wo_log=False):
        # x \in Instances x Dim
        if ("MOSBY" in self.version) and (self.training):
            sample_size = np.random.randint(0, 8000)
            sample_size = min(len(patches), sample_size)
            indices = torch.randperm(len(patches))[:sample_size]

            patches = patches[indices]

        x = self.prj(patches)
        if self.prop:
            cls_prob = self.cls_head(patches)
            cls_prob = F.softmax(cls_prob.mean(0), dim=-1)
            px_scale = F.softmax(x.mean(0), dim=-1)
            return px_scale, cls_prob
        if wo_log:
            px_scale = F.softmax(x.mean(0), dim=-1)
            return px_scale
        else:
            x = self.softplus(x)
            bulk_exp = x.mean(0) * 1e4
        return torch.log(1 + bulk_exp)

    def local_estimation(self, patches, wo_log=False):
        x = self.prj(patches)

        if wo_log:
            px_scale = F.softmax(x, dim=-1)
            return px_scale
        else:
            x = self.softplus(x)
            bulk_exp = x * 1e4
        return torch.log(1 + bulk_exp)
