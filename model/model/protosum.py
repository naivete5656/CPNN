import pickle

import pandas as pd
import numpy as np
import anndata as ad
import anndata as ad


import torch
import torch.nn as nn
import torch.nn.functional as F

from .projection_module import ConsNexProjectionModule, Adapter, LongProjectionModule


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ProtoSum(nn.Module):

    def __init__(self, args, cfg):
        super().__init__()
        self.version = args.version

        data_dir = f"{cfg.DATASET.data_dir}/{args.dataset}-digital_slide"

        if args.resolution == "":
            args.resolution = "fine"
        if args.version == "naive":
            inf_vector = pd.read_csv(f"{data_dir}/{args.fold}/{args.resolution}_naive_vector.csv", index_col=0)
            self.cell_type = inf_vector.index
            self.org_temp = torch.Tensor(inf_vector.values)
        else:
            with open(f"{data_dir}/{args.fold}/{args.resolution}_parameter_dict.pkl", "rb") as f:
                params = pickle.load(f)

            self.cell_type = params["theta"].columns
            self.org_temp = torch.Tensor(params["theta"].values).T
        self.num_genes = self.org_temp.shape[0]

        if "mask" in self.version:
            self.mask = 1 - torch.Tensor(params["mask"].values.T)
        if "static" in self.version:
            self.exp_template = torch.log(self.org_temp)
        else:
            self.exp_template = torch.nn.Parameter(torch.log(self.org_temp))

        if ("pcc" not in self.version) and ("wocorrection" not in self.version) and ("naive" not in self.version):
            self.m_g = torch.Tensor(params["m_g"])
            self.alpha_g = torch.nn.Parameter(torch.log(self.m_g))

            if "beta" in self.version:
                self.beta_g = torch.nn.Parameter(torch.log(torch.full_like(self.m_g, 1e-5)))
            else:
                self.beta_g = torch.nn.Parameter(torch.log(torch.full_like(self.m_g, 1e-10)))

        if "scratch" in self.version:
            self.exp_template = torch.nn.Parameter(torch.randn(self.org_temp.shape))

        self.softplus = nn.Softplus()
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=-1)

        if args.projector == "adapter":
            self.prj = Adapter(cfg.MODEL.input_dim, self.exp_template.shape[0])
        elif args.projector == "convnex":
            self.prj = ConsNexProjectionModule(cfg.MODEL.input_dim, self.exp_template.shape[0])
        elif args.projector == "longformer":
            self.prj = LongProjectionModule(cfg.MODEL.input_dim, self.exp_template.shape[0])
        else:
            self.prj = self.projection(cfg.MODEL.input_dim, self.exp_template.shape[0])

        initialize_weights(self)

    def projection(self, input_dim, out_dim, num_layer=2, hidden_dims=[1024, 512]):
        projection_layers = []
        last_dim = input_dim
        for i in range(num_layer):
            projection_layers.append(nn.Linear(last_dim, hidden_dims[i]))
            projection_layers.append(nn.GELU())
            last_dim = hidden_dims[i]

        projection_layers.append(nn.Linear(last_dim, out_dim))
        return nn.Sequential(*projection_layers)

    def forward(self, patches, coords=None, only_prop=False):
        logits_per_image = self.prj(patches)

        probs = self.softmax(logits_per_image)
        if "mask" in self.version:
            exp_template = self.exp_template.to(probs) * self.mask.to(probs)
        else:
            exp_template = self.exp_template.to(probs)
        # exp_template = self.softmax(exp_template)
        exp_template = torch.exp(exp_template)

        exps = probs @ exp_template

        if (
            ("pcc" in self.version)
            or ("scratch" in self.version)
            or ("wocorrection" in self.version)
            or ("naive" in self.version)
        ):
            bulk_exp = exps.mean(0)
        else:
            bulk_exp = (torch.exp(self.alpha_g.to(probs)) * exps + torch.exp(self.beta_g.to(probs))).mean(0)

        bulk_exp = bulk_exp / bulk_exp.sum()
        if "proto_reg" in self.version:
            return bulk_exp, probs, (exp_template, self.org_temp)
        else:
            return bulk_exp, probs

    def local_estimation(self, patches, wo_log=False):
        logits_per_image = self.prj(patches)
        probs = self.softmax(logits_per_image)
        exp_template = self.softmax(self.exp_template.to(probs))

        exps = probs @ exp_template

        return exps, probs


if __name__ == "__main__":
    import sys

    from main import get_parse
    from utils import load_config

    args = get_parse()
    if args.method in ["AbMIL", "CLAM_MB", "DSMIL", "ILRA", "SumExpModel"]:
        args.config_dir = f"{args.config_dir}/train_cfg.yaml"
    else:
        args.config_dir = f"{args.config_dir}/{args.method}.yaml"
    cfg = load_config(args.config_dir)

    args.simple_img = True

    x = torch.randn(128, 2048).to("cuda")

    model = templateAggregate(args, cfg)
    model = model.to("cuda")
    y = model(x)
