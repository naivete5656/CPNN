"""
MambaMIL
"""

import torch
import torch.nn as nn

from mamba_ssm import SRMamba
from mamba_ssm import BiMamba
from mamba_ssm import Mamba
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        n_classes,
        dropout,
        act,
        survival=False,
        layer=2,
        rate=10,
        type="SRMamba",
    ):
        internal_dim = 128
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, internal_dim)]
        if act.lower() == "relu":
            self._fc1 += [nn.ReLU()]
        elif act.lower() == "gelu":
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(internal_dim)
        self.layers = nn.ModuleList()
        self.survival = survival

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(internal_dim),
                        SRMamba(
                            d_model=internal_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(internal_dim),
                        Mamba(
                            d_model=internal_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(internal_dim),
                        BiMamba(
                            d_model=internal_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.n_classes = n_classes
        self.classifier = nn.Linear(internal_dim, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(internal_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):  # sequence shape: (num_patch, feature_dim)
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)  # (1, num_patch, feature_dim)
        h = x.float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, 512];   project from feature_dim -> 512

        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)  # LayerNorm
                h = layer[1](h)  # Mamba layer
                h = h + h_
        elif self.type == "SimpleMamba":
            h = self.layers(h)

        # Attention-based aggregation
        h = self.norm(h)  # LayerNorm
        A = self.attention(h)  # [1, num_patch, 1]
        A_raw = A.detach().clone().squeeze(2)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)  # A: attention weights of patches
        h = torch.bmm(A, h)  # weighted combination to obtain slide feature
        h = h.squeeze(0)

        logits = self.classifier(h)  # [1, n_classes]
        return logits[0]

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = None

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, None

        return logits, Y_prob, Y_hat, A_raw, results_dict

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)


class MambaMILvanira(MambaMIL):
    def __init__(self, args, cfg):
        in_dim = cfg.MODEL.input_dim
        n_classes = cfg.MODEL.output_dim
        dropout = 0.25
        act = "gelu"
        survival = False
        layer = 1
        rate = 10

        super().__init__(
            in_dim, n_classes, dropout, act, survival, layer, rate, "Mamba"
        )


class SRMambaMIL(MambaMIL):
    def __init__(self, args, cfg):

        in_dim = cfg.MODEL.input_dim
        n_classes = cfg.MODEL.output_dim
        dropout = 0.25
        act = "gelu"
        survival = False
        layer = 1
        rate = 10
        super().__init__(
            in_dim, n_classes, dropout, act, survival, layer, rate, "SRMamba"
        )

        # in_dim = 512, n_classes=10, dropout=0, act='gelu', layer = 1, rate =10, type = args.mambamil_type
