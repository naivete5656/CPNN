import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import initialize_weights

"""
Ilse, M., Tomczak, J. and Welling, M., 2018, July. 
Attention-based deep multiple instance learning. 
In International conference on machine learning (pp. 2127-2136). PMLR.
"""

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class AbMIL(nn.Module):

    def __init__(self, args, cfg):
        super(AbMIL, self).__init__()
        feat_dim = cfg.MODEL.input_dim
        gate = True

        dropout = False
        n_classes = cfg.MODEL.output_dim
        size = [feat_dim, 512, 384]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.head = nn.Linear(size[1], n_classes)
        initialize_weights(self)

        self.version = args.version
        if ("max" in self.version) or ("mean" in self.version):
            self.head = nn.Linear(size[0], n_classes)

        self.prop = cfg.MODEL.prop
        if self.prop:
            self.cls_head = nn.Linear(size[1], cfg.MODEL.prop_cls)

    def forward(self, h, wo_log=False):

        h = h.squeeze()
        if self.version == "output_mean":
            logits = self.head(h)
            logits = logits.mean(0)
            return logits

        elif "mean" in self.version:
            M = h.mean(0).unsqueeze(0)
        elif "max" in self.version:
            M = h.max(0)[0].unsqueeze(0)
        else:
            # print(f"raw h: {h.shape}")
            A, h = self.attention_net(h)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, h)
        logits = self.head(M)

        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        if self.prop:
            cls_prob = self.cls_head(M)
            cls_prob = F.softmax(cls_prob.mean(0), dim=-1)
            px_scale = F.softmax(logits[0], dim=-1)
            return px_scale, cls_prob
        if wo_log:
            px_scale = F.softmax(logits[0], dim=-1)

            return px_scale
        else:
            return logits[0]
        # , Y_prob, Y_hat

    def local_estimation(self, h):
        # print(f"raw h: {h.shape}")
        A, h = self.attention_net(h)  # NxK
        local_logits = self.head(h)
        return local_logits


if __name__ == "__main__":
    bag_size = 100
    feat_dim = 2048
    x = torch.randn((bag_size, feat_dim))
    net = AbMIL()
    logits, Y_prob, Y_hat = net(h=x)
    print(f"logits: {logits} Y_prob: {Y_prob} Y_hatL {Y_hat}")
