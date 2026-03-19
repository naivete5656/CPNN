import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import initialize_weights

"""
Ilse, M., Tomczak, J. and Welling, M., 2018, July. 
Attention-based deep multiple instance learning. 
In International conference on machine learning (pp. 2127-2136). PMLR.

and

https://github.com/maragraziani/interpretableWSItoRNAseq/tree/master
"""


class AbRegMIL(nn.Module):

    def __init__(self, args, cfg):
        # def __init__(self):
        super().__init__()
        feat_dim = cfg.MODEL.input_dim
        n_classes = cfg.MODEL.output_dim
        # feat_dim = 2048
        # n_classes = 16355
        size = [feat_dim, 512, 384]

        self.fc = nn.Linear(size[0], size[1])

        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(size[1], size[2]),
            torch.nn.Tanh(),
            torch.nn.Linear(size[2], 1),
        )

        self.head = nn.Linear(size[1], n_classes)

        initialize_weights(self)

        # self.prop = cfg.MODEL.prop
        # if self.prop:
        #     self.cls_head = nn.Linear(size[1], cfg.MODEL.prop_cls)

    def forward(self, h, wo_log=False):
        h = self.fc(h)
        gene_exp = self.head(h)

        A = self.attention_net(h)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        gene_exp = A @ gene_exp

        return gene_exp[0]
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
    net = AbRegMIL()
    logits = net(h=x)
    print(f"logits: {logits} Y_prob: {Y_prob} Y_hatL {Y_hat}")
