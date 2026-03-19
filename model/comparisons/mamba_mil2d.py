import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .mamba_simple import MambaConfig as SimpleMambaConfig
from .mamba_simple import Mamba as SimpleMamba


def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL_2D(nn.Module):
    def __init__(self, args, cfg):
        super().__init__()
        input_dim = cfg.MODEL.input_dim
        n_classes = cfg.MODEL.output_dim
        survival = False
        mambamil_layer = 1
        mambamil_dim = 256
        mambamil_state_dim = 8
        mambamil_inner_layernorms = False
        pscan = True
        cuda_pscan = True
        mamba_2d_max_w = 1024 * 256
        mamba_2d_max_h = 1024 * 256
        # if args.dataset == "BRCA":
        #     mamba_2d_max_w = 212297
        #     mamba_2d_max_h = 418560
        # elif args.dataset == "LUAD":
        #     mamba_2d_max_w = 197796
        #     mamba_2d_max_h = 110976
        # elif args.dataset == "KIRC":
        #     mamba_2d_max_w = 205262
        #     mamba_2d_max_h = 222080
        mamba_2d_pad_token = "trainable"
        # mamba_2d_pad_token = "zero"
        mamba_2d_patch_size = 256

        self.pos_emb_type = None
        pos_emb_dropout = 0

        drop_out = 0.25
        patch_encoder_batch_size = cfg.TRAIN.batch_size

        self._fc1 = [nn.Linear(input_dim, mambamil_dim)]
        self._fc1 += [nn.GELU()]
        if drop_out > 0:
            self._fc1 += [nn.Dropout(drop_out)]

        self._fc1 = nn.Sequential(*self._fc1)

        self.norm = nn.LayerNorm(mambamil_dim)

        self.layers = nn.ModuleList()
        self.patch_encoder_batch_size = patch_encoder_batch_size
        config = SimpleMambaConfig(
            d_model=mambamil_dim,
            n_layers=mambamil_layer,
            d_state=mambamil_state_dim,
            inner_layernorms=mambamil_inner_layernorms,
            pscan=pscan,
            use_cuda=cuda_pscan,
            expand_factor=1,
            mamba_2d=True,
            mamba_2d_max_w=mamba_2d_max_w,
            mamba_2d_max_h=mamba_2d_max_h,
            mamba_2d_pad_token=mamba_2d_pad_token,
            mamba_2d_patch_size=mamba_2d_patch_size,
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = n_classes

        self.attention = nn.Sequential(
            nn.Linear(mambamil_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(mambamil_dim, self.n_classes)
        self.survival = survival

        if self.pos_emb_type == "linear":
            self.pos_embs = nn.Linear(2, mambamil_dim)
            self.norm_pe = nn.LayerNorm(mambamil_dim)
            self.pos_emb_dropout = nn.Dropout(pos_emb_dropout)
        else:
            self.pos_embs = None

        self.apply(initialize_weights)

    def forward(self, x, coords):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)  # (1, num_patch, feature_dim)
        hid = x.float()  # [1, num_patch, feature_dim]

        hid = self._fc1(
            hid
        )  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.pos_emb_type == "linear":
            pos_embs = self.pos_embs(coords)
            hid = hid + pos_embs.unsqueeze(0)
            hid = self.pos_emb_dropout(hid)

        hid = self.layers(hid, coords, self.pos_embs)

        hid = self.norm(hid)  # LayerNorm
        A = self.attention(hid)  # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:
            A = A.permute(0, 3, 1, 2)
            A = A.view(1, 1, -1)
            hid = hid.view(1, -1, self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        hid = torch.bmm(
            A, hid
        )  # [1, 1, 512] , weighted combination to obtain slide feature
        hid = hid.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(hid)  # [1, n_classes]
        return logits[0]

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = None

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None  # same return as other models

        # return logits, Y_prob, Y_hat, results_dict, None  # same return as other models

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
