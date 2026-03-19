# Code from the awesome lucidrains: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
# with some of my own modifications

import torch
from torch import nn
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from einops import rearrange
import pdb


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SEQUOIA(nn.Module):
    def __init__(self, args, cfg):
        super().__init__()

        num_outputs = cfg.MODEL.output_dim
        dim = 1024
        depth = 6
        heads = 16
        mlp_dim = 2048
        dim_head = 64
        num_clusters = 100

        self.pos_emb1D = nn.Parameter(torch.randn(num_clusters, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_outputs))

    def forward(self, x):
        # pe = posemb_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + self.pos_emb1D

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)[0]


if __name__ == "__main__":
    import sys

    sys.path.append("/home/hdd/kazuya/bulk_trial/simple_baseline")

    from main import get_parse
    from utils import load_config

    args = get_parse()
    args.config_dir = f"{args.config_dir}/train_cfg.yaml"
    cfg = load_config(args.config_dir, args.trainer)

    from dataloader.dataset import SimpleDataset

    dataset = SimpleDataset("train", args, cfg)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = SEQUOIA(
        num_outputs=16355,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        device=device,
    )
    k = 100

    from sklearn.cluster import KMeans

    np.random.seed(42)
    kmeans = KMeans(n_clusters=k, random_state=42)
    for data in dataset:
        feats = data["patch"]
        coords = data["coords"].numpy()
        kmeans.fit(coords)
        labels = kmeans.labels_

        feat_list = []
        for n_label in np.unique(labels):
            feat_list.append(feats[labels == n_label].mean(0))

        feat = torch.Tensor(np.array(feat_list))

        y = model(feat.unsqueeze(0))

        print(1)

    y = model(torch.randn(128, 3, 224, 224))
