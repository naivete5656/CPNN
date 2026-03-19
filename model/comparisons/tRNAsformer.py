import torch
import torch.nn as nn
import numpy as np


class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.
    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).
    patch_size : int
        Size of the patch (it is a square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        The emmbedding dimension.
    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class tRNAsformer(nn.Module):
    """Simplified implementation of the Vision transformer.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(self, args, cfg):
        super().__init__()

        img_size = 224
        patch_size = 32
        in_chans = 1
        embed_dim = 192
        n_classes = 3
        n_genes = cfg.MODEL.output_dim
        have_pos_embed = True
        model_type = "multitask"
        depth = 4
        n_heads = 4
        mlp_ratio = 4
        qkv_bias = True
        p = 0.0
        attn_p = 0.0

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.have_pos_embed = have_pos_embed
        if self.have_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
            )
        self.pos_drop = nn.Dropout(p=p)
        self.drop = nn.Dropout(p=attn_p)
        self.dropout = nn.Dropout(p=0.25)

        dpr = [x.item() for x in torch.linspace(0, attn_p, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=dpr[i],
                    attn_p=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.model_type = model_type
        if self.model_type == "multitask":
            self.conv = nn.Conv1d(
                in_channels=embed_dim,
                out_channels=n_genes,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        if self.model_type == "multitask":
            self.ks = [1, 2, 5, 10, 20, 49]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward_features(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        if self.have_pos_embed:
            x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.dropout(x)
        x = self.conv(x) * mask
        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x[0]

    def forward(self, x):
        x = self.forward_features(x)  # just the CLS token
        x = self.drop(x)
        cls_preds = self.head(x[:, 0])
        if self.model_type == "multitask":
            if self.training:
                k = int(np.random.choice(self.ks))
                gene_preds = self.forward_fixed_k(x[:, 1:].transpose(2, 1), k)
            else:
                gene_preds = 0
                for k in self.ks:
                    gene_preds += self.forward_fixed_k(
                        x[:, 1:].transpose(2, 1), int(k)
                    ) / len(self.ks)
            # return gene_preds, cls_preds
            return gene_preds
        else:
            return [], cls_preds


class HE2RNA(nn.Module):
    """Model that generates one score per tile and per predicted gene.

    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """

    def __init__(self, args, cfg):
        super(HE2RNA, self).__init__()
        input_dim = cfg.MODEL.input_dim
        output_dim = cfg.MODEL.output_dim
        layers = [1024, 1024]
        nonlin = nn.ReLU()
        ks = [1, 2, 5, 10, 20, 50, 100]
        dropout = 0.25
        # device = "cpu"
        bias_init = None

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(
                in_channels=layers[i],
                out_channels=layers[i + 1],
                kernel_size=1,
                stride=1,
                bias=True,
            )
            setattr(self, "conv" + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        # self.device = device
        # self.to(self.device)

    def forward(self, x):
        if self.training:
            k = int(np.random.choice(self.ks))
            if x.shape[2] < k:
                k = x.shape[2]
            return self.forward_fixed_k(x, k)
        else:
            pred = 0
            for k in self.ks:
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred

    def forward_fixed_k(self, x, k):
        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()
        x = self.conv(x) * mask
        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
        x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
        return x[0]

    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim :]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
        x = self.layers[-1](x)
        return x


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

    if args.method == "tRNAsformer":
        model = VisionTransformer(
            img_size=tile_size,
            patch_size=vit_patch_size,
            in_chans=1,
            n_classes=3,
            embed_dim=embed_dim,
            have_pos_embed=have_pos_embed,
            depth=num_blocks,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            attn_p=0.0,
        )
        k = 49
    else:
        model = HE2RNA(1024, 16351)
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

        if args.method == "tRNAsformer":
            model(feat.reshape(224, 224).unsqueeze(0).unsqueeze(0))
        else:
            model(feat.permute(1, 0).unsqueeze(0))

        print(1)

    y = model(torch.randn(128, 3, 224, 224))
