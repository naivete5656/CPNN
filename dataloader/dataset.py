from pathlib import Path
import pickle

import numpy as np
import h5py
import pandas as pd
import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        args,
        cfg,
    ):
        self.data_dir = self.load_sample_paths(args, cfg, split)

        if "stop_sampling" not in args.version or "MOSBY" in args.version:
            self.stop_sampling = False
        else:
            self.stop_sampling = True

        self.loss_type = cfg.MODEL.loss_type
        self.raw_count = "raw_count" in args.version

        seed = getattr(args, "seed", 42)
        self.np_rng = np.random.RandomState(seed)
        self.torch_gen = torch.Generator().manual_seed(seed)

    def load_sample_paths(self, args, cfg, split):
        data_dir = f"{cfg.DATASET.data_dir}/{args.dataset}-digital_slide"
        # sample_pair_{args.feat_name}
        with open(f"{data_dir}/split.pkl", "rb") as f:
            split_dict = pickle.load(f)
        sample_paths = split_dict[args.fold][split]
        self.sample_paths = [
            Path(str(p).replace("sample_pair_feature", f"sample_pair_{args.feat_name}")) for p in sample_paths
        ]

        if args.data_type == "ds":
            self.sample_paths = [sample_path for sample_path in self.sample_paths if "_DX." in sample_path.name]
        elif args.data_type == "ts":
            self.sample_paths = [sample_path for sample_path in self.sample_paths if "_TS." in sample_path.name]

        return data_dir

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample_path = self.sample_paths[index]
        with h5py.File(sample_path, "r", swmr=True) as f:
            feats = f["feat"][:]
            coords = f["coord"][:]
            if self.loss_type == "NegativeBinomial":
                exps = f["raw_count"][:]
            else:
                exps = f["tpm"][:]
        # coords = np.round(coords / 256)
        coords = np.round(coords)
        feats = torch.Tensor(feats)
        exps = torch.Tensor(exps)

        if self.loss_type != "NegativeBinomial":
            exps = exps / exps.sum() * 1e4
            exps = torch.log(1 + exps)

        if (feats.shape[0] > 4096) and (self.stop_sampling):
            sampled_indices = torch.randperm(feats.shape[0], generator=self.torch_gen)[:4096]
            # サンプリングした特徴を返す
            feats = feats[sampled_indices]
            coords = coords[sampled_indices]

        return {
            "patch": feats,
            "exp": exps,
            "slide_name": sample_path.stem,
            "coords": torch.Tensor(coords),
        }


def collate_fn(batch: tuple):
    """Custom collate function of train dataloader for TRIPLEX.

    Args:
        batch (tuple): batch of returns from Dataset

    Returns:
        tuple: batch data
    """

    return batch


if __name__ == "__main__":
    import sys

    from utils import load_config
    from main import get_parse

    args = get_parse()
    cfg = load_config(args.config_name)

    dataset = SimpleDataset("train", args, cfg)

    for data in dataset:
        pass
