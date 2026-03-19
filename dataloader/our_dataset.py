from pathlib import Path
import pickle

import numpy as np
import h5py
import pandas as pd
import torch

CellType = [
    "Endothelial ACKR1",
    "Endothelial RGS5",
    "Endothelial CXCL12",
    "CAFs MSC iCAF-like",
    "CAFs myCAF-like",
    "PVL Differentiated",
    "PVL Immature",
    "Endothelial Lymphatic LYVE1",
    "B cells Memory",
    "B cells Naive",
    "T cells CD8+",
    "T cells CD4+",
    "NK cells",
    "Cycling T-cells",
    "NKT cells",
    "Macrophage",
    "Monocyte",
    "Cycling_Myeloid",
    "DCs",
    "Myoepithelial",
    "Luminal Progenitors",
    "Mature Luminal",
    "Plasmablasts",
    "Cancer Cycling",
    "Cancer Her2 SC",
    "Cancer LumB SC",
    "Cycling PVL",
    "Cancer Basal SC",
    "Cancer LumA SC",
]


class PropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        args,
        cfg,
    ):
        data_dir = self.load_sample_paths(args, cfg, split)

        if args.resolution == "":
            args.resolution = "fine"
        with open(f"{data_dir}/{args.fold}/{args.resolution}_parameter_dict.pkl", "rb") as f:
            params = pickle.load(f)

        self.cell_prop = params["W"]
        self.cell_prop.columns = self.cell_prop.columns.str.replace("meanscell_abundance_w_sf_", "")
        self.cell_prop[sorted(self.cell_prop.columns)]

        if hasattr(args, "stop_sampling"):
            self.stop_sampling = False
        else:
            self.stop_sampling = True
        self.sampling_st = args.sampling_st if hasattr(args, "sampling_st") else "constant"

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
        exclude_names = [
            "TCGA-OL-A66L-01Z_14_DX.h5",
            "TCGA-OL-A66K-01Z_2058_DX.h5",
            "TCGA-OL-A66J-01Z_797_DX.h5",
        ]
        # for sample_path in self.sample_paths:
        #     if sample_path.name in exclude_names:
        #         print(f"Excluding {sample_path.name} from dataset")
        self.sample_paths = [path for path in self.sample_paths if path.name not in exclude_names]
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
        coords = np.round(coords / 256)
        feats = torch.Tensor(feats)
        exps = torch.Tensor(exps)

        try:
            prop = torch.Tensor(self.cell_prop.loc[sample_path.stem].values)
            prop = prop / prop.sum()
        except:
            prop = None

        if self.loss_type != "NegativeBinomial":
            exps = exps / exps.sum() * 1e4
            exps = torch.log(1 + exps)

        if (feats.shape[0] > 4096) and self.stop_sampling:
            if self.sampling_st == "constant":
                sampled_indices = torch.randperm(feats.shape[0], generator=self.torch_gen)[:4096]
            else:
                sample_num = self.np_rng.randint(4096, 8092)
                sampled_indices = self.np_rng.choice(feats.shape[0], sample_num, replace=False)

            # サンプリングした特徴を返す
            feats = feats[sampled_indices]
            coords = coords[sampled_indices]

        return {
            "patch": feats,
            "exp": exps,
            "slide_name": sample_path.stem,
            "prop": prop,
            "coords": coords,
        }
        # return {
        #     "patch": feats,
        #     "exp": exps,
        #     "slide_name": sample_path.stem,
        #     "prop": prop,
        #     "coords": coords,
        # }


if __name__ == "__main__":
    import sys

    sys.path.append("/home/hdd/kazuya/digital_slide")
    from utils import load_config
    from main import get_parse

    args = get_parse()

    if args.method in ["AbMIL", "CLAM_MB", "DSMIL", "ILRA", "SumExpModel"]:
        args.config_dir = f"{args.config_dir}/train_cfg.yaml"
    else:
        args.config_dir = f"{args.config_dir}/{args.method}.yaml"
    cfg = load_config(args.config_dir, args.trainer)

    dataset = PropDataset("train", args, cfg)

    for data in dataset:
        pass
