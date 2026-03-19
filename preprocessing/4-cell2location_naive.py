import argparse
from pathlib import Path
import pickle

import scanpy as sc
import cell2location

# create the regression model
from cell2location.models import RegressionModel

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

BATCH_KEY_DICT = {
    "BRCA": {
        "batch_key": "Patient",
        "label_key": {
            "coarse": "celltype_major",
            "medium": "celltype_subset",
            "fine": "celltype_minor",
        },
    },
    "LUAD": {
        "batch_key": "Sample",
        "label_key": {
            "coarse": "Cell_type",
            "medium": "Cell_type.refined",
            "fine": "Cell_subtype",
        },
    },
    "KIRC": {
        "batch_key": "patient",
        "label_key": {
            "coarse": "summaryDescription",
            "medium": "broad_type",
            "fine": "annotation",
        },
    },
}


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BRCA")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dataset",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./raw_dataset/processing",
    )
    parser.add_argument("--version", type=str, default="-digital_slide")
    parser.add_argument("--resolution", type=str, default="medium")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    return args


def protptype_generation(scadata, batch_key, label_key):
    scadata = scadata[~scadata.obs[label_key].isna()]
    labels = scadata.obs[label_key].astype("category").cat.categories

    mean_list = []
    for label in labels:
        subset = scadata[scadata.obs[label_key] == label]
        mean_expr = np.asarray(subset.X.mean(axis=0)).ravel()
        mean_list.append(mean_expr)

    inf_aver = pd.DataFrame(mean_list, index=labels, columns=scadata.var_names)
    return inf_aver


def main(args):
    Path(f"{args.base_dir}/{args.dataset}/cell2location{args.version}/{args.fold}/{args.resolution}").mkdir(
        parents=True, exist_ok=True
    )

    # load scdata and bulk data
    if Path(f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata_clustering.h5py").exists():
        scadata = sc.read_h5ad(f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata_clustering.h5py")
    else:
        scadata = sc.read_h5ad(f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata.h5py")
        sc.tl.pca(scadata)
        sc.pp.neighbors(scadata)
        sc.tl.umap(scadata)
        for res in [0.02, 0.5, 2.0]:
            sc.tl.leiden(scadata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")
        scadata.write_h5ad(f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata_clustering.h5py")

    adata_bulk = sc.read_h5ad(f"{args.base_dir}/{args.dataset}/TCGA{args.version}/adata_bulk_processed.h5ad")

    # load scdata and bulk data
    with open(f"{args.save_dir}/{args.dataset}{args.version}/split.pkl", "rb") as f:
        split_dict = pickle.load(f)

    if "all_slides" not in args.version:
        slide_name_list = []
        for slide_path in split_dict[args.fold]["train"]:
            slide_name_list.append(slide_path.stem)
        for slide_path in split_dict[args.fold]["val"]:
            slide_name_list.append(slide_path.stem)
        adata_bulk = adata_bulk[slide_name_list]

    if scadata.shape[1] != adata_bulk.shape[1]:
        common_gene = np.intersect1d(adata_bulk.var_names, scadata.var_names)
        scadata = scadata[:, common_gene]
        adata_bulk = adata_bulk[:, common_gene]

    for args.resolution in [f"leiden_res_0.50", f"leiden_res_2.00"]:
        # estimate mean expression profiles from single cell data
        Keys = BATCH_KEY_DICT[args.dataset]
        batch_key = Keys["batch_key"]
        if args.resolution in ["fine", "medium", "coarse"]:
            label_key = Keys["label_key"][args.resolution]
        else:
            label_key = args.resolution
        inf_aver = protptype_generation(
            scadata,
            batch_key,
            label_key,
        )

        save_dir = Path(f"{args.save_dir}/{args.dataset}{args.version}/{args.fold}")
        save_dir.mkdir(parents=True, exist_ok=True)
        inf_aver.to_csv(
            f"{save_dir}/{args.resolution}_naive_vector.csv",
        )


if __name__ == "__main__":
    args = get_parse()
    main(args)
