from pathlib import Path
import argparse

import scanpy as sc
import anndata as ad
import numpy as np
import pickle
import os


def generate_mask(scadata, label_key):
    cell_level_mask = {}
    for cell_type in scadata.obs[label_key].unique():
        mask = (
            np.array(scadata.X[scadata.obs[label_key] == cell_type].mean(axis=0))[0]
            == 0
        )
        cell_level_mask[cell_type] = mask
    return cell_level_mask


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
    parser.add_argument("--resolution", type=str, default="fine")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    scadata = sc.read_h5ad(
        f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata.h5py"
    )
    adata_bulk = sc.read_h5ad(
        f"{args.base_dir}/{args.dataset}/TCGA{args.version}/adata_bulk_processed.h5ad"
    )

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

    Keys = BATCH_KEY_DICT[args.dataset]
    batch_key = Keys["batch_key"]
    label_key = Keys["label_key"][args.resolution]

    save_dir = Path(f"{args.save_dir}/{args.dataset}{args.version}/{args.fold}")
    with open(
        f"{save_dir}/{args.resolution}_parameter_dict.pkl",
        "rb",
    ) as f:
        parameter_dict = pickle.load(f)

    mask = generate_mask(scadata, label_key)
    parameter_dict["mask"] = parameter_dict["theta"].copy()
    for cell_type, mask_array in mask.items():
        parameter_dict["mask"].loc[:, cell_type] = mask_array

    with open(
        f"{save_dir}/{args.resolution}_parameter_dict.pkl",
        "wb",
    ) as f:
        pickle.dump(parameter_dict, f)
