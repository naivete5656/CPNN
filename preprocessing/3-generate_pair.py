from pathlib import Path
import argparse
import pickle
import random

import numpy as np
import anndata as ad
from tqdm import tqdm
import h5py


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BRCA")
    parser.add_argument("--base_dir", type=str, default="./raw_dataset/processing")
    parser.add_argument("--save_dir", type=str, default="./dataset")
    parser.add_argument("--num_fold", type=int, default="4")
    parser.add_argument("--version", type=str, default="-digital_slide")
    parser.add_argument(
        "--feat_name",
        type=str,
        default="feature",
        choices=[
            "feature",
            "featuredinov2",
            "feature4mil",
            "feature_opt",
            "feature_conch",
        ],
    )

    args = parser.parse_args()
    return args


def generate_pair(img_dir, bulk_dir, save_dir):
    patch_paths = sorted(Path(img_dir).glob("*.h5"))
    bulk_rna = ad.read_h5ad(bulk_dir)
    save_dir = Path(save_dir)

    for patch_path in tqdm(patch_paths):
        if patch_path.stem not in bulk_rna.obs_names:
            continue

        with h5py.File(patch_path, "r") as hdf5_file:
            feats = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        slide_name = patch_path.stem
        # raw_cnt = bulk_rna[patch_path.stem].X[0]
        tpm = bulk_rna[slide_name].X[0]
        raw_cnt = bulk_rna[slide_name].layers["raw_count"][0]

        # Create HDF5 file and save data
        with h5py.File(f"{save_dir}/{patch_path.stem}.h5", "w") as f:
            # Create datasets and save data
            f.create_dataset("feat", data=feats)
            f.create_dataset("coord", data=coords)
            f.create_dataset("tpm", data=tpm)
            f.create_dataset("raw_count", data=raw_cnt)


def split_data(save_dir):
    split_dict = {}
    patch_dir = Path(save_dir)

    patch_paths = sorted(patch_dir.glob("*.h5"))
    sample_ids = [patch_path.stem.split("_")[0] for patch_path in patch_paths]
    uniq_sample_ids = np.unique(sample_ids)

    np.random.seed(42)
    random.shuffle(uniq_sample_ids)

    split_list_org = np.array_split(uniq_sample_ids, args.num_fold)

    for fold in range(args.num_fold):
        split_list = split_list_org.copy()
        split_list.pop(fold)

        if fold == args.num_fold - 1:
            val_slideid = split_list.pop(0)
        else:
            val_slideid = split_list.pop(fold)

        train_slideid = np.concatenate(split_list)

        train_paths = []
        val_paths = []
        test_paths = []
        for patch_path in patch_paths:
            slide_id = patch_path.stem.split("_")[0]
            if slide_id in train_slideid:
                train_paths.append(patch_path)
            elif slide_id in val_slideid:
                val_paths.append(patch_path)
            else:
                test_paths.append(patch_path)
        split_dict[fold] = {"train": train_paths, "test": test_paths, "val": val_paths}
    save_dir = Path(save_dir).parent
    with open(f"{save_dir}/split.pkl", mode="wb") as f:
        pickle.dump(split_dict, f)


if __name__ == "__main__":
    args = get_parse()

    save_dir = Path(
        f"{args.save_dir}/{args.dataset}{args.version}/sample_pair_{args.feat_name}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    img_dir = (
        f"{args.base_dir}/{args.dataset}/TCGA{args.version}/{args.feat_name}/h5_files"
    )
    bulk_dir = (
        f"{args.base_dir}/{args.dataset}/TCGA{args.version}/adata_bulk_processed.h5ad"
    )

    generate_pair(img_dir, bulk_dir, save_dir)
    split_data(save_dir)
