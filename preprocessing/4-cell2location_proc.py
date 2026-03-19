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


def protptype_generation(scadata, save_dir, batch_key, label_key):
    scadata = scadata[~scadata.obs[label_key].isna()]
    # prepare anndata for the regression model
    cell2location.models.RegressionModel.setup_anndata(
        adata=scadata,
        batch_key=batch_key,
        labels_key=label_key,
    )

    mod = RegressionModel(scadata)

    # view anndata_setup as a sanity check
    mod.view_anndata_setup()

    scadata.X = scadata.layers["counts"]

    mod.train(max_epochs=250)  # 250
    mod.plot_history(20)

    adata_ref = mod.export_posterior(scadata, sample_kwargs={"num_samples": 1000, "batch_size": 2500})

    # Save model
    mod.save(f"{save_dir}/sc_rec", overwrite=True)

    # Save anndata object with results
    adata_file = f"{save_dir}/c2loc_scadata.h5ad"
    adata_ref.write(adata_file)

    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
    ].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]
    # inf_aver.iloc[0:5, 0:5]
    # inf_aver.shape
    # inf_aver.to_csv(f"{save_dir}/sc_temp.csv")
    # mod.plot_QC()
    return inf_aver


def deconvolution(adata_bulk, inf_aver, save_dir, args):

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata_bulk, layer="raw_count")

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_bulk,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=20,
    )
    mod.view_anndata_setup()
    mod.train(
        max_epochs=30000,
        # max_epochs=1,
        # train using full data (batch_size=None)
        batch_size=None,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1,
    )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    # mod.plot_history(1000)
    # plt.legend(labels=["full data training"])

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_vis = mod.export_posterior(
        adata_bulk,
        sample_kwargs={"num_samples": 1000, "batch_size": mod.adata.n_obs},
    )

    # Save model
    mod.save(
        f"{save_dir}/cell2loc_v2",
        overwrite=True,
    )

    # mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

    # Save anndata object with results
    adata_file = f"{save_dir}/cell2loc_bulk_adata.h5ad"
    adata_vis.write(adata_file)

    parameter_dict = {
        "W": adata_vis.obsm["means_cell_abundance_w_sf"],
        "theta": inf_aver,
        "m_g": adata_vis.uns["mod"]["post_sample_means"]["m_g"],
        "s_eg": adata_vis.uns["mod"]["post_sample_means"]["s_g_gene_add"],
        "y_s": adata_vis.uns["mod"]["post_sample_means"]["detection_y_s"],
    }
    return parameter_dict


def generate_mask(scadata, label_key):
    cell_level_mask = {}
    for cell_type in scadata.obs[label_key].unique():
        mask = np.array(scadata.X[scadata.obs[label_key] == cell_type].mean(axis=0))[0] == 0
        cell_level_mask[cell_type] = mask
    return cell_level_mask


def main(args):
    Path(f"{args.base_dir}/{args.dataset}/cell2location{args.version}/{args.fold}/{args.resolution}").mkdir(
        parents=True, exist_ok=True
    )

    # load scdata and bulk data
    scadata = sc.read_h5ad(f"{args.base_dir}/{args.dataset}/scdataset{args.version}/adata.h5py")

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

    # estimate mean expression profiles from single cell data
    Keys = BATCH_KEY_DICT[args.dataset]
    batch_key = Keys["batch_key"]
    label_key = Keys["label_key"][args.resolution]
    inf_aver = protptype_generation(
        scadata,
        f"{args.base_dir}/{args.dataset}/cell2location{args.version}/{args.fold}/{args.resolution}",
        batch_key,
        label_key,
    )

    # estimate cell adundance from bulk and estimated mean expression
    parameter_dict = deconvolution(
        adata_bulk,
        inf_aver,
        f"{args.base_dir}/{args.dataset}/cell2location{args.version}/{args.fold}/{args.resolution}",
        args,
    )

    mask = generate_mask(scadata, label_key)
    parameter_dict["mask"] = parameter_dict["theta"].copy()
    for cell_type, mask_array in mask.items():
        parameter_dict["mask"].loc[:, cell_type] = mask_array

    save_dir = Path(f"{args.save_dir}/{args.dataset}{args.version}/{args.fold}")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(
        f"{save_dir}/{args.resolution}_parameter_dict.pkl",
        "wb",
    ) as f:
        pickle.dump(parameter_dict, f)


if __name__ == "__main__":
    args = get_parse()
    main(args)
