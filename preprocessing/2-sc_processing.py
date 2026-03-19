from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import scipy.io
import anndata as ad
from scipy.io import mmread
import scanpy as sc
import scvi
import numpy as np


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BRCA")
    parser.add_argument("--save_dir", type=str, default="./raw_dataset/processing")
    parser.add_argument("--sc_raw_data_dir", type=str, default="./raw_dataset/sc_raw")
    parser.add_argument("--version", type=str, default="digital_slide")
    args = parser.parse_args()
    return args


def load_dataset(sc_dir, dataset):

    if dataset == "KIRC":
        adata = sc.read_h5ad(f"{sc_dir}/{dataset}/RCC_upload_final_raw_counts.h5ad")
        batch_label = "patient"
        annotation_label = "annotation"
    elif dataset == "LUAD":
        # Load .mtx file as a sparse matrix
        mtx_path = f"{sc_dir}/{dataset}/GSE131907_Lung_Cancer_raw_counts.mtx"
        sparse_matrix = mmread(mtx_path)

        print(sparse_matrix.shape)  # (n_rows, n_cols)
        print(type(sparse_matrix))  # scipy.sparse sparse matrix format

        annot_file = f"{sc_dir}/{dataset}/GSE131907_Lung_Cancer_cell_annotation.txt.gz"
        annot = pd.read_csv(annot_file, sep="\t", compression="gzip", index_col=0)
        var_names = pd.read_csv(
            f"{sc_dir}/{dataset}/GSE131907_Lung_Cancer_var_names.txt",
            header=None,
        )
        obs_names = pd.read_csv(
            f"{sc_dir}/{dataset}/GSE131907_Lung_Cancer_obs_names.txt",
            header=None,
        )
        adata = ad.AnnData(
            X=sparse_matrix.T.tocsr(),  # expression matrix of cells x genes
            obs=pd.DataFrame(index=obs_names[0].values),  # per-cell metadata
            # annot,  # cell-level annotations
            var=pd.DataFrame(index=var_names[0].values),  # per-gene metadata
        )
        adata.obs = annot.loc[adata.obs_names]
        batch_label = "Sample"
        annotation_label = "Cell_type"
    elif dataset == "BRCA":
        meta = pd.read_csv(
            f"{sc_dir}/{dataset}/Whole_miniatlas_meta.csv",
            index_col="NAME",
        )
        adata = sc.read_10x_mtx(f"{sc_dir}/{dataset}/BrCa_Atlas_Count_out")
        adata.obs = adata.obs.join(meta)
        batch_label = "Patient"
        annotation_label = "celltype_minor"

    return adata, batch_label, annotation_label


def processing_adata(adata, threshold=30):
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=True,
    )

    # filtering data
    adata = adata[adata.obs.pct_counts_mt < threshold, :]
    adata = adata[adata.obs.n_genes_by_counts < 2500, :].copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=200)

    # sc.pp.scrublet(adata, batch_key="Patient")

    # Saving count data
    adata.layers["counts"] = adata.X.copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Logarithmize the data
    sc.pp.log1p(adata)

    # sc.pp.highly_variable_genes(adata, n_top_genes=2048, batch_key=batch_label)
    # sc.pl.highly_variable_genes(adata)

    return adata


def make_bulk_adata(
    save_dir,
    iddict_path,
):
    rna_paths = sorted(Path(f"{save_dir}/RNA").glob("*.tsv"))
    for rna_path in tqdm(rna_paths):
        bulk_exp = pd.read_csv(rna_path, delimiter="\t", skiprows=1)
        bulk_exp["gene_id"]
        bulk_exp["gene_id"] = bulk_exp["gene_id"].str.split(".").apply(lambda x: x[0])

        bulk_exp.set_index("gene_id", inplace=True)
        bulk_exp = bulk_exp[["unstranded"]].rename(columns={"unstranded": rna_path.stem})
        bulk_exp.drop(
            index=["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"],
            inplace=True,
        )
        duplicated_keys = bulk_exp.index[bulk_exp.index.duplicated(keep=False)]
        sum_exp = bulk_exp.loc[duplicated_keys].groupby(level=0).sum()
        bulk_exp.drop(index=duplicated_keys, inplace=True)
        bulk_exp = pd.concat([bulk_exp, sum_exp]).sort_index()
        # exp_df_list.append(bulk_exp)
        if ("df" in globals()) or ("df" in locals()):
            df = pd.merge(df, bulk_exp, how="inner", on="gene_id")
        else:
            df = bulk_exp
            df = df.dropna(axis=0, how="any")
    bulk_exp_raw_count = df
    id2name = pd.read_csv(iddict_path, index_col=0)
    geneids = bulk_exp_raw_count.index

    gene_name_list = []
    for geneid in geneids:
        if geneid in id2name.index:
            gene_name = id2name.loc[geneid]["gene_name"]
        else:
            gene_name = geneid
        gene_name_list.append(gene_name)

    print(len([gene for gene in gene_name_list if gene.startswith("ENSG")]))

    bulk_exp_raw_count["gene_name"] = gene_name_list
    bulk_exp_raw_count.set_index("gene_name", inplace=True)
    duplicated_keys = bulk_exp_raw_count.index[bulk_exp_raw_count.index.duplicated(keep=False)]
    sum_exp = bulk_exp_raw_count.loc[duplicated_keys].groupby(level=0).sum()
    bulk_exp_raw_count.drop(index=duplicated_keys, inplace=True)
    bulk_exp_raw_count = pd.concat([bulk_exp_raw_count, sum_exp]).sort_index()

    duplicated_keys = bulk_exp_raw_count.index[bulk_exp_raw_count.index.duplicated(keep=False)]
    sum_exp = bulk_exp_raw_count.loc[duplicated_keys].groupby(level=0).sum()
    bulk_exp_raw_count.drop(index=duplicated_keys, inplace=True)
    bulk_exp_raw_count = pd.concat([bulk_exp_raw_count, sum_exp]).sort_index()

    # adata_bulk = ad.AnnData(bulk_exp_raw_count.loc[scadata.var_names].T)
    adata_bulk = ad.AnnData(bulk_exp_raw_count.T)

    adata_bulk.write_h5ad(f"{save_dir}/bulk_raw_count.h5ad")

    # adata_bulk = sc.read_h5ad(f"{save_dir}/bulk_raw_count.h5ad")
    tpm = sc.read_h5ad(f"{save_dir}/TCGA_bulk_adata.h5ad")
    tpm.layers["raw_count"] = adata_bulk[tpm.obs_names, tpm.var_names].X

    return tpm


def main(args):
    save_dir = Path(f"{args.save_dir}/{args.dataset}")
    save_dir.joinpath(f"scdataset{args.version}").mkdir(parents=True, exist_ok=True)

    adata, batch_label, annotation_label = load_dataset(args.sc_raw_data_dir, args.dataset)
    adata = processing_adata(adata)
    if args.dataset == "BRCA":
        adata.obs["nCount_RNA"] = adata.obs["nCount_RNA"].astype(int)
        adata.obs["Percent_mito"] = adata.obs["Percent_mito"].astype(float)
        adata.obs["nFeature_RNA"] = adata.obs["nFeature_RNA"].astype(int)
    adata.write_h5ad(f"{save_dir}/scdataset{args.version}/adata.h5py")

    adata_bulk = ad.read_h5ad(f"{save_dir}/TCGA{args.version}/adata_bulk_concat.h5ad")
    common_genes = np.intersect1d(adata_bulk.var_names, adata.var_names)
    np.save(f"{save_dir}/common_genes{args.version}", common_genes)
    adata_bulk[:, common_genes].write_h5ad(f"{save_dir}/TCGA{args.version}/adata_bulk_processed.h5ad")

    # generate_mask()


if __name__ == "__main__":
    args = get_parse()
    main(args)
