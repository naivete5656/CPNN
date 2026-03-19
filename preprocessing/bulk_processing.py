from pathlib import Path

from tqdm import tqdm
import pandas as pd
import anndata as ad
import scanpy as sc


def generate_bulkadata(save_dir, iddict_path, data_type="tpm"):
    rna_paths = sorted(Path(f"{save_dir}/RNA").glob("*.tsv"))
    # combining bulk_rna
    for rna_path in tqdm(rna_paths):
        bulk_exp = pd.read_csv(rna_path, delimiter="\t", skiprows=1)
        bulk_exp["gene_id"]
        bulk_exp["gene_id"] = bulk_exp["gene_id"].str.split(".").apply(lambda x: x[0])

        bulk_exp.set_index("gene_id", inplace=True)
        if data_type == "tpm":
            bulk_exp = bulk_exp[["tpm_unstranded"]].rename(
                columns={"tpm_unstranded": rna_path.stem}
            )
        else:
            bulk_exp = bulk_exp[["unstranded"]].rename(
                columns={"unstranded": rna_path.stem}
            )
        bulk_exp.drop(
            index=["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"],
            inplace=True,
        )
        duplicated_keys = bulk_exp.index[bulk_exp.index.duplicated(keep=False)]
        sum_exp = bulk_exp.loc[duplicated_keys].groupby(level=0).sum()
        bulk_exp.drop(index=duplicated_keys, inplace=True)
        bulk_exp = pd.concat([bulk_exp, sum_exp]).sort_index()
        if ("df" in globals()) or ("df" in locals()):
            df = pd.merge(df, bulk_exp, how="inner", on="gene_id")
        else:
            df = bulk_exp
            df = df.dropna(axis=0, how="any")

    bulk_exp = df
    # add gene name 2 gene id
    id2name = pd.read_csv(iddict_path, index_col=0)
    geneids = bulk_exp.index

    gene_name_list = []

    for geneid in geneids:
        if geneid in id2name.index:
            gene_name = id2name.loc[geneid]["gene_name"]
        else:
            gene_name = geneid
        gene_name_list.append(gene_name)

    len([gene for gene in gene_name_list if gene.startswith("ENSG")])

    bulk_exp["gene_name"] = gene_name_list
    bulk_exp.set_index("gene_name", inplace=True)
    # bulk_exp = bulk_exp.drop(columns=["gene_id"])
    duplicated_keys = bulk_exp.index[bulk_exp.index.duplicated(keep=False)]
    sum_exp = bulk_exp.loc[duplicated_keys].groupby(level=0).sum()
    bulk_exp.drop(index=duplicated_keys, inplace=True)
    bulk_exp = pd.concat([bulk_exp, sum_exp]).sort_index()

    bulk_adata = ad.AnnData(bulk_exp.T)
    bulk_adata.write_h5ad(f"{save_dir}/bulk_tpm.h5ad")

    bulk_adata.obs = bulk_adata.obs.astype(str)

    if data_type == "tpm":
        # mitochondrial genes, "MT-" for human, "Mt-" for mouse
        bulk_adata.var["mt"] = bulk_adata.var_names.str.startswith("MT-")
        # ribosomal genes
        bulk_adata.var["ribo"] = bulk_adata.var_names.str.startswith(("RPS", "RPL"))
        # hemoglobin genes
        bulk_adata.var["hb"] = bulk_adata.var_names.str.contains("^HB[^(P)]")

        sc.pp.calculate_qc_metrics(
            bulk_adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )

        bulk_adata = bulk_adata[bulk_adata.obs.pct_counts_mt < 30, :].copy()
        bulk_adata = bulk_adata[bulk_adata.obs.n_genes_by_counts < 40000, :].copy()

        sc.pp.filter_cells(bulk_adata, min_genes=5000)
        sc.pp.filter_genes(bulk_adata, min_cells=50)

        bulk_adata = bulk_adata[:, ~bulk_adata.var["mt"]]
        bulk_adata = bulk_adata[:, ~bulk_adata.var["ribo"]]
        bulk_adata = bulk_adata[:, ~bulk_adata.var["hb"]]

    bulk_adata.write_h5ad(f"{save_dir}/TCGA_bulk_adata_{data_type}.h5ad")
    return bulk_adata


def concat_rawexp_tpm(
    save_dir,
):
    raw_cout = sc.read_h5ad(f"{save_dir}/TCGA_bulk_adata_raw_count.h5ad")
    tpm = sc.read_h5ad(f"{save_dir}/TCGA_bulk_adata_tpm.h5ad")
    tpm.layers["raw_count"] = raw_cout[tpm.obs_names, tpm.var_names].X

    return tpm
