import argparse
from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm
import h5py
import openslide
import anndata as ad

from bulk_processing import generate_bulkadata, concat_rawexp_tpm


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BRCA")
    parser.add_argument("--source_dir", type=str, default="raw_dataset/TCGA_raw")
    parser.add_argument("--save_dir", type=str, default="raw_dataset/processing")
    parser.add_argument(
        "--id_dict_path", type=str, default="raw_dataset/gene_id_conv_df.csv"
    )
    parser.add_argument("--version", type=str, default="digital_slide")
    args = parser.parse_args()

    return args


def find_correspond_samples(wsi_df, rna_df):
    corr_df = pd.DataFrame(
        columns=[
            "WSI File Name",
            "Sample ID",
            "Case ID",
            "RNA File Name",
            "Sample ID wo A",
            "RNA File Name wo A",
        ]
    )

    multi_rna_samples = []
    for index, row in wsi_df[wsi_df["Sample Type"] == "Primary Tumor"].iterrows():
        df_item = {}
        df_item["WSI File Name"] = row["File Name"]
        df_item["Case ID"] = row["Case ID"]
        df_item["Sample ID"] = row["Sample ID"]

        # extract corresponds rna
        sample_id = df_item["Sample ID"]
        rna_info = rna_df[rna_df["Sample ID"] == sample_id]
        if rna_info.shape[0] == 0:
            df_item["RNA File Name"] = None
        elif rna_info.shape[0] > 1:
            df_item["RNA File Name"] = rna_info["File Name"].values
            multi_rna_samples.append(sample_id)
        else:
            df_item["RNA File Name"] = rna_info["File Name"].item()

        # get rna even the sample
        df_item["Sample ID wo A"] = row["Sample ID"][:-1]
        rna_info = rna_df[rna_df["Sample ID"].str[:-1] == sample_id[:-1]]
        # rna_info = rna_df[rna_df["Sample ID"] == sample_id]
        if rna_info.shape[0] == 0:
            df_item["RNA File Name wo A"] = None
        elif rna_info.shape[0] > 1:
            df_item["RNA File Name wo A"] = rna_info["File Name"].values
        else:
            df_item["RNA File Name wo A"] = rna_info["File Name"].item()
        corr_df = pd.concat([corr_df, pd.DataFrame([df_item])], ignore_index=True)
    return corr_df


def obtain_rna_path(rna_paths, rna_name_list, row):
    multiple = False

    if row["RNA File Name"] is None:
        try:
            rna_idx = rna_name_list.index(row["RNA File Name wo A"])
            rna_path = rna_paths[rna_idx]
        except ValueError:
            rna_idx = rna_name_list.index(row["RNA File Name wo A"][0])
            rna_path = rna_paths[rna_idx]
            multiple = True
    else:
        try:
            rna_idx = rna_name_list.index(row["RNA File Name"])
            rna_path = rna_paths[rna_idx]
        except ValueError:
            rna_idx = rna_name_list.index(row["RNA File Name"][0])
            rna_path = rna_paths[rna_idx]
            multiple = True

    return (rna_path, multiple)


def save_correspond_files(corr_df, save_dir, source_dir):
    wsi_paths = sorted(Path(f"{source_dir}/WSI").glob("**/*.svs"))
    wsi_name_list = [path.name for path in wsi_paths]
    rna_paths = sorted(Path(f"{source_dir}/RNA").glob("**/*.tsv"))
    rna_name_list = [path.name for path in rna_paths]

    filtered_df = corr_df[~corr_df["RNA File Name wo A"].isna()]
    for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        wsi_idx = wsi_name_list.index(row["WSI File Name"])
        wsi_path = wsi_paths[wsi_idx]

        if "-DX" in wsi_path.stem:
            save_name = f"{row['Sample ID']}_{index}_DX"
        else:
            save_name = f"{row['Sample ID']}_{index}_TS"

        rna_path, multiple = obtain_rna_path(rna_paths, rna_name_list, row)

        if ~multiple:
            shutil.copyfile(wsi_path, f"{save_dir}/WSI/{save_name}.svs")
            shutil.copyfile(rna_path, f"{save_dir}/RNA/{save_name}.tsv")


def save_thumbnail(processed_dir):
    wsi_paths = sorted(Path(f"{processed_dir}/WSI").glob("*.svs"))
    save_dir = Path(f"{processed_dir}/thumbnail")

    for file_path in tqdm(wsi_paths):
        if Path(f"{save_dir}/{file_path.stem}.png").exists():
            continue
        img = openslide.OpenSlide(file_path)
        try:
            thumbnail = img.get_thumbnail((256, 256))
            thumbnail.save(f"{save_dir}/{file_path.stem}.png")
        except openslide.lowlevel.OpenSlideError:
            print(file_path)
            Path(f"{file_path.parent.parent}/RNA/{file_path.stem}.tsv").unlink()
            file_path.unlink()


def main(args):
    data_dir = Path(f"{args.source_dir}/TCGA-{args.dataset}")

    # make save directory
    save_dir = Path(f"{args.save_dir}/{args.dataset}/TCGA{args.version}")
    save_dir.joinpath("WSI").mkdir(parents=True, exist_ok=True)
    save_dir.joinpath("RNA").mkdir(parents=True, exist_ok=True)
    save_dir.joinpath("thumbnail").mkdir(parents=True, exist_ok=True)

    samp_sheets_dir = (
        f"{data_dir.parent}/samplesheets/gdc_{args.dataset}_sample_sheet.tsv"
    )
    # read dataset
    df = pd.read_csv(samp_sheets_dir, delimiter="\t")
    df = df[df["Sample Type"] == "Primary Tumor"]

    rna_df = df[df["Data Category"] == "Transcriptome Profiling"]
    wsi_df = df[df["Data Category"] == "Biospecimen"]

    corr_df = find_correspond_samples(wsi_df, rna_df)
    corr_df.to_csv(f"{save_dir}/WSI_RNA_corr_tab.csv")

    save_correspond_files(corr_df, save_dir, data_dir)

    save_thumbnail(save_dir)

    generate_bulkadata(save_dir, args.id_dict_path, "tpm")
    generate_bulkadata(save_dir, args.id_dict_path, "raw_count")

    adata = concat_rawexp_tpm(save_dir)
    adata.write_h5ad(f"{save_dir}/adata_bulk_concat.h5ad")


if __name__ == "__main__":
    args = get_parse()
    main(args)
