# Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Whole Slide Image
[Kazuya Nishimura](https://naivete5656.github.io/index-e.html), [Ryoma Bise](https://human.ait.kyushu-u.ac.jp/~bise/index-en.html), Shinnosuke Matsuo, Haruka Hirose, [Yasuhiro Kojima](https://researchmap.jp/yskjm?lang=en)

## Environment

```
bash ./docker_mamba build
```

## Dataset generation

<details>
<summary>Processing of TCGA</summary>

## Download TCGA dataset and scdataset

<details>
<summary>Download instructions</summary>

## Download TCGA datasets
[Used GDC data transfer tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

Download manifest file and sample sheets from following projects.
[[BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)]
[[KIRC](https://portal.gdc.cancer.gov/projects/TCGA-KIRC)]
[[LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD)]

- [ ] upload sample_sheet, manifestfile to github


```
cd ./raw_dataset/TCGA_raw
mkdir ./TCGA-${DATASET}/WSI
mkdir ./TCGA-${DATASET}/RNA
./gdc-client download -m ./manifests/gdc_${DATASET}_manifest_wsi.txt -d ./TCGA-${DATASET}/WSI
./gdc-client download -m ./manifests/gdc_${DATASET}_manifest_rna.txt -d ./TCGA-${DATASET}/RNA
```

## Download single-cell datasets
- Breast cencer dataset (BRCA) [[URL](https://singlecell.broadinstitute.org/single_cell/study/SCP1039/a-single-cell-and-spatially-resolved-atlas-of-human-breast-cancers?hiddenTraces=CAFs%20MSC%20iCAF-like%20s2#study-visualize)]
- RCC dataset (KIRC) [[URL](https://data.mendeley.com/datasets/g67bkbnhhg/1)]
- Lung Cancer (LUAD) [[URL](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907)]

Folder structure
```
raw_dataset
├── TCGA_raw
|   ├── manifests
|   ├── samplesheets
|   ...
|   L TCGA-LUAD
└── sc_raw
    ├── BRCA
    ...
    L LUAD
```
</details>

## Step 1: Processing whole slide image

<details>
<summary>Processing instructions</summary>

### Find matched whole slide image and gene expression

```
python preprocessing/1-processing_TCGA.py --dataset ${DATASET} --source_dir ./raw_dataset/TCGA_raw --save_dir ./raw_dataset/processing --id_dict_path ./raw_dataset/gene_id_conv_df.csv
```

### Feature extraction by [CLAM](https://github.com/mahmoodlab/CLAM) from whole slide image
Please refer CLAM repository to use following command
```
git clone https://github.com/mahmoodlab/CLAM.git
```
For the feature extraction by DINO v2
Please add following script in the line 78 of CLAM/models/builder.py
```
elif model_name == "dinov2":
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    data_config = timm.data.resolve_model_data_config(model)
    img_transforms = timm.data.create_transform(**data_config, is_training=False)
```

```
cd CLAM

DATASET=LUAD
python create_patches_fp.py --source ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/WSI --save_dir ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/proc_wsi4mil --patch_size 256 --preset tcga.csv --seg --patch --stitch

python extract_features_fp.py --data_h5_dir ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/proc_wsi4mil --data_slide_dir ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/WSI --csv_path ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/proc_wsi4mil/process_list_autogen.csv --feat_dir ../raw_dataset/processing/${DATASET}/TCGA-digital_slide/feature --batch_size 512 --slide_ext .svs --model_name dinov2
```
</details>

## Step 2: Processing single-cell dataset
```
python preprocessing/2-sc_processing.py --dataset ${DATASET} --save_dir ./raw_dataset/processing --sc_raw_data_dir ./raw_dataset/sc_raw
```


### Step 3: Save feature and gene expression with h5 format
```
python preprocessing/3-generate_pair.py --dataset ${DATASET} --base_dir ./raw_dataset/processing --save_dir ./dataset --feat_name feature
```

### Step 4: Run single-cell deconvolution method 
```
for fold in {1..5}
do 
    python preprocessing/4-cell2location_proc.py --dataset ${DATASET}  --save_dir dataset \
    --base_dir ./raw_dataset/processing --resolution fine --fold ${fold}
done
```

</details>






## Overview

<!-- overview.png will be added here -->

ProtoSum is a cell-type prototype-informed neural network that estimates gene expression profiles from whole slide images (WSI). It leverages single-cell RNA-seq data to construct cell-type prototypes and uses them to guide gene expression prediction from histology patches.

## Training

Experiment scripts are provided in the `scripts/` directory.

### Training code

```bash
# Single fold
python main.py --method ProtoSum --dataset BRCA --fold 0 \
    --trainer DeconvExp --version "1reg_mse_reg_1e3" --data_type ts

# All folds across all datasets
bash scripts/ours.sh
```

### Ablation study

```bash
bash scripts/ours_granularity.sh
bash scripts/ours_ablation.sh
```

### Comparison methods

Scripts for all comparison methods are in `scripts/comparisons/`.

| Script | Method | Trainer |
|--------|--------|---------|
| `MIL_abmil.sh` | AbMIL | — |
| `MIL_max.sh` | AbMIL (max pooling) | — |
| `MIL_mean.sh` | AbMIL (mean pooling) | — |
| `MIL_ILRA.sh` | ILRA | — |
| `MIL_mambamil.sh` | MambaMIL | — |
| `MIL_srmamba.sh` | SRMambaMIL | — |
| `MIL_2DMambaMIL.sh` | MambaMIL_2D | Mamba2DTrainer |
| `MIL_s4model.sh` | S4Model | — |
| `MIR_HE2RNA.sh` | HE2RNA | — |
| `MIR_tRNAformer.sh` | tRNAsformer | — |
| `MIR_SEQUOIA.sh` | SEQUOIA | — |
| `MIR_MOSBY.sh` | MOSBY (SumExpModel) | — |
| `MIR_abreg.sh` | AbRegMIL | — |

Example:
```bash
bash scripts/comparisons/MIR_HE2RNA.sh
```

## Inference

```bash
python inference.py --method ProtoSum --dataset BRCA --fold 0 \
    --trainer DeconvExp --version "1reg_mse_reg_1e3" --data_type ts
```

## Citation

```bibtex
@inproceedings{nishimura2025protosum,
  title={Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Whole Slide Image},
  author={Nishimura, Kazuya and Bise, Ryoma and Matsuo, Shinnosuke and Hirose, Haruka and Kojima, Yasuhiro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledge
Our code refer following works:
- [TRIPLEX](https://github.com/NEXGEM/TRIPLEX).
- [Cell2location](https://github.com/maxpmx/HisToGene), licensed under the [Apache License Version 2.0](https://github.com/BayraktarLab/cell2location/tree/master?tab=Apache-2.0-1-ov-file).
- [SCVI](https://github.com/scverse/scvi-tools), licensed under the [BSD-3-Clause License](https://github.com/scverse/scvi-tools?tab=BSD-3-Clause-1-ov-file)