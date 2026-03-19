import pandas as pd
import anndata as ad
import os

# Define the path to the data directory
data_dir = '/home/hdd/kazuya/digital_slide/raw_dataset/sc_raw/CSCC'

# List of count files
count_files = [
    'GSE144236_CAL27_counts.txt.gz',
    'GSE144236_CAL27_vitro_counts.txt.gz',
    'GSE144236_SCC13_counts.txt.gz',
    'GSE144236_XG_TME_counts.txt.gz',
    'GSE144236_cSCC_counts.txt.gz'
]

# Read and concatenate all count matrices
all_counts = []
for f in count_files:
    filepath = os.path.join(data_dir, f)
    print(f"Reading {filepath}...")
    # Read the data, using the first column as the index
    counts = pd.read_csv(filepath, sep='\t', index_col=0)
    all_counts.append(counts)

# Concatenate along the columns (axis=1)
print("Concatenating count matrices...")
combined_counts = pd.concat(all_counts, axis=1)

# Transpose the matrix so that cells are rows and genes are columns
combined_counts = combined_counts.T

# Read the metadata
metadata_file = os.path.join(data_dir, 'GSE144236_patient_metadata_new.txt.gz')
print(f"Reading metadata from {metadata_file}...")
metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)

# Align the metadata with the count data
# This will ensure that the rows in the metadata correspond to the rows in the count matrix
aligned_metadata = metadata.loc[combined_counts.index]

# Create the AnnData object
print("Creating AnnData object...")
adata = ad.AnnData(X=combined_counts, obs=aligned_metadata)

# Save the AnnData object
output_file = os.path.join(data_dir, 'combined_sc_data.h5ad')
print(f"Saving AnnData object to {output_file}...")
adata.write(output_file)

print("Done.")
