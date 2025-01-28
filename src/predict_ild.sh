#!/bin/bash

# List of cell types
cell_types=('A375' 'A549' 'HA1E' 'PC3' 'HBEC' 'CD34+' 'CD8+' 'PAD' 'HEK293T' 'HepG2')

# Iterate over each cell type
for cell in "${cell_types[@]}"; do
    echo "Running prediction for cell: $cell"
    python prediction.py --seed=42 --modz_path='/home/jovyan/project/platform-publication/benchmarking/transigen/data/modz_ild.pickle' --cell "$cell"

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred while running prediction for cell: $cell"
        exit 1
    fi
done

echo "All predictions completed successfully."
