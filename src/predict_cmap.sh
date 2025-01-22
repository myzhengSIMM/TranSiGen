#!/bin/bash

# List of cell types
cell_types=('A375' 'A549' 'HA1E' 'HCC515' 'HEPG2' 'HT29' 'MCF7' 'PC3' 'VCAP')

# Iterate over each cell type
for cell in "${cell_types[@]}"; do
    echo "Running prediction for cell: $cell"
    python prediction.py --cell "$cell"

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error occurred while running prediction for cell: $cell"
        exit 1
    fi
done

echo "All predictions completed successfully."
