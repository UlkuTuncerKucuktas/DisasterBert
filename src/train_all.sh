#!/bin/bash

# Base directory for datasets
BASE_DIR="/app/DisasterBert/events_set1"

# Ordered list of datasets (smallest to largest)
DATASETS=(
    "srilanka_floods_2017"           # 575 samples
    "italy_earthquake_aug_2016"      # 1,240 samples
    "ecuador_earthquake_2016"        # 1,594 samples
    "hurricane_matthew_2016"         # 1,659 samples
    "puebla_mexico_earthquake_2017"  # 2,036 samples
    "kaikoura_earthquake_2016"       # 2,217 samples
    "canada_wildfires_2016"          # 2,259 samples
    "cyclone_idai_2019"              # 3,944 samples
    "hurricane_maria_2017"           # 7,328 samples
    "hurricane_harvey_2017"          # 9,164 samples
    "hurricane_irma_2017"            # 9,467 samples
)

# Corresponding batch sizes (in same order as DATASETS)
BATCH_SIZES=(
    8   # srilanka_floods_2017
    8   # italy_earthquake_aug_2016
    8   # ecuador_earthquake_2016
    8   # hurricane_matthew_2016
    8   # puebla_mexico_earthquake_2017
    16  # kaikoura_earthquake_2016
    16  # canada_wildfires_2016
    16  # cyclone_idai_2019
    64  # hurricane_maria_2017
    64  # hurricane_harvey_2017
    64  # hurricane_irma_2017
)

# Training parameters
EPOCHS=15

# Create results directory if it doesn't exist
mkdir -p results

# Print header
echo "============================================="
echo "Starting training for all datasets"
echo "Ordered by dataset size (smallest to largest)"
echo "============================================="
echo ""

# Loop through each dataset
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    batch_size=${BATCH_SIZES[$i]}
    
    echo "============================================="
    echo "Running for $dataset"
    echo "Batch size: $batch_size"
    echo "============================================="
    
    # Run training
    python3 train.py \
        --dataset_dir "$BASE_DIR/$dataset" \
        --epochs $EPOCHS \
        --batch_size $batch_size
    
    echo "============================================="
    echo "Finished $dataset"
    echo "============================================="
    echo ""
done

echo "All training completed!" 