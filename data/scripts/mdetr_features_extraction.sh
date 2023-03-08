#!/bin/bash

DATASET="$1"
SUBSET="$2"
TORCH_HUB_DIR="/PATH/TO/HUB/DIR"
python ./scripts/mdetr_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                              -d ./${DATASET}/features/mdetr_features/${SUBSET} \
                                              -l ./${DATASET}/${SUBSET}.order \
                                              --text ./${DATASET}/${SUBSET}.en \
                                              --threshold 0.5 \
                                              --hub_dir ${TORCH_HUB_DIR}