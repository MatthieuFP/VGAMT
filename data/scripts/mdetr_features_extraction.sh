#!/bin/bash

DATASET="$1"
SUBSET="$2"
source activate mdetr_env
python ./scripts/mdetr_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                              -d ./${DATASET}/features/mdetr_features \
                                              -l ./${DATASET}/${SUBSET}.order \
                                              --text ./${DATASET}/${SUBSET}.en \
                                              --threshold 0.5