#!/bin/bash

DATASET="$1"
SUBSET="$2"
BATCH_SIZE=256
source activate vgamt
python ./scripts/clip_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                             -l ./${DATASET}/${SUBSET}.order \
                                             -d ./${DATASET}/features/clip_features \
                                             -b ${BATCH_SIZE}