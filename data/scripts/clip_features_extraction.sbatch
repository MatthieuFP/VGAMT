#!/bin/bash

DATASET=multi30k
SUBSET=test_2018_flickr
BATCH_SIZE=128
source activate vgamt
python ./scripts/clip_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                             -l ./${DATASET}/${SUBSET}.order \
                                             -d ./${DATASET}/features/clip_features/${SUBSET}_features \
                                             -b ${BATCH_SIZE}