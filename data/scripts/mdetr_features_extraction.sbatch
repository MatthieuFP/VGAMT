#!/bin/bash

DATASET=multi30k
SUBSET=test_2018_flickr
source activate mdetr_env
python ./scripts/mdetr_features_extraction.py -i ./${DATASET}/images/${SUBSET} \
                                              -d ./${DATASET}/features/mdetr_features/${SUBSET}_features \
                                              -l ./${DATASET}/${SUBSET}.order \
                                              --text ./${DATASET}/${SUBSET}.en