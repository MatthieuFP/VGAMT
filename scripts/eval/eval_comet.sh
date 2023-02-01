#!/bin/bash

REFERENCE_SRC_LG="/PATH/TO/REF/FILE/SRC/LANG"
REFERENCE_TGT_LG="PATH/TO/REF/FILE/TGT/LANG"
HYPOTHESIS_TGT_LG="PATH/TO/GENERATED/TEXT/TO/EVAL"
PATH_TO_COMET_STORAGE="PATH/TO/COMET/STORAGE"

comet-score -s ${REFERENCE_SRC_LG} -t ${HYPOTHESIS_TGT_LG} -r ${REFERENCE_TGT_LG} \
            --model ${PATH_TO_COMET_STORAGE}/checkpoints/model.ckpt \
            --model_storage_path ${PATH_TO_COMET_STORAGE} \
            --disable_cache > comet.log