#!/bin/bash

METEOR_FILE="/PATH/TO/METEOR/"
REFERENCE_PATH="/PATH/TO/REF/FILE"
HYPOTHESIS_PATH="/PATH/TO/HYP/FILE"

meteor_eval() {
    java -Xmx2G -jar ${METEOR_FILE} "$1" "$2" -l fr > meteor_score.log
}

meteor_eval ${REFERENCE_PATH} ${HYPOTHESIS_PATH}