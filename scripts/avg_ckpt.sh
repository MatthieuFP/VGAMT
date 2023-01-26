#!/bin/bash

CKPT=$1

python ./src/average_checkpoints.py \
--input ${CKPT} \
--output checkpoint_avg_5before.pth \