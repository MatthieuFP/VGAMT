#!/bin/bash

CACHE_HUGGINGFACE="/PATH/TO/CACHE_HUGGINGFACE"
DATA_PATH="/PATH/TO/DATA/FOLDER"
DUMP_PATH="/RESULTS/SAVING/PATH"
MODEL_PATH="/PATH/TO/MODEL"
SRC_LANG="en"
TGT_LANG="fr"

WANDB_MODE=offline python main.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size 50000 --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu,50" --validation_metrics "valid_${SRC_LANG}-${TGT_LANG}_mt_bleu" \
  --iter_seed 1 --other_seed 1 --save_periodic 1 --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --reload_model ${MODEL_PATH} \
  --eval_only true --test_data_set ${DATASET_NAME} $@