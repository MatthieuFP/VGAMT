#!/bin/bash

CACHE_HUGGINGFACE="/PATH/TO/CACHE_HUGGINGFACE"
DATA_PATH="/PATH/TO/TEXT_ONLY_DATA/FOLDER"
DUMP_PATH="/RESULTS/SAVING/PATH"
EXP_NAME="EXPERIENCE-NAME"
EPOCH_SIZE=500000
SRC_LANG="en"
TGT_LANG="fr"
seed="SEED"

WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=1234 \
--master_addr=${MASTER_ADDR} \
--nnodes=2 \
--node_rank=${SLURM_PROCID} \
--nproc_per_node=4 \
main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 0 \
  --stopping_criterion 'valid_${SRC_LANG}-${TGT_LANG}_mt_bleu,50' --validation_metrics 'valid_${SRC_LANG}-${TGT_LANG}_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 4 $@