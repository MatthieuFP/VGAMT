#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/text-only/en-de"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de"
EXP_NAME="XLM_nmt_en-de_OPUS"
EPOCH_SIZE=200000
seed=12345

WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=1234 \
--nproc_per_node=8 \
main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'de' --dropout '0.4' \
  --batch_size 8 --optimizer 'adam_inverse_sqrt,lr=0.005,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --num_workers 8 \
  --stopping_criterion 'valid_en-de_mt_bleu,50' --validation_metrics 'valid_en-de_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type xlm \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 32 $@