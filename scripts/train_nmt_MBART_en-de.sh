#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/text-only/en-de"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de"
EXP_NAME="MBART_nmt_en-de_OPUS_smaller_batch"
EPOCH_SIZE=500000
seed=12345

WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=1234 \
--master_addr=${MASTER_ADDR} \
--nnodes=8 \
--node_rank=${SLURM_PROCID} \
--nproc_per_node=4 \
main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'de' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 60 --num_workers 0 \
  --stopping_criterion 'valid_en-de_mt_bleu,50' --validation_metrics 'valid_en-de_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 1 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/MBART_nmt_en-de_OPUS_smaller_batch/1222488/periodic-415.pth \
  --reload_optim $@