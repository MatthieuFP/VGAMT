#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/multi30k"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs"
EXP_NAME="MBART_nmt_en-cs_OPUS"
EPOCH_SIZE=500000
seed=12345

WANDB_MODE=offline python main.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'cs' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-cs_mt_bleu,50' --validation_metrics 'valid_en-fr_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/MBART_nmt_en-cs_OPUS/826354/best-valid_en-cs_mt_bleu.pth \
  --eval_only true --test_data_set test_2018_flickr $@

WANDB_MODE=offline python main.py --beam_size 8 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'cs' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-cs_mt_bleu,50' --validation_metrics 'valid_en-cs_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/MBART_nmt_en-cs_OPUS/826354/best-valid_en-cs_mt_bleu.pth \
  --eval_only true $@

# --adapters