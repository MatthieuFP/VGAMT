#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/commute/homograph"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs"
seed=12345

WANDB_MODE=offline python main.py --beam_size 8 --dump_path ${DUMP_PATH} --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'cs' --dropout '0.4' \
  --batch_size 1 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-cs_mt_bleu,50' --validation_metrics 'valid_en-cs_mt_bleu' --iter_seed ${seed} --other_seed ${seed} --smoothing 0.0 \
  --save_periodic 1 --model_type mbart --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/MBART_mix_MT_MLM_en-cs_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_LONGER_TRAINING_100minEPOCHS/1819042/best-valid_en-cs_mt_bleu.pth \
  --eval_only true --test_data_set commute --adapters --commute_generation $@