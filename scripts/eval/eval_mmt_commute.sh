#!/bin/bash

CACHE_HUGGINGFACE="/PATH/TO/CACHE_HUGGINGFACE"
DATA_PATH="/PATH/TO/COMMUTE/FOLDER"
DUMP_PATH="/RESULTS/SAVING/PATH"
FEAT_PATH="/PATH/TO/MDETR/FEATURES,/PATH/TO/CLIP/FEATURES"
MODEL_PATH="/PATH/TO/MODEL"
SRC_LANG="en"
TGT_LANG="fr"

WANDB_MODE=offline python main.py --beam_size 8 --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} --batch_size 1 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --max_epoch 5000 --max_len 80 --num_workers 4 --stopping_criterion 'valid_${SRC_LANG}-${TGT_LANG}_mt_bleu,50' --validation_metrics 'valid_${SRC_LANG}-${TGT_LANG}_mt_bleu' \
  --iter_seed 1 --other_seed 1 --save_periodic 1 --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True \
  --reload_model ${MODEL_PATH} --eval_only true --multimodal_model --encoder_attn_mask_text_only --adapters --test_data_set commute --guided_self_attention $@