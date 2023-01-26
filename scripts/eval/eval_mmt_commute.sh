#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/commute/homograph"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr"
FEAT_PATH="./data/commute/homograph/clip_features/standard,./data/commute/homograph/mdetr_features/array_format/standard"
seed=12345

WANDB_MODE=offline python main.py --beam_size 8 --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'fr' --dropout '0.4' --batch_size 1 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --max_epoch 500 --max_len 80 --num_workers 4 --stopping_criterion 'valid_en-fr_mt_bleu,50' --validation_metrics 'valid_en-fr_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.0 --save_periodic 1 --model_type mbart --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 1 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr/MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask15/1227232/best-valid_en-fr_mt_bleu.pth \
  --eval_only true --multimodal_model --encoder_attn_mask_text_only --adapters --test_data_set commute --guided_self_attention $@

# --guided_self_attention --encoder_attn_mask_temxt_only --adapters