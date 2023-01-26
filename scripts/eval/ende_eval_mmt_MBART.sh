#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/multi30k"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de"
EXP_NAME="MBART_mmt_en-de"
FEAT_PATH_COCO="./data/multi30k/features/mdetr_features/mdetr_coco2017_features,./data/multi30k/features/clip_features/clip_mscoco2017_features"
FEAT_PATH_FLICKR17="./data/multi30k/features/mdetr_features/mdetr_flickr2017_features,./data/multi30k/features/clip_features/clip_flickr2017_features"
FEAT_PATH_FLICKR16="./data/multi30k/features/mdetr_features/mdetr_flickr2016_features,./data/multi30k/features/clip_features/clip_flickr2016_features"
EPOCH_SIZE=500000
seed=12345

WANDB_MODE=offline python main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH_COCO} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'de' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-de_mt_bleu,50' --validation_metrics 'valid_en-de_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/MBART_mix_MMT_VMLM_en-de_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr2e5_smaller_batch_100minEPOCHS/2103335/best-valid_en-de_mt_bleu.pth \
  --eval_only true --test_data_set test_2017_mscoco --multimodal_model --encoder_attn_mask_text_only --adapters --guided_self_attention $@

WANDB_MODE=offline python main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH_FLICKR17} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'de' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-de_mt_bleu,50' --validation_metrics 'valid_en-de_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/MBART_mix_MMT_VMLM_en-de_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr2e5_smaller_batch_100minEPOCHS/2103335/best-valid_en-de_mt_bleu.pth \
  --eval_only true --test_data_set test_2017_flickr --multimodal_model --encoder_attn_mask_text_only --adapters --guided_self_attention $@

WANDB_MODE=offline python main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH_FLICKR16} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'de' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam_inverse_sqrt,lr=0.0001,warmup_updates=2000' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 500 --max_len 80 --num_workers 4 \
  --stopping_criterion 'valid_en-de_mt_bleu,50' --validation_metrics 'valid_en-de_mt_bleu' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 16 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/MBART_mix_MMT_VMLM_en-de_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr2e5_smaller_batch_100minEPOCHS/2103335/best-valid_en-de_mt_bleu.pth \
  --eval_only true --multimodal_model --encoder_attn_mask_text_only --adapters --guided_self_attention $@