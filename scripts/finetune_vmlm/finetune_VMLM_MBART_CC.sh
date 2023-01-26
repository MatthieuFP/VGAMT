#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/conceptual_captions"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr"
FEAT_PATH="./data/conceptual_captions/features/mdetr_features,./data/conceptual_captions/features/clip_features"
EXP_NAME="MBART_VMLM_en-fr_CC_from_NMT_adapters_textparamsfrozen_guidedSA_PMask50_CLIPMDETR_encoder_decoder"
EPOCH_SIZE=2000000
seed=12345

WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--master_addr=${MASTER_ADDR} \
--nnodes=8 \
--node_rank=${SLURM_PROCID} \
--nproc_per_node=4 \
main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --features_path ${FEAT_PATH} --features_type 'mdetr+clip' \
  --data_path ${DATA_PATH} --src_lang 'en' --tgt_lang 'fr' --dropout '0.4' \
  --batch_size 16 --optimizer 'adam,lr=0.0001' \
  --epoch_size ${EPOCH_SIZE} --eval_bleu true --max_epoch 5000 --max_len 80 --num_workers 0 \
  --stopping_criterion 'valid_en_vmlm_acc,20' --validation_metrics 'valid_en_vmlm_acc' \
  --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 --model_type mbart \
  --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 1 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr/MBART_nmt_en-fr_OPUS/2006603/best-valid_en-fr_mt_bleu.pth \
  --multimodal_model --adapters --freeze_text_parameters --guided_self_attention --prob_mask_text 0.5 --encdec_vmlm --encoder_attn_mask_text_only \
  --start_new_xp_from_ckpt $@

# --prob_mask_text 0.5 --p_sample_multimodal_masking 0.25 --guided_self_attention --start_new_xp_from_ckpt