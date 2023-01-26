#!/bin/bash

CACHE_HUGGINGFACE="/gpfsstore/rech/ncm/ueh14bh/huggingface"
DATA_PATH="./data/multi30k"
DATA_MIX_PATH="./data/conceptual_captions"
DUMP_PATH="/gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs"
EXP_NAME="MBART_mix_MT_MLM_en-cs_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_LONGER_TRAINING_100minEPOCHS"
EPOCH_SIZE=58000
seed=42

WANDB_MODE=offline NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--master_addr=${MASTER_ADDR} \
--nnodes=8 \
--node_rank=${SLURM_PROCID} \
--nproc_per_node=4 \
main.py --beam_size 4 --exp_name ${EXP_NAME} --dump_path ${DUMP_PATH} --data_path ${DATA_PATH} --data_mix_path ${DATA_MIX_PATH} \
  --src_lang 'en' --tgt_lang 'cs' --dropout '0.4' --batch_size 16 --optimizer 'adam,lr=0.0001' --epoch_size ${EPOCH_SIZE} \
  --eval_bleu true --max_epoch 5000 --max_len 80 --num_workers 0 --stopping_criterion 'valid_en-cs_mt_bleu,10' \
  --validation_metrics 'valid_en-cs_mt_bleu' --iter_seed ${seed} --other_seed ${seed} --smoothing 0.1 --save_periodic 1 \
  --model_type mbart --cache_dir ${CACHE_HUGGINGFACE} --amp 1 --fp16 True --accumulate_gradients 1 \
  --reload_model /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/MBART_nmt_en-cs_OPUS/826354/best-valid_en-cs_mt_bleu.pth \
  --adapters --freeze_text_parameters --start_new_xp_from_ckpt --mix_xp .5 --prob_mask_text 0.25 --encdec_vmlm --min_epoch 80 $@

# --prob_mask_text 0.5 --p_sample_multimodal_masking 0.25 --guided_self_attention