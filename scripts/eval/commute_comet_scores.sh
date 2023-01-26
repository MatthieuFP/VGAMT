#!/bin/bash
cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr/

path_models=(MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25 
      MBART_mix_MT_MLM_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25)


for path in ${path_models[@]}; do
      echo ${path}
      path_commute_results=()
      for dir in ${path}/*; do
            if [[ -d ${dir} ]]
                  then path_commute_results+=(${dir}/hypothesis_commute_generation/hyp*);
            fi
      done
      ref_commute_dir=`dirname "${path_commute_results[0]}"`

      comet-score -s commute.en -t ${path_commute_results[@]} \
            -r ${ref_commute_dir}/ref.en-fr.commute_generation* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_commute_results.log

done
wait


cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/

path_models=(MBART_mix_MMT_VMLM_en-de_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr2e5_smaller_batch_100minEPOCHS 
      MBART_mix_MT_MLM_en-de_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_lr1e4_smaller_batch_100minEPOCHS)


for path in ${path_models[@]}; do
      echo ${path}
      path_commute_results=()
      for dir in ${path}/*; do
            if [[ -d ${dir} ]]
                  then path_commute_results+=(${dir}/hypothesis_commute_generation/hyp*);
            fi
      done
      ref_commute_dir=`dirname "${path_commute_results[0]}"`

      comet-score -s commute.en -t ${path_commute_results[@]} \
            -r ${ref_commute_dir}/ref.en-de.commute_generation* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_commute_results.log

done
wait

cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/

path_models=(MBART_mix_MMT_VMLM_en-cs_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr1e5_500_EPOCH_TRAINING_100minEpochs 
      MBART_mix_MT_MLM_en-cs_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_LONGER_TRAINING_100minEPOCHS)


for path in ${path_models[@]}; do
      echo ${path}
      path_commute_results=()
      for dir in ${path}/*; do
            if [[ -d ${dir} ]]
                  then path_commute_results+=(${dir}/hypothesis_commute_generation/hyp*);
            fi
      done
      ref_commute_dir=`dirname "${path_commute_results[0]}"`

      comet-score -s commute.en -t ${path_commute_results[@]} \
            -r ${ref_commute_dir}/ref.en-cs.commute_generation* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_commute_results.log

done
wait

