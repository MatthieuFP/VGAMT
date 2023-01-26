#!/bin/bash

: '
cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-de/

path_models=(MBART_mix_MMT_VMLM_en-de_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr2e5_smaller_batch_100minEPOCHS 
	MBART_mix_MT_MLM_en-de_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_lr1e4_smaller_batch_100minEPOCHS 
	MBART_nmt_en-de_OPUS_smaller_batch)


for path in ${path_models[@]}; do
	echo ${path}
      path_flickr16_results=()
      path_flickr17_results=()
      path_mscoco17_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			then path_flickr16_results+=(${dir}/hypothesis/hyp*test_2016_flickr.txt);
			path_flickr17_results+=(${dir}/hypothesis_test_2017_flickr/hyp*test_2017_flickr.txt);
                  path_mscoco17_results+=(${dir}/hypothesis_test_2017_mscoco/hyp*test_2017_mscoco.txt);
		fi
	done
	ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
	ref_flickr17_dir=`dirname "${path_flickr17_results[0]}"`
      ref_mscoco17_dir=`dirname "${path_mscoco17_results[0]}"`

	comet-score -s test_2016_flickr.en -t ${path_flickr16_results[@]} \
            -r ${ref_flickr16_dir}/ref.en-de.test_2016_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr16_results.log

      comet-score -s test_2017_flickr.en -t ${path_flickr17_results[@]} \
            -r ${ref_flickr17_dir}/ref.en-de.test_2017_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr17_results.log

      comet-score -s test_2017_mscoco.en -t ${path_mscoco17_results[@]} \
            -r ${ref_mscoco17_dir}/ref.en-de.test_2017_mscoco* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_mscoco17_results.log

done
wait
'


cd /gpfsstore/rech/ncm/ueh14bh/vtlm_exp/models

path_models=(1591022_best_valid_en_de_mlm_ppl_pth_ftune_nmt_over_tlm_BPE10K_bs256_lr0.00001_LARGE_SIZE 
	1732755_best_valid_en_de_mlm_ppl_pth_ftune_mmt_vtlm_BPE10K_bs256_lr0.00001_AVG 
	nmt-from-scratch-multi30k_noam_BPE_10K)


for path in ${path_models[@]}; do
	echo ${path}
      #path_flickr16_results=()
      #path_flickr17_results=()
      path_mscoco17_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			#then path_flickr16_results+=(${dir}/hypotheses_test/hyp*detok.txt);
			#path_flickr17_results+=(${dir}/hypotheses_test_2017_flickr/hyp*test_2017_flickr.detok.txt);
                  then path_mscoco17_results+=(${dir}/hypotheses_test_2017_mscoco/hyp*test_2017_mscoco.detok.txt);
		fi
	done
	#ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
      #ref_flickr17_dir=`dirname "${path_flickr17_results[0]}"`
      ref_mscoco17_dir=`dirname "${path_mscoco17_results[0]}"`

	#comet-score -s ${ref_flickr16_dir}/ref.de-en.test.txt.detok -t ${path_flickr16_results[@]} \
      #      -r ${ref_flickr16_dir}/ref.en-de.test.detok.txt \
      #      --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
      #      --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
      #      --disable_cache > ${path}/comet_flickr16_results.log

      #comet-score -s ${ref_flickr17_dir}/ref.de-en.test_2017_flickr.txt.detok -t ${path_flickr17_results[@]} \
      #      -r ${ref_flickr17_dir}/ref.en-de.test_2017_flickr.detok.txt \
      #      --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
      #      --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
      #      --disable_cache > ${path}/comet_flickr17_results.log

      comet-score -s ${ref_mscoco17_dir}/ref.de-en.test_2017_mscoco.txt.detok -t ${path_mscoco17_results[@]} \
            -r ${ref_mscoco17_dir}/ref.en-de.test_2017_mscoco.detok.txt \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_mscoco17_results.log

done
wait

echo "GRAPH-MMT"
cd /gpfswork/rech/ncm/ueh14bh/GMNMT/decoding_mdetr/en-de

: '
comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2016_flickr.en -t mdetr_boxfeat_new_params_*2016.de.b4trans.detok \
            -r test_2016_flickr.lc.norm.tok.de.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr16_results.log

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2017_flickr.en -t mdetr_boxfeat_new_params_*2017.de.b4trans.detok \
            -r test_2017_flickr.lc.norm.tok.de.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr17_results.log
'

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2017_mscoco.en -t mdetr_boxfeat_new_params_*2017coco.b4trans.detok \
            -r test_2017_mscoco.lc.norm.tok.de.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_mscoco17_results.log


: '
echo "Gated Fusion"
cd /gpfswork/rech/ncm/ueh14bh/Revisit-MMT/results/hypotheses

comet-score -s DE_M30K16.EN.ref.detok -t DE_gated.en-de.tiny_*M30K16.hyp.detok \
            -r DE_multi30k2016.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/DE_comet_flickr16_results.log

comet-score -s DE_M30K17.EN.ref.detok -t DE_gated.en-de.tiny_*M30K17.hyp.detok \
            -r DE_multi30k2017.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/DE_comet_flickr17_results.log

comet-score -s DE_COCO17.EN.ref.detok -t DE_gated.en-de.tiny_*COCO17.hyp.detok \
            -r DE_mscoco2017.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/DE_comet_mscoco17_results.log
'


