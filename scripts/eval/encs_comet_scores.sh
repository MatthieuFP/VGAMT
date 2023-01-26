#!/bin/bash

cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-cs/


path_models=(MBART_mix_MMT_VMLM_en-cs_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25_lr1e5_500_EPOCH_TRAINING_100minEpochs 
	MBART_mix_MT_MLM_en-cs_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_LONGER_TRAINING_100minEPOCHS 
	MBART_nmt_en-cs_OPUS)


for path in ${path_models[@]}; do
	echo ${path}
	path_flickr16_results=()
	path_flickr18_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			then path_flickr16_results+=(${dir}/hypothesis/hyp*test_2016_flickr.txt);
			path_flickr18_results+=(${dir}/hypothesis_test_2018_flickr/hyp*test_2018_flickr.txt);
		fi
	done
	ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
	ref_flickr18_dir=`dirname "${path_flickr18_results[0]}"`

	comet-score -s test_2016_flickr.en -t ${path_flickr16_results[@]} \
            -r ${ref_flickr16_dir}/ref.en-cs.test_2016_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr16_results.log

      comet-score -s test_2018_flickr.en -t ${path_flickr18_results[@]} \
            -r ${ref_flickr18_dir}/ref.en-cs.test_2018_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr18_results.log

done
wait



cd /gpfsstore/rech/ncm/ueh14bh/vtlm_exp/models_cs

path_models=(1303933_best_valid_en_cs_mlm_ppl_pth_ftune_nmt_over_tlm_bs256_lr0.00001 
	1304042_best_valid_en_cs_mlm_ppl_pth_ftune_mmt_vtlm_bs256_lr0.00001_AVG 
	nmt-from-scratch-multi30k)


for path in ${path_models[@]}; do
	echo ${path}
	path_flickr16_results=()
	path_flickr18_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			then path_flickr16_results+=(${dir}/hypotheses_test/hyp*detok);
			path_flickr18_results+=(${dir}/hypotheses_test_2018_flickr/hyp*detok);
		fi
	done
	ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
	ref_flickr18_dir=`dirname "${path_flickr18_results[0]}"`

	comet-score -s ${ref_flickr16_dir}/ref.cs-en.test.txt.detok -t ${path_flickr16_results[@]} \
            -r ${ref_flickr16_dir}/ref.en-cs.test.txt.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr16_results.log

    comet-score -s ${ref_flickr18_dir}/ref.cs-en.test_2018_flickr.txt.detok -t ${path_flickr18_results[@]} \
            -r ${ref_flickr18_dir}/ref.en-cs.test_2018_flickr.txt.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr18_results.log

done
wait

echo "GRAPH-MMT"
cd /gpfswork/rech/ncm/ueh14bh/GMNMT/decoding_mdetr/en-cs

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2016_flickr.en -t mdetr_graph_MMT_cs_*2016.cs.b4trans.detok \
            -r test_2016_flickr.lc.norm.tok.cs.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr16_results.log

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2018_flickr.en -t mdetr_graph_MMT_cs_*2018.cs.b4trans.detok \
            -r test_2018_flickr.lc.norm.tok.cs.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr18_results.log


echo "Gated Fusion"
cd /gpfswork/rech/ncm/ueh14bh/Revisit-MMT/results/hypotheses

comet-score -s CS_M30K16.EN.ref.detok -t CS_gated.en-cs.tiny_*M30K16.hyp.detok \
            -r CS_multi30k2016.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/CS_comet_flickr16_results.log

comet-score -s CS_M30K17.EN.ref.detok -t CS_gated.en-cs.tiny_*M30K17.hyp.detok \
            -r CS_multi30k2018.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/CS_comet_flickr18_results.log







