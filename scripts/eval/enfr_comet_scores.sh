#!/bin/bash

: '
cd /gpfsstore/rech/ncm/ueh14bh/guided_sa/en-fr/

path_models=(MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_CLIPMDETR_mix50_PMask25 
	MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_MDETR_mix50_PMask25 
	MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen_CLIPMDETR_mix50_PMask25_NOGUIDEDSA 
	MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen_CLIP_mix50_PMask25 
	MBART_mix_MT_MLM_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25 
	MBART_MMT_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen_guidedSA_CLIPMDETR 
	MBART_MMT_en-fr_Multi30k_from_VMLM_CC_share_lm_head_adapters_textparamsfrozen_guidedSA_CLIPMDETR_lr1e4 
	MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_guidedSA_CLIPMDETR_mix50_PMask50 
	MBART_NMT_en-fr_Multi30k_from_NMT_adapters_textparamsfrozen 
	MBART_nmt_en-fr_OPUS)


for path in ${path_models[@]}; do
	echo ${path}
	path_flickr16_results=()
	path_flickr17_results=()
	path_mscoco17_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			then path_flickr16_results+=(${dir}/hypothesis/hyp*test_2016_flickr.txt);
			path_flickr17_results+=(${dir}/hypothesis_test_2017_flickr/hyp*test_2017_flickr.txt);
			path_mscoco17_results+=(${dir}/hypothesis_test_2017_mscoco/hyp*test_2017_mscoco.txt)
		fi
	done
	ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
	ref_flickr17_dir=`dirname "${path_flickr17_results[0]}"`
	ref_mscoco17_dir=`dirname "${path_mscoco17_results[0]}"`

	comet-score -s test_2016_flickr.en -t ${path_flickr16_results[@]} \
            -r ${ref_flickr16_dir}/ref.en-fr.test_2016_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr16_results.log

    comet-score -s test_2017_flickr.en -t ${path_flickr17_results[@]} \
            -r ${ref_flickr17_dir}/ref.en-fr.test_2017_flickr* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr17_results.log

    comet-score -s test_2017_mscoco.en -t ${path_mscoco17_results[@]} \
            -r ${ref_mscoco17_dir}/ref.en-fr.test_2017_mscoco* \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_mscoco17_results.log

done
wait
'

cd /gpfsstore/rech/ncm/ueh14bh/vtlm_exp/models_fr

path_models=(532848_best_valid_en_fr_mlm_ppl_pth_ftune_nmt_over_tlm_BPE10K_bs256_lr0.00001 
	538950_best_valid_en_fr_mlm_ppl_pth_ftune_mmt_vtlm_BPE10K_bs256_lr0.00001_AVG 
	nmt-from-scratch-multi30k_BPE_10K)


for path in ${path_models[@]}; do
	echo ${path}
    path_flickr16_results=()
    path_flickr17_results=()
    path_mscoco17_results=()
	for dir in ${path}/*; do
		if [[ -d ${dir} ]]
			then path_flickr16_results+=(${dir}/hypotheses_test/hyp*txt.detok);
			path_flickr17_results+=(${dir}/hypotheses_test_2017_flickr/hyp*test_2017_flickr.txt.detok);
            path_mscoco17_results+=(${dir}/hypotheses_test_2017_mscoco/hyp*test_2017_mscoco.txt.detok);
		fi
	done
	ref_flickr16_dir=`dirname "${path_flickr16_results[0]}"`
    ref_flickr17_dir=`dirname "${path_flickr17_results[0]}"`
    ref_mscoco17_dir=`dirname "${path_mscoco17_results[0]}"`

	comet-score -s ${ref_flickr16_dir}/ref.fr-en.test.txt.detok -t ${path_flickr16_results[@]} \
            -r ${ref_flickr16_dir}/ref.en-fr.test.txt.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr16_results.log

    comet-score -s ${ref_flickr17_dir}/ref.fr-en.test_2017_flickr.txt.detok -t ${path_flickr17_results[@]} \
            -r ${ref_flickr17_dir}/ref.en-fr.test_2017_flickr.txt.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_flickr17_results.log

    comet-score -s ${ref_mscoco17_dir}/ref.fr-en.test_2017_mscoco.txt.detok -t ${path_mscoco17_results[@]} \
            -r ${ref_mscoco17_dir}/ref.en-fr.test_2017_mscoco.txt.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > ${path}/comet_mscoco17_results.log

done
wait


: '
echo "GRAPH-MMT"
cd /gpfswork/rech/ncm/ueh14bh/GMNMT/decoding_mdetr/en-fr

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2016_flickr.en -t mdetr_graph_MMT_fr_*2016.fr.b4trans.detok \
            -r test_2016_flickr.lc.norm.tok.fr.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr16_results.log

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2017_flickr.en -t mdetr_graph_MMT_fr_*2017.fr.b4trans.detok \
            -r test_2017_flickr.lc.norm.tok.fr.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_flickr17_results.log

comet-score -s /gpfsscratch/rech/ncm/ueh14bh/data/multi30k/test_2017_mscoco.en -t mdetr_graph_MMT_fr_*2017coco.b4trans.detok \
            -r test_2017_mscoco.lc.norm.tok.fr.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/comet_mscoco17_results.log
'

echo "Gated Fusion"
cd /gpfswork/rech/ncm/ueh14bh/Revisit-MMT/results/hypotheses

: '
comet-score -s FR_M30K16.EN.ref.detok -t FR_gated.en-fr.tiny_*M30K16.hyp.detok \
            -r FR_multi30k2016.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/FR_comet_flickr16_results.log

comet-score -s FR_M30K17.EN.ref.detok -t FR_gated.en-fr.tiny_*M30K17.hyp.detok \
            -r FR_multi30k2017.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/FR_comet_flickr17_results.log
'
comet-score -s FR_COCO17.EN.ref.detok -t FR_gated.en-fr.tiny_*COCO17.hyp.detok \
            -r FR_mscoco17.ref.detok \
            --model /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da/checkpoints/model.ckpt \
            --model_storage_path /gpfsstore/rech/ncm/ueh14bh/comet/wmt20-comet-da \
            --disable_cache > comet_scores/FR_comet_mscoco17_results.log



