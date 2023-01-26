#!/bin/bash

meteor_eval() {
    java -Xmx2G -jar ../../../../../meteor-1.5/meteor-1.5.jar "$1" "$2" -l de > best_model_meteor_score.log
}

cd $STORE/guided_sa/en-de

path="MBART_mix_MT_MLM_en-de_Multi30k_from_NMT_adapters_textparamsfrozen_mix50_PMask25_lr1e4_smaller_batch_100minEPOCHS"

for dir in ${path}/*; do
	if [ -d ${dir} ]; then
		echo ${dir}
		cd ${dir}
		cd hypothesis_test_2017_flickr; meteor_eval ref.en-de.test_2017_flickr* hyp*
		cd ../hypothesis_test_2017_mscoco; meteor_eval ref.en-de.test_2017_mscoco* hyp*
		cd ../hypothesis; meteor_eval ref.en-de.test_2016_flickr* *test_2016_flickr.txt
		cd ../../..;
	fi
done
wait