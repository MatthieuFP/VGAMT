#!/bin/bash

meteor_eval() {
    java -Xmx2G -jar ../../../../../meteor-1.5/meteor-1.5.jar "$1" "$2" -l fr > best_model_meteor_score.log
}

cd $STORE/guided_sa/en-fr

path="MBART_mix_MMT_VMLM_en-fr_Multi30k_from_NMT_adapters_guidedSA_textparamsfrozen_MDETR_mix50_PMask25"

for dir in ${path}/*; do
	if [ -d ${dir} ]; then
		echo ${dir}
		cd ${dir}
		cd hypothesis_test_2017_flickr; meteor_eval ref.en-fr.test_2017_flickr* hyp*
		cd ../hypothesis_test_2017_mscoco; meteor_eval ref.en-fr.test_2017_mscoco* hyp*
		cd ../hypothesis; meteor_eval ref.en-fr.test_2016_flickr* *test_2016_flickr.txt
		cd ../../..;
	fi
done
wait