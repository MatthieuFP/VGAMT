# Download and preprocess data

### 1. Download data

```
cd ./data
```
#### Text-only data

Please download data from the [open parallel corpus](https://opus.nlpl.eu/). To reproduce our results, you must preprocess data in the following way:

```
git clone https://github.com/moses-smt/mosesdecoder.git
SRC=en
TGT={fr,de,cs}
sh scripts/text_only_preprocessing.sh ${SRC} ${TGT}
cd ./text_only/${SRC}-${TGT}
cat clean.${SRC} clean.${TGT} | shuf > data.${SRC}${TGT}
head -n -5000 data.${SRC}${TGT} > train.${SRC}${TGT}
tail -n 5000 data.${SRC}${TGT} > val.${SRC}${TGT}
```

#### Image-Text data

- **Multi30k** can be accessed [here](https://github.com/multi30k/dataset). Please request access to the images from the 2016 subset [here](https://forms.illinois.edu/sec/229675) and download the images from 2017 and 2018 subsets [here](https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt?usp=share_link).
- **CoMMuTE**: our contrastive evaluation dataset made of ambiguous examples can be downloaded from [here](https://github.com/MatthieuFP/CoMMuTE).
- **Conceptual Captions** 3M training and validation splits can be accessed [here](https://ai.google.com/research/ConceptualCaptions/download). You have to download the images using the provided url links:
```
cut -f1 -d$'\t' Train_GCC-training.tsv > train.en
cut -f2 -d$'\t' Train_GCC-training.tsv > url.txt
mkdir images features
index=0
touch train.order
cat url.txt | while read url; do 
    fname=$(printf "%09d\n" $index)
    echo ${fname}.jpg >> train.order; 
    wget ${url} -O images/${fname}.jpg; 
    index=$((index+1)); 
done
```

### 2. Compute CLIP and MDETR features

#### 2.1 CLIP 
To extract CLIP [CLS] embedding for all Multi30k and Conceptual Captions subsets, please run the following commands:

```
sh clip_features_extraction.sh multi30k {train,val,test_2016_flickr,test_2017_flickr,test_2018_flickr}
sh clip_features_extraction.sh conceptual_captions {train,val}
```

To extract CLIP [CLS] embedding for CoMMuTE, our contrastive evaluation dataset, please run the following commands:
```
DATASET=commute
SUBSET=en-{fr,de,cs}
BATCH_SIZE=100
source activate vgamt
python ./scripts/clip_features_extraction.py -i ./${DATASET}/images \
                                             -l ./${DATASET}/${SUBSET}/img.order \
                                             -d ./${DATASET}/${SUBSET}/features/clip_features \
                                             -b ${BATCH_SIZE}
```

#### 2.1 MDETR

Due to compatibility issues, you need to create a new conda environment to be able to use MDETR pre-trained models:
```
conda create -n mdetr_env python=3.8
conda activate mdetr_env
pip install -r requirements.txt
```

You then have to run the following scripts to extract MDETR features:
```
sh mdetr_features_extraction.sh multi30k {train,val,test_2016_flickr,test_2017_flickr,test_2018_flickr}
sh mdetr_features_extraction.sh conceptual_captions {train,val}
```

To extract MDETR features for CoMMuTE, please run the following commands:
```
DATASET=commute
SUBSET=en-{fr,de,cs}
source activate mdetr_env
python ./scripts/mdetr_features_extraction.py -i ./${DATASET}/images \
                                              -d ./${DATASET}/${SUBSET}/features/mdetr_features \
                                              -l ./${DATASET}/${SUBSET}/img.order \
                                              --text ./${DATASET}/${SUBSET}/src.en \
                                              --threshold 0.5
```

### References

- Jörg Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC'2012)


- Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. 2016. Multi30K: Multilingual EnglishGerman image descriptions. In Proceedings of the 5th Workshop on Vision and Language, pages 70– 74, Berlin, Germany. Association for Computational Linguistics.


- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, Melbourne, Australia. Association for Computational Linguistics.


- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR.


- Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. 2021. MDETR - modulated detection for end-to-end multi-modal understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV),
pages 1760–1770.
