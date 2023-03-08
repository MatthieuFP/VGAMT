# Download and preprocess data

### 1. Download data

```
cd ./data
```

#### Text-only data

Please download data from the [open parallel corpus](https://opus.nlpl.eu/) to ./text-only/${SRC}-${TGT} folders. We used OpenSubtitles_v2018, Books_v1 (not available in cs), Wikipedia_v1.0, TED2020_v1 and TED2013_v1.1 (not available in cs) for our experiments. To reproduce our results, you must preprocess data in the following way (WARNING! It could be long):

```
git clone https://github.com/moses-smt/mosesdecoder.git
SRC=en
TGT={fr,de,cs}
sh scripts/text_only_preprocessing.sh ${SRC} ${TGT}
cd ./text_only/${SRC}-${TGT}
touch data.${SRC} data.${TGT}
for file in */clean.${SRC}; do cat ${file} >> data.${SRC}; done
for file in */clean.${TGT}; do cat ${file} >> data.${TGT}; done
paste data.${SRC} data.${TGT} | shuf > data.${SRC}${TGT}
head -n -10000 data.${SRC}${TGT} > train.${SRC}${TGT}
tail -n 10000 data.${SRC}${TGT} > testval.${SRC}${TGT}
head -n -5000 testval.${SRC}${TGT} > test.${SRC}${TGT}
tail -n 5000 testval.${SRC}${TGT} > val.${SRC}${TGT}
rm testval.${SRC}${TGT}
```

#### Image-Text data

- **Multi30k** can be accessed [here](https://github.com/multi30k/dataset). Please request access to the images from the 2016 subset [here](https://forms.illinois.edu/sec/229675) and download the images from 2017 and 2018 subsets [here](https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt?usp=share_link).
- **CoMMuTE**: our contrastive evaluation dataset made of ambiguous examples can be downloaded from [here](https://github.com/MatthieuFP/CoMMuTE).
- **Conceptual Captions** 3M training and validation splits can be accessed [here](https://ai.google.com/research/ConceptualCaptions/download). You have to download the images using the provided url links:
```
cd conceptual_captions
cut -f1 -d$'\t' Train_GCC-training.tsv > train.en
cut -f2 -d$'\t' Train_GCC-training.tsv > url.txt
mkdir images features
mkdir images/train
index=0
touch train.order
cat url.txt | while read url; do 
    fname=$(printf "%09d\n" $index)
    echo ${fname}.jpg >> train.order; 
    wget ${url} -O images/train/${fname}.jpg; 
    index=$((index+1)); 
done
```

### 2. Compute CLIP and MDETR features

#### 2.1 CLIP 

Create a new conda environment to run CLIP:
```
conda create -n clip python=3.8
conda activate clip
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

To extract CLIP [CLS] embedding for all Multi30k and Conceptual Captions subsets, please run the following commands:

```
source activate clip
sh ./scripts/clip_features_extraction.sh multi30k {train,val,test_2016_flickr,test_2017_flickr,test_2018_flickr}
sh ./scripts/clip_features_extraction.sh conceptual_captions {train,val}
```

To extract CLIP [CLS] embedding for CoMMuTE, our contrastive evaluation dataset, please run the following commands:
```
DATASET=CoMMuTE
SUBSET=en-{fr,de,cs}
BATCH_SIZE=100
source activate clip
python ./scripts/clip_features_extraction.py -i ./${DATASET}/images \
                                             -l ./${DATASET}/${SUBSET}/img.order \
                                             -d ./${DATASET}/${SUBSET}/features/clip_features \
                                             -b ${BATCH_SIZE}
```

#### 2.1 MDETR

Due to compatibility issues, you need to create a new conda environment to be able to use MDETR pre-trained models:
```
conda create -n mdetr python=3.8.12
conda activate mdetr
pip install -r mdetr_requirements.txt
```

You then have to run the following scripts to extract MDETR features after having inform TORCH_HUB_DIR variable in _mdetr_features_extraction.sh_:
```
source activate mdetr
sh ./scripts/mdetr_features_extraction.sh multi30k {train,val,test_2016_flickr,test_2017_flickr,test_2018_flickr}
sh ./scripts/mdetr_features_extraction.sh conceptual_captions {train,val}
```

To extract MDETR features for CoMMuTE, please run the following commands:
```
DATASET=CoMMuTE
TORCH_HUB_DIR="PATH/TO/HUB/DIR"
SUBSET=en-{fr,de,cs}
source activate mdetr
python ./scripts/mdetr_features_extraction.py -i ./${DATASET}/images \
                                              -d ./${DATASET}/${SUBSET}/features/mdetr_features \
                                              -l ./${DATASET}/${SUBSET}/img.order \
                                              --text ./${DATASET}/${SUBSET}/src.en \
                                              --threshold 0.5 \
                                              --hub_dir ${TORCH_HUB_DIR}
```

### References

- Jörg Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC'2012)


- Desmond Elliott, Stella Frank, Khalil Sima’an, and Lucia Specia. 2016. Multi30K: Multilingual EnglishGerman image descriptions. In Proceedings of the 5th Workshop on Vision and Language, pages 70– 74, Berlin, Germany. Association for Computational Linguistics.


- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, Melbourne, Australia. Association for Computational Linguistics.


- Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR.


- Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, and Nicolas Carion. 2021. MDETR - modulated detection for end-to-end multi-modal understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV),
pages 1760–1770.
