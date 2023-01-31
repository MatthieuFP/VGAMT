# Extract visual features

### 1. Download data

- **Multi30k**: It can be accessed [here](https://github.com/multi30k/dataset). Please do the following instructions:

```
git clone https://github.com/multi30k/dataset.git
mv dataset multi30k
cd multi30k
for fname in data/task1/raw/*; do gunzip ${fname}; done
mv data/task1/raw/* .
for fname in data/task1/image_splits/*; do new_fname="$(basename -- $fname)"; new_fname=${new_fname/txt/order}; mv ${fname} ${new_fname}; done
for name in test_2016_flickr*; do new_name=${name/test_2016_flickr/test}; cp ${name} ${new_name}; done
rm -r data scripts README.md
mkdir features images
```

You then have to request access to the images from the 2016 subset [here](https://forms.illinois.edu/sec/229675) and download the images from 2017 and 2018 subsets [here](https://drive.google.com/drive/folders/1kfgmYFL5kup51ET7WQNxYmKCvwz_Hjkt?usp=share_link).


- **Conceptual Captions**: 

### 2. Compute CLIP and MDETR features


### References


