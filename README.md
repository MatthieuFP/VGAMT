# Tackling Ambiguity with Images: Improved Multimodal Machine Translation and Contrastive Evaluation (VGAMT)

<a href="https://zupimages.net/viewer.php?id=23/05/h21p.png"><img src="https://zupimages.net/up/23/05/h21p.png" alt="" width=100% /></a>

[Read the paper (arXiv)](https://arxiv.org/pdf/2212.10140.pdf)

<p align="justify"> One of the major challenges of machine translation (MT) is ambiguity, which can in some cases be resolved by accompanying context such as an image. However, recent work in multimodal MT (MMT) has shown that obtaining improvements from images is challenging, limited not only by the difficulty of building effective cross-modal representations but also by the lack of specific evaluation and training data. We present a new MMT approach based on a strong text-only MT model, which uses neural adapters and a novel guided self-attention mechanism and which is jointly trained on both visual masking and MMT. We also release <a href="https://github.com/MatthieuFP/CoMMuTE">CoMMuTE</a>, a Contrastive Multilingual Multimodal Translation Evaluation dataset, composed of ambiguous sentences and their possible translations, accompanied by disambiguating images corresponding to each translation. Our approach obtains competitive results over strong text-only models on standard English-to-French benchmarks and outperforms these baselines and state-of-the-art MMT systems with a large margin on our contrastive test set. </p>



If you use our codebase, please cite:
```
@article{vgamt,
  doi = {10.48550/ARXIV.2212.10140},
  url = {https://arxiv.org/abs/2212.10140},
  author = {Futeral, Matthieu and Schmid, Cordelia and Laptev, Ivan and Sagot, Benoît and Bawden, Rachel},
  title = {Tackling Ambiguity with Images: Improved Multimodal Machine Translation and Contrastive Evaluation},
  publisher = {arXiv},
  year = {2022}
}
```

### Clone repository with submodules

```
git clone --recurse-submodules https://github.com/MatthieuFP/VGAMT.git
```

# Data preparation

In this work, we exploit OPUS text-only, Multi30k multilingual text-image and Conceptual Caption English text-image data. To download and extract the features we use in our work, please follow the instructions [here](https://github.com/MatthieuFP/VGAMT/blob/main/data/README.md). 

# Training

Create a conda environment from the _requirements.txt_ file. This work was conducted using SLURM job scheduler. Please adapt the scripts to your local configuration.

Install adapter-transformers
```
cd adapter-transformers
pip install .
```

For all experiments, please fill in the following variables:
- CACHE_HUGGINGFACE
- DATA_PATH
- DUMP_PATH
- FEAT_PATH (if MMT experiment)
- EXP_NAME
- seed

### Text-only Machine Translation model

You need a strong text-only MT model before training VGAMT, please run the following command lines:

```
source activate vgamt

echo "NODELIST="${SLURM_NODELIST}
echo "JOB_NODELIST="${SLURM_JOB_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun ./scripts/training/train_MT_from_MBART.sh 
```

### VGAMT

To train VGAMT from a strong MT model, please inform the additional variables:

- DATA_MIX_PATH (if using VMLM objective)
- FEAT_PATH_MIX (if using VMLM objective)
- MT_MODEL_PATH

```
source activate vgamt

echo "NODELIST="${SLURM_NODELIST}
echo "JOB_NODELIST="${SLURM_JOB_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun ./scripts/training/finetune_mix_MMT_VMLM_from_MT.sh
```

# Evaluation

- **BLEU scores**:

Please, first inform MODEL_PATH variable and run _./scripts/eval/eval_mmt_bleu.sh_ to compute BLEU scores and translation generation.

- **METEOR scores**:

Inform METEOR_FILE, REFERENCE_PATH, HYPOTHESIS_PATH and TGT_LANG variables. To install meteor, please have a look [here](https://docs.meteor.com/install.html). Then, run _./scripts/eval/eval_meteor.sh_

- **COMET scores**:

Inform REFERENCE_SRC_LG, REFERENCE_TGT_LG, HYPOTHESIS_TGT_LG, PATH_TO_COMET_STORAGE. To install comet, please have a look [here](https://github.com/Unbabel/COMET). In our work, we use the _wmt20-comet-da_ model. Then, run _./scripts/eval/eval_comet.sh_

- **CoMMuTE ranking accuracy**:

To compute CoMMuTE accuracy for your model, you can run _./scripts/eval/eval_mmt_commute.sh_ after having filled in the variables described in VGAMT section.