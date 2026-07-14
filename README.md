<h1 align="center">
Morphoscreen

</h1>

## Deep learning extracts MoA-specific signatures from high-throughput images of chemically and genetically perturbed Corynebacteria

<p align="center">
    <a href="https://www.biorxiv.org/content/10.64898/2026.02.23.707449v1"><img alt="Paper" src="https://img.shields.io/badge/paper-bioRxiv-%23b62b39"></a>
    <a href="https://github.com/krentzd/morphoscreen"><img alt="github" src="https://img.shields.io/github/stars/krentzd/morphoscreen?style=social"></a>
    <a href="https://github.com/krentzd/morphoscreen"><img alt="github" src="https://img.shields.io/github/forks/krentzd/morphoscreen?style=social"></a>
</p>
</p>

## Overview 
This repository contains the source code to reproduce the analysis from "Deep learning extracts MoA-specific signatures from high-throughput images of chemically and genetically perturbed Corynebacteria".
![width=10](docs%2Fimages%2FOverview_figure.png)

## Installation

### Install dependencies within conda environment
First, create and activate a [conda](https://docs.conda.io/en/latest/) environment:

```bash
conda create -n ai4ab_env python=3.9
conda activate ai4ab_env
```

Then, clone this repository and install the required dependencies:
```bash
git clone https://github.com/krentzd/ai4ab.git
cd ai4ab
pip install torch==1.12.1 torchvision==0.13.1  # Install PyTorch
pip install -r requirements.txt
```

### Singularity image 
Alternatively, you can build a [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) image using the provided recipe as follows:

```bash
git clone https://github.com/krentzd/ai4ab.git
cd ai4ab
singularity build ai4ab.sif ai4ab.def
```

After building the singularity image, you can directly run training and testing scripts from the terminal:

```bash
singularity exec --bind PATH_TO_AI4AB:PATH_TO_AI4AB --nv ai4ab.sif python model/fine_tune_train.py --save_dir SAVE_DIR
```

## Usage 

### Datset preparation

Your dataset must obey the following folder structure: 

```
DATA_DIR
└── Plate_1
│   └── Compound_1_Concentration_1
│   │   ├── img_1.tiff
│   │   ├── img_2.tiff
│   │   └── ...
│   ├── ...
│   └── Compound_N_Concentration_M
├── ...
└── Plate_K
```


### Model pre-training

To pre-train a model from scratch, run the following command within your conda environment: 

```bash
python model/pre_train.py \
    --data_dir DATA_DIR \
    --save_dir SAVE_DIR \ 
    --train_dir Plate_1 Plate_2 \
    --test_dir Plate_N \
```

### Model fine-tuning

To fine-tune a pre-trained model on a single concentration (e.g. 10xMIC), run the followong command within your conda environment:

```bash
python model/fine_tune_train.py \
    --data_dir DATA_DIR \
    --save_dir SAVE_DIR \ 
    --train_dir Plate_1 Plate_2 \
    --test_dir Plate_N \
    --ckpt_path PATH_TO_PRE_TRAINED_CKPT \
    --dose 10xMIC
```

### Model testing
To test the model on the Plate defined in `test_dir`, run the following command within your conda environment: 

```bash
python model/fine_tune_test.py \
    --save_dir SAVE_DIR \
    --ckpt -1 \                       # -1 selects the checkpoint with the lowest validation loss
```

## Reproduce figures from manuscript
Run analysis notebooks in the [`analysis` folder](analysis)

## How to cite
@article{krentzel2026deep,
  title={Deep learning extracts MoA-specific signatures from high-throughput images of chemically and genetically perturbed Corynebacteria},
  author={Krentzel, Daniel and Petit, Julienne and Boudehen, Yves-Marie and Mahtal, Nassim and Sadowski, Elodie and Zettor, Agn{\`e}s and Aubry, Alexandra and Chiaravalli, Jeanne and Aulner, Nathalie and Petrella, St{\'e}phanie and others},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
