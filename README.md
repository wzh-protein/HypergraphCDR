# HypergraphCDR

Code for Prediction of Cancer Drug Response Based on Hypergraph Convolutional Network and Contrastive Learning.

---
## 1. Clone the Repository

```bash
git clone https://github.com/wzh-protein/HypergraphCDR.git
cd HypergraphCDR

```

## 2. Environment Setup

We recommend using **conda** to create the runtime environment.

Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate hypergraphcdr
```

## 3. Dataset Preparation
```bash
#The experimental dataset can be downloaded from Zenodo:https://zenodo.org/records/18321386
```
After downloading, extract the dataset files into the `./` directory.

## 4. Preprocessing
First, create a directory to store intermediate and final results:

```bash
mkdir result
```

Then, run the preprocessing scripts in the following order:

Process IC50 drug response data

```bash
python ./process/ic50.py
```

Process drug features

```bash
python ./process/drug.py
```

Process cell features

```bash
python ./process/cell.py
```


## 5. Training

First, train the autoencoder for feature compression:
```bash
python run_AE.py
```

Then, train the main HypergraphCDR model:
```bash
python run_main.py
```
