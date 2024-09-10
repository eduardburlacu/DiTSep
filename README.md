---
title: 
url: TBC
labels: [Target Speaker Extraction, Diffusion Models, , ] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [MNIST, CIFAR10, Sent140, Shakespeare] # list of datasets you include in your baseline
---

# Diffusion Models for Source Separation and Extraction
Master Thesis Project at University of Cambridge.
**Paper link:** 

**Authors:** Eduard Burlacu, Stylianos Venieris, Aaron Zhao, Robert Mullins

**Abstract:** 


> Note: If you use this baseline in your work, please remember to cite the original authors of the paper.

## About this baseline
**What's implemented:** Source code used for producing the results in _____ paper.

**Datasets:**
* 
 
**Hardware Setup:** These experiments were run on ___

**Contributor:** Eduard Burlacu

## Environment Setup

To construct the Python environment follow these steps:
```bash
# install the base Poetry environment
poetry install

# activate the environment
poetry shell

# install PyTorch with GPU support. Please note this baseline is very lightweight so it can run fine on a CPU.
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Experiments
### Experiment 1:
**Motivation:** 

**Tasks:** 
* 

**Models:** This directory implements the following models:
- 

**Datasets:**  The settings are as follows:

| Dataset     | #speakers | target method | SI-SDR |
|:------------|:---------:|:-------------:|:------:|



**Training Hyperparameters:**
The following table shows the main hyperparameters for this baseline with their default value

## Running the Experiments
### Example


### Expected results


