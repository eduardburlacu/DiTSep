---
title: TSE
url: TBC
labels: [Target Speaker Extraction, Diffusion Models ] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [ Sent140, Shakespeare] # list of datasets you include in your baseline
---

# Diffusion Models for Source Separation and Extraction
Master Thesis Project at University of Cambridge. 

**Paper link:** \
**Author:** Eduard Burlacu \
**Supervisors:** Brian Sun, Phil Woodland

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
#Setup source separation env
conda env create -f environment.yaml
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# install dependencies for repo
pip install diffusers["torch"] transformers
pip install -r requirements.txt
# install PyTorch with GPU support. Please note this baseline is very lightweight so it can run fine on a CPU.
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


