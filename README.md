---
title: DiTSep
labels: [Source Separation, Diffusion Models ] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [WSJ0-2mix, WSJ0-3mix, Libri2mix, Libri3mix] # list of datasets you include in your baseline
---

# Latent Diffusion Model for Source Separation
Master Thesis Project at University of Cambridge.

<!--**Site link:** \-->
<!--**Paper link:** \-->
**Author:** Eduard Burlacu \
**Supervisors:** Brian Sun, Phil Woodland

**Abstract:** 

## About this project
**What's implemented:** Source code used for producing the results in _____ paper.

**Datasets: Libri2Mix, WSJ0-2mix**
* 
 
**Hardware Setup:** These experiments were run on ___

**Contributor:** Eduard Burlacu

## Environment Setup

To construct the Python environment follow these steps:
```bash
#Setup source separation env
conda env create -f env/environment.yaml
```

## Experiments
### Training the OobleckVAE:

We use the StabilityAI's [ `stable-audio-tools` ](https://github.com/Stability-AI/stable-audio-tools) to train an [ `OobleckVAE` ](https://huggingface.co/docs/diffusers/v0.30.3/en/api/models/autoencoder_oobleck) specially-designed for source separation, being able to encode and reconstruct multi-speaker audio samples.


**Useful for these Tasks:** 
Blind Source Separation, Speech Enhancement, Target Speaker Extraction

**Models:** 
- 

**Datasets:**  The settings are as follows:

| Dataset     | #speakers | target method | SI-SDR |
|:------------|:---------:|:-------------:|:------:|

