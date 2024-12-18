#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="DiTSep",
    version="0.0.1",
    description="Transformer-based diffusion for Source Separation and Target Speech Extraction",
    author="Eduard Burlacu",
    author_email="efb48@cam.ac.uk",
    url="https://github.com/eduardburlacu/DiTSep",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
