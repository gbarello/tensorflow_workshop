#!/bin/bash

conda create -n barello_TF_workshop python=3.5.4

source activate barello_TF_workshop

conda install gcc
conda install numpy
pip install tensorflow-gpu==1.7.0
conda install matplotlib
conda install -c conda-forge jupyterlab

source deactivate
echo 'Done with setup'
