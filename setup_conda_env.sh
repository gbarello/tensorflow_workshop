#!/bin/bash

conda create -n barello_TF_workshop python=3.5.4

source activate barello_TF_workshop

conda install numpy
pip install tensorflow-gpu==1.7.0
conda install -c conda-forge jupyterlab
conda install matplotlib

source deactivate
echo 'Done with setup'
