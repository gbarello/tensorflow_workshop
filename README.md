# A Short Tensorflow Workshop

This is a small workshop on tensorflow written by Gabriel Barello for the Ahmadian/Mazzucatto group meeting in January of 2019.

If you are reading this you've already gotten ahold of the git repo. Good job!

The following is designed to get you up and running on a server that you have ssh access to. It assumes that the server has an up-to-date version of conda installed. This entire tutorial was tested with conda version:

```
$ conda --version
conda 4.3.21
```

If you want to run it on your own personal machine (without connecting to a server) you do everything the same except you don't need to run the second `ssh` command below (the one I ask you to run from your personal machine). Instead, just run the `jupyter lab` command and then connect directly to `localhost:XXXX` in your browser.

## Building your environment and starting Jupyter Lab

We are going to create a conda environment for you to use tensorflow in. To do so, `ssh` into the server and clone this git repo. Navigate to the repo root and run:

`$./setup_conda_env.sh`

Accept the installation and let it run, you'll have to type in `y` a few times as it installs new things. This script created a barello_TF_workshop conda environment with all the appropriate packages, activate it by running

`$source activate barello_TF_workshop`

Now we can start the `jupyter lab` server. To do this run

`jupyter lab --port XXXX`

where `XXXX` is your personal choice for the port number.

Now you need to connect to the `jupyter lab` instance from your personal computer. To do so, open a new ssh terminal on your personal computer and run:

`ssh username@server.address -f -N -L XXXX:localhost:XXXX`

If you are running on your own personal machine (without connecting to a server) you should skip this step. 

Finally, open your browser and naviate to `localhost:XXXX` to connect to your jupyter lab server. From here you can naviage to the git repo root and start playing with the notebooks!

When you are done, `ctrl-C` to in the server to close the jupyter lab server, and then run `$source deactivate`.

## Introduction - (Introduction.ipynb)

Open the `Introduction.ipynb` file in the jupyter. I recommend reading through and running each cell one at a time.

## Multi Layer Perceptron (MLP) Learning - (TF_MLP_notebook.ipynb)

To run this notebook you need to unzip the MNIST data files. to do so run:

`unzip mnist_TRAIN.zip`
`unzip mnist_TEST.zip`

Open the `TF_MLP_notebook.ipynb` file in the jupyter, and hit 'run all cells'. It will take a minute but should run through, train the network, and visualize the learning curve.

## Recurrent Neural Network (RNN) Learning - (TF_RNN_notebook.ipynb)

To run this notebook you need to unzip the MNIST data files. to do so run:

`unzip mnist_TRAIN.zip`
`unzip mnist_TEST.zip`

Open the `TF_RNN_notebook.ipynb` file in the jupyter, and hit 'run all cells'. It will take a minute but should run through, train the network, and visualize the network output.

## Stabilized Supralinear Network (SSN) Simulation - (Building_SNN.ipynb)

Open the `Building_SSN.ipynb` file in the jupyter, and hit 'run all cells'. It will take a minute but should run through and simulate the netowrk on the GPU and on the CPU and print the results, as well as the computation time for each.
