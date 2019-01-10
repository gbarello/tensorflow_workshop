This is a small workshop on tensorflow written by Gabriel Barello for the Ahmadian/Mazzucatto group meeting in January of 2019.

If you are reading this you've already gotten ahold of the git repo. Good job!

The following is designed to get you up and running on a server that you have ssh access to.

# Building your environment and starting Jupyter Lab

We are going to create a conda environment for you to use tensorflow in. To do so, `ssh` into the server and clone this git repo. Navigate to the repo root and run:

`$./setup_conda_env.sh`

Accept the installation and let it run, you'll have to type in `y` a few times as it installs new things. This script created a barello_TF_workshop conda environment with all the appropriate packages, activate it by running

`$source activate barello_TF_workshop`

Now we can start the `jupyter lab` server. To do this run

`jupyter lab --port XXXX`

where `XXXX` is your personal choice for the port number.

Now you need to connect to the `jupyter lab` instance from your personal computer. To do so, open a new ssh terminal on your personal computer and run:

`ssh username@server.address -f -N -L XXXX:localhost:XXXX`

Finally, open your browser and naviate to `localhost:XXXX` to connect to your jupyter lab server. From here you can naviage to the git repo root and start playing with the notebooks!

When you are done, `ctrl-C` to in the server to close the jupyter lab server, and then run `$source deactivate`.

## Multi Layer Perceptron Learning

To run this notebook you need to unzip the MNIST data files. to do so run:

`unzip mnist_TRAIN.zip`
`unzip mnist_TEST.zip`

Open the `TF_MLP_notebook.ipynb` file in the jupyter, and hit 'run all cells'. It will take a minute but should run through, train the network, and visualize the learning curve.