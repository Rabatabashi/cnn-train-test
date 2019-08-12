#!/bin/bash
# This bash script create a virtual environment for the cnn-train-test project.
# It is installing all necessary modules for project.
# Correct usage of this script is: /PATH-TO-THE-DIRECTORY-OF-THIS-SCRIPT/create-python-virtual-environment-to-this-project.sh <PATH-TO-VIRTUAL-ENV-DIR>


#The first command line arguments after the name of script is the path where we want to create the virtual environment.
VIRTUAL_ENV_DIR=$1

#Create a python3 virtual environment
python3 -m venv $VIRTUAL_ENV_DIR
source $VIRTUAL_ENV_DIR/bin/activate

#necessary modules
pip install numpy
pip install python-mnist
pip install argparse

# After this program finished you can run "source VIRTUAL_ENV_DIR/bin/activate" command and the terminal will be entering to the virtual environment which created by this script.
echo "You can enter to the cnn-train-test virtual-environment, with the following(next line) command:"
echo "source $VIRTUAL_ENV_DIR/bin/activate"