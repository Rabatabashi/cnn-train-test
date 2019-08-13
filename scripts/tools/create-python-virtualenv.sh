#!/bin/bash

# This bash script create a virtual environment for the cnn-train-test project.
# It is installing all necessary modules for project.
#
# @author Kisházi "janohhank" János
# @author Nagy "rabatabashi" Márton

declare -r SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function usage(){
	echo "$0 VIRTUALENV-DIR TF-TYPE"
	echo -e "\t VIRTUALENV-DIR is the target directory path where will the virtualenv created."
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

declare -r VIRTUALENV_DIR="$1"

if [ ! -d "$VIRTUALENV_DIR" ]; then
	echo "[$0][ERROR] The VIRTUALENV_DIR does not denote a directory: $VIRTUALENV_DIR!"
	exit 1
fi

if [ ! -x "$(command -v virtualenv)" ]; then
	echo "[$0][ERROR] virtualenv missing!"
	exit 1
fi

declare -r VIRTUALENV_PATH="$VIRTUALENV_DIR"/virtualenv-cnn-train-test
echo "[$0][INFO] Setting up virtualenv into $VIRTUALENV_PATH."

virtualenv -p python3 "$VIRTUALENV_PATH"
source "$VIRTUALENV_PATH"/bin/activate
cd "$VIRTUALENV_PATH"/bin
python pip install --upgrade numpy argparse python-mnist
cd "$SELFDIR"

echo "[$0][INFO] Virtualenv setup is done."