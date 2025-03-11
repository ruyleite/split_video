#!/usr/bin/env bash
python3 -m venv ${HOME}/.env/default

source ${HOME}/.env/default/bin/activate
export DIR_PATH=$(dirname "$0")

cd $DIR_PATH

pip3 install -r requirements.txt


