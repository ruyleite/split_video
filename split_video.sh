#!/usr/bin/env bash
source ${HOME}/.env/default/bin/activate
export DIR_PATH=$(dirname "$0")

cd ${DIR_PATH}

${DIR_PATH}/split_video.py -f $1