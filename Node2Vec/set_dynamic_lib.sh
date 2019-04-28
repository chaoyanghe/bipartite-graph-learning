#!/bin/bash
#获得该文件的位置
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
echo $SCRIPT_DIR
LIB_PATH="libs"
SCRIPT_DIR=${SCRIPT_DIR}/${LIB_PATH}
echo $SCRIPT_DIR
export LD_LIBRARY_PATH=${SCRIPT_DIR}:${LD_LIBRARY_PATH}
echo ${LD_LIBRARY_PATH}
