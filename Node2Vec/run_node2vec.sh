#!/bin/bash
# 清理临时文件
#rm -r out
# 设置库环境
source ./set_dynamic_lib.sh
# 运行算法
python3 ./gen_emb.py
python3 ./demo.py

