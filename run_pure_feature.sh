#!/usr/bin/env bash
python3 ./classifier/logistic_regression.py --verbose 0 \
--input_folder "./data/tencent/" \
--node_file "node_list" \
--res_file "./out/pure" \
--max_iter 3000  \
--pure True


#python3 ./classifier/multiclass_lr.py --verbose 0 \
#--input_folder "./data/citeseer/" \
#--node_file "node_list" \
#--res_file "./out/pure" \
#--max_iter 3000  \
#--pure True

