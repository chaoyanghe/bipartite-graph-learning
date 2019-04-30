#!/usr/bin/env python
# coding=utf-8
'''
l=##l##
d=##d##   
r=##r##
k=##k##
e=##e##
p=##p##
neg=##neg##
emb_alpha=##emb_alpha##
'''
l = 8
d = 32
r = 10
#r = 60
k = 9
e = 2
p = 1.49007357791
neg = 11
emb_alpha = 0.040073675
it = 3000

#for the unweighted model
weighted = ""
# for the weighted model
# weighted = "-weighted"
method = "tmp"
#local version
input_folder = "./../data/Tencent-QQ"
output_folder = "./out"
#seven version
#input_folder = "/opt/ml/disk/cdg_data"
#output_folder = "/opt/ml/env/out"
