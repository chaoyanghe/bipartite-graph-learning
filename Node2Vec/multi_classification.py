#!/usr/bin/env python
# coding=utf-8
# The example of embedding generation  and classification demostration.
# Royrong(royrong@tencent.com) 2018/10/24
# The parameters are defined in conf.py
import argparse
import logging
import os

import conf


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cora', required=True)

	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	dataset = args.dataset

	method = conf.method
	input_folder = conf.input_folder + str(dataset)

	output_folder = conf.output_folder + "/" + str(dataset)

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	l = conf.l
	r = conf.r
	p = conf.p

	d = conf.d
	k = conf.k
	e = conf.e
	neg = conf.neg
	it = conf.it
	emb_alpha = conf.emb_alpha

	res_file = os.path.join(output_folder, "./%s.res" % (method))
	emb_file = os.path.join(output_folder, "./%s.emb" % (method))
	logging.info("This is the demo for logistic regression using the embedding vectors")

	hgcn_node_file = "hgcn_node_list"
	print(hgcn_node_file)

	# Performing example logistic regression
	if os.path.exists(emb_file):
		max_iter = 300
		lr_cmd = "python ./classifier/multiclass_lr.py --verbose 0 --input_folder %s --emb_file %s --node_file %s --res_file %s --max_iter %d" % (
			input_folder, emb_file, hgcn_node_file, res_file, it)

		os.system(lr_cmd)
	else:
		logging.info("no emb file")
		exit(1)
	res_file = res_file + ".prec_rec"

	if os.path.exists(res_file):
		fin = open(res_file, 'r')
		ap = 0
		for l in fin:
			ap = ap + float(l.strip().split(' ')[-2])
		fout = open(os.path.join(output_folder, "result_file"), 'w')
		fout.write("object_value=%f" % (ap))
		fout.close()
	else:
		logging.info("No res file.")
		exit(1)
