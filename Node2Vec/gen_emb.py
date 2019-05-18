#!/usr/bin/env python
# coding=utf-8
# The example of embedding generation  and classification demostration.
# Royrong(royrong@tencent.com) 2018/10/24
# The parameters are defined in conf.py
import argparse
import logging
import os
import time

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

	output_folder = conf.output_folder + str(dataset)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	l = conf.l
	r = conf.r
	p = conf.p

	d = conf.d
	k = conf.k
	e = conf.e
	neg = conf.neg
	emb_alpha = conf.emb_alpha
	is_weighted = conf.weighted
	input_file = os.path.join(input_folder, "edgelist")
	nodelist_file = None
	res_file = "./%s.res" % (method)
	walk_file = os.path.join(output_folder, "%s.walks" % (method))
	start = time.time()
	logging.info("Start Graph Embedding")
	# Graph random sampling.
	if not os.path.exists(walk_file):
		binary = "./Node2Vec/bin/rd_w"
		logging.info("%s not found. Perform Random Walk." % (walk_file))
		# if nodelist_file is None:
		#    walk_cmd = "%s -input %s -output %s -samplelength %d -repeat %d -p %f -q 1.0 -max_node_idx 1000000000 -num_threads 38"%(binary, input_file, walk_file,l,r,p)
		# else:
		#    walk_cmd = "%s -nodelist %s -input %s -output %s -samplelength %d -repeat %d -p %f -q 1.0 -max_node_idx 1000000000 -num_threads 38"%(binary, nodelist_file, input_file, walk_file,l,r,p)
		#        walk_cmd = "%s -input:%s -output:%s -samplelength:%d -repeat:%d -p:%f -q:1.0 -weighted"%(binary, input_file, walk_file, l, r, p)
		walk_cmd = "%s -input %s -output %s -samplelength %d -repeat %d -p %f -q 1.0 %s" % (binary,
																							input_file,
																							walk_file,
																							l,
																							r,
																							p,
																							is_weighted)
		# logging.info(walk_file)
		logging.info(walk_cmd)
		os.system(walk_cmd)
	emb_file = os.path.join(output_folder, "./%s.emb" % (method))

	# Embedding generation
	if (not os.path.exists(emb_file)) and os.path.exists(walk_file):
		data = walk_file
		binary = "./bin/pn2v_opt"
		output_file = emb_file
		emb_size = d
		window = k
		negative = neg
		sample = 1e-4
		vocabmaxsize = 79600000
		ncores = 38
		niters = e
		mincount = 0
		savevocab = os.path.join(output_folder, "./%s.vocab" % (method))
		batchsize = 2 * window + 1
		alpha = emb_alpha

		# "numactl --interleave=all"
		emb_cmd = "%s -train %s -output %s -size %s -window %d -negative %d -sample %f -vocab-max-size %d -threads %d -iter %d -min-count %d -save-vocab %s -batch-size %d -debug 4 -alpha %s" % (
		binary, data, output_file, emb_size, window, negative, sample, vocabmaxsize, ncores, niters, mincount,
		savevocab, batchsize, alpha)
		logging.info(emb_cmd)
		os.system(emb_cmd)
	elif not os.path.exists(walk_file):
		logging.info("no walk file")

	end = time.time()
	logging.info("Finish Embedding. Embedding Time: %d, Embedding File Path: %s" % (end - start, emb_file))
