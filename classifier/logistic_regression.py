# coding=utf-8
# Example script for the user classification
# Royrong (royrong@tencent.com) 2018/10/24
import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_emb_data(fname, ind=None):
	data = pd.read_csv(fname, delimiter=" ", skiprows=2, index_col=0, header=None)
	if ind is not None:
		data = data.reindex(ind, copy=False, fill_value=0)
	return data


def load_label_data(fname, ind):
	true_label = np.loadtxt(fname, dtype=int)
	print("true_label = %s" % true_label)
	y = pd.DataFrame(np.zeros(ind.shape), index=ind, dtype=int)
	print(y)
	# cp()
	for label_id in true_label:
		if label_id in y.index:
			y.loc[label_id] = 1
	return y


def load_index_data(fname):
	ind = pd.read_csv(fname, delimiter='\t', index_col=0, header=None)
	return ind.index


def load_node_attr(fname, ind=None):
	# Load the node attributes vector.
	# If there is no attribute vector, ignore it.

	data = pd.read_csv(fname, delimiter='\t', header=None, index_col=0)
	data.fillna(0, inplace=True)
	data.drop_duplicates(inplace=True)
	data.drop([2, 3], axis=1, inplace=True)
	logging.info(
		"WARN: Column 2,3 in attr_data is dropped! If you do not want to drop any column, please uncomment line 39 in logistic_regression.py.")
	if ind is not None:
		data = data.reindex(ind, copy=False, fill_value=0)
	# converters = {0: lambda s: u2i_dict[s.decode("utf-8")], 1: decode_helper, 4: decode_helper, 5: decode_helper, 6: decode_helper,
	#              7: decode_helper, 8: decode_helper, 9: decode_helper, 10: decode_helper}
	logging.info("data.shape")
	trans = MinMaxScaler(feature_range=(-1, 1)).fit(data)
	# data_values = data.values / data.values.max(axis=0)
	return pd.DataFrame(trans.transform(data), index=data.index, columns=data.columns)


def construct_data(data, y, attr_data=None):
	logging.info("Construct:")
	# sh = data.shape
	# logging.info(sh)
	if attr_data is not None:
		data = data.merge(attr_data, how="outer", right_index=True, left_index=True)
		data.fillna(0, inplace=True)
	y = y.reindex(data.index, fill_value=0)
	# logging.info(type(data))
	# logging.info(data.shape)
	# remove the embeddding to run pure LR on attributes. 32 is the embe dim.
	# data = data.iloc[:,32:]
	# logging.info(data.shape)
	return data, y


def ap(rank_list, N=None):
	if N is None:
		N = len(rank_list)
	tp = 0.0
	al = 0.0
	sump = 0.0
	cnt = 0
	t_len = int(np.sum(rank_list))
	for r in rank_list:
		cnt = cnt + 1
		if r == 1:
			tp = tp + 1
			al = al + 1
			sump = tp / al + sump
		else:
			al = al + 1
		if tp == t_len:
			break
		if cnt >= N:
			break
	if tp == 0:
		return 0
	return sump / float(tp)


def auc(y, pred):
	return metrics.roc_auc_score(y, pred)


def prec_rec(rank_list, K=None):
	N = np.sum(np.array(rank_list))
	if K is None or K > len(rank_list):
		K = N
	M = np.sum(np.array(rank_list[0:K]))
	return (M / float(K), M / float(N))


def run_exp(input_folder, emb_file, args):
	# Load Data
	## index
	if args.verbose:
		logging.info("Loading index from %s ..." % os.path.join(input_folder, args.node_file))
	node_id_file = args.node_file

	# if the parameters is node_list, choose the node list in the data directory,
	# otherwise, choose from the embedding output directory
	if args.node_file == "node_list":
		node_id_file = os.path.join(input_folder, node_id_file)

	print("node_id_file = %s" % node_id_file)
	node_ids = load_index_data(node_id_file)

	## attr
	if os.path.exists(os.path.join(input_folder, 'node_attr')):
		node_attr_file = os.path.join(input_folder, 'node_attr')
	else:
		node_attr_file = None
	if node_attr_file is not None:
		if args.verbose:
			logging.info("Loading attributes from %s ..." % os.path.join(input_folder, 'node_attr'))
		attr_data = load_node_attr(node_attr_file, node_ids)
		print("attr_data.shape = %s" % attr_data.shape[0])
	else:
		if args.verbose:
			logging.info("Cannot find %s. Skip loading attributes." % os.path.join(input_folder, 'node_attr'))
		attr_data = None

	## emb
	if args.verbose:
		logging.info("Loading emb from %s ..." % emb_file)
	data = load_emb_data(emb_file, node_ids)
	print("emb_file len = %s" % data.shape[0])

	## labels
	if args.verbose:
		logging.info("Loading labels from %s ..." % os.path.join(input_folder, 'node_true'))
	node_label_file = os.path.join(input_folder, 'node_true')
	print("node_label_file = %s" % node_label_file)
	# print("nodes_ids = %s" % node_ids)
	node_labels = load_label_data(node_label_file, node_ids)

	## Construct data
	if args.verbose:
		logging.info("Constructing data ...")
	user_x, user_y = construct_data(data, node_labels, attr_data)
	del data, node_labels, attr_data
	if args.verbose:
		logging.info("user: %d" % (user_y.shape[0]))
		logging.info(user_x.shape)

	# Self split
	test_ratio = [0.2]

	train_aps = {}
	test_aps = {}
	test_prec = {}
	test_rec = {}
	for test_size in test_ratio:
		train_aps[test_size] = []
		test_aps[test_size] = []  # aps
		test_prec[test_size] = []  # precision
		test_rec[test_size] = []  # recall
		# Resplitting
		if args.verbose:
			logging.info("Splitting data ...")
			print("Splitting data ...")
		print("len_x = %d" % len(user_x))
		print("user_y = %d" % len(user_y))

		train_x, test_x, train_y, test_y = train_test_split(user_x, user_y, test_size=test_size, random_state=42)
		del user_x, user_y
		test_y = test_y.values

		# Training
		if args.verbose:
			logging.info("Start training ...")
			print("Start training ...")
		clf = SGDClassifier(loss='log', alpha=args.alpha, max_iter=args.max_iter, shuffle=True, n_jobs=48,
							class_weight='balanced', verbose=args.verbose, tol=None)
		clf.fit(train_x, train_y)
		# Testing
		if args.verbose:
			logging.info("Start testing ...")
			print("Start testing ...")

		# f1 score
		test_predict_y = clf.predict(test_x)

		# metric
		test_precision = precision_score(test_y, test_predict_y, average="macro")
		print("test_precision = %f" % test_precision)
		logging.info("test_precision = %f" % test_precision)

		test_recall = recall_score(test_y, test_predict_y, average="macro")
		print("test_recall = %f" % test_recall)
		logging.info("test_recall = %f" % test_recall)

		test_macro_f1 = f1_score(test_y, test_predict_y, average="macro")
		print("test_macro_f1 = %f" % test_macro_f1)
		logging.info("test_macro_f1 = %f" % test_macro_f1)

		test_predict_prob = clf.predict_proba(test_x)
		print("test_predict_prob = %s" % test_predict_prob)
		# logging.info("test_predict_prob = %s" % test_predict_prob)

		# Area Under Curve of ROC (Receiver operating characteristic):
		# https://en.wikipedia.org/wiki/Receiver_operating_characteristic
		auc_s = auc(test_y, test_predict_prob)
		logging.info("auc=%.6f" % (auc_s))
		print("auc=%.6f" % (auc_s))
		fauc = open(args.res_file + "_auc", 'w')
		fauc.write(str(auc_s))
		fauc.close()

		s_idx = np.argsort(-test_predict_prob)

		rank_test = test_y[s_idx]
		judge_n = np.sum(test_predict_prob > 0.5)
		pos_n = np.sum(test_y == 1)

		N_list = [1000, 10000, pos_n, judge_n]
		if args.verbose:
			logging.info("Test Size: %f" % (test_size))
		for n in N_list:
			res_test = ap(rank_test, n)
			(prec, rec) = prec_rec(rank_test, n)
			if n == 10000:
				logging.info("object_value=%.5f" % prec)
			if n > len(rank_test):
				n = "all"
			logging.info("Test AP@%s:\t%.5f\tPrec: %.5f\tRec: %.5f" % (str(n), res_test, prec, rec))
			test_aps[test_size].append(res_test)
			test_prec[test_size].append(prec)
			test_rec[test_size].append(rec)

	# Write the evaluation result.
	fout = open(args.res_file, 'w')
	for ts in test_ratio:
		name = emb_file.split('/')[-1]
		wstr = "%s %f" % (name, ts)
		for a in test_aps[ts]:
			wstr = wstr + " " + "%.8f" % (a)
		fout.write(wstr + "\n")
	fout.close()
	fout = open(args.res_file + ".prec_rec", 'w')
	for ts in test_ratio:
		name = emb_file.split('/')[-1]
		wstr = "%s %s %f" % (name, "prec", ts)
		for a in test_prec[ts]:
			wstr = wstr + " " + "%.8f" % (a)
		fout.write(wstr + "\n")
		wstr = "%s %s %f" % (name, "rec", ts)
		for a in test_rec[ts]:
			wstr = wstr + " " + "%.8f" % (a)
		fout.write(wstr + "\n")

	fout.close()


def main():
	parser = ArgumentParser("emb_lr", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	parser.add_argument("--input_folder", required=True)
	parser.add_argument("--node_file", default="node_list")
	parser.add_argument("--emb_file", required=True)
	parser.add_argument("--res_file", required=True)
	parser.add_argument("--verbose", default=False, type=int)
	parser.add_argument("--max_iter", default=100, type=int)
	parser.add_argument("--alpha", default=0.005, type=float)
	args = parser.parse_args()

	print("input_folder = %s" % args.input_folder)
	print("emb_file = %s" % args.emb_file)
	run_exp(args.input_folder, args.emb_file, args)


if __name__ == "__main__":
	main()
