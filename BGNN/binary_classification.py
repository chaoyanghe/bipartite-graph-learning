import argparse
import logging
import os

import conf
# import wandb

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cora', required=True)
	parser.add_argument('--model', type=str, default='adv', required=True)
	parser.add_argument('--rank', type=int, default=-1,
						help='process ID for MPI Simple AutoML')

	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	dataset = args.dataset
	model = args.model
	rank = args.rank

	method = conf.method
	input_folder = conf.input_folder + str(dataset)
	output_folder = None
	if model == "adv":
		output_folder = conf.output_folder_bgnn_adv + "/" + str(dataset)
	elif model == "mlp":
		output_folder = conf.output_folder_bgnn_mlp + "/" + str(dataset)

	print("output_folder = %s" % output_folder)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	it = conf.it

	emb_file = str(output_folder) + "/bgnn.emb"
	print(emb_file)

	# the BGNN node list is smaller than the raw node list because some illegal nodes are filtered
	bgnn_node_file = str(output_folder) + "/node_list"

	res_file = str(output_folder) + "/bgnn.res"
	f = open(res_file, "+w")
	f.write("")
	f.close()
	print(res_file)

	# Performing example logistic regression
	if os.path.exists(emb_file):
		max_iter = 300
		lr_cmd = "python3 ./classifier/logistic_regression.py --verbose 0 --input_folder %s --emb_file %s --node_file %s --res_file %s --max_iter %d" % (
			input_folder,
			emb_file,
			bgnn_node_file,
			res_file,
			it)

		os.system(lr_cmd)
	else:
		print("no emb file")
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

		# wandb.run.summary["output_model_accuracy"] = str(ap)
	else:
		logging.info("No res file.")
		exit(1)
