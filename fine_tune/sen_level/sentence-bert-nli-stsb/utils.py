'''utils for saving and loading.'''
import glob
import pickle
import os
import re
import sys
import shutil
import logging

from datetime import datetime

def init_logger(name):
	'''set up training logger.'''
	logger = logging.getLogger(name)
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	h = logging.StreamHandler(sys.stdout)
	h.flush = sys.stdout.flush
	logger.addHandler(h)
	return logger

# setup logger
logger = init_logger(__name__)

def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):

	# Check if we should delete older checkpoint(s)
	checkpoints_sorted = sorted_checkpoints(args, checkpoint_prefix, use_mtime)
	if len(checkpoints_sorted) <= args.save_total_limit:
		return

	number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
	checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
	for checkpoint in checkpoints_to_be_deleted:
		logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
		shutil.rmtree(checkpoint)

'''saving and loading modules from Hugging face.'''
def sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
	ordering_and_checkpoint_path = []

	glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

	for path in glob_checkpoints:
		if use_mtime:
			ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
		else:
			regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
			if regex_match and regex_match.groups():
				ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

	checkpoints_sorted = sorted(ordering_and_checkpoint_path)
	checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
	return checkpoints_sorted

'''avoid duplicate results in the result.pkl'''
def avoid_duplicate(args, filename="result_sts.pkl"):
	if os.path.exists(filename):
		result = pickle.load(open(filename, 'rb'))
	else:
		return False
	all_log_dir = []
	for i in result:
		all_log_dir.append(i['save_dir'])

	if args.output_dir in all_log_dir:
		return True
	else:
		return False

'''write results into result.pkl'''
def write_result(args, scores, filename="result_sts.pkl"):
	if os.path.exists(filename):
		result = pickle.load(open(filename, 'rb'))
	else:
		result = list()
	result.append({
		'score_time': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
		'save_dir': args.output_dir,
		'tb_dir': args.desc_string,
		'num_epochs': args.num_epochs,
		'batch_size': args.batch_size,
		'learning_rate': args.learning_rate,
		'eval_score': scores
	})
	with open(filename, 'wb') as f:
		pickle.dump(result, f)
		