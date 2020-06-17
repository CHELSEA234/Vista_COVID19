'''
converting data into the tokenizer.
'''
import subprocess
import os
import time
import torch
import pickle
import h5py
import argparse
import shutil

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
	MODEL_WITH_LM_HEAD_MAPPING,
	WEIGHTS_NAME,
	AdamW,
	AutoConfig,
	AutoModelWithLMHead,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)

class TextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, original_data_path: str, block_size=512, line_break_num=100000):
		assert os.path.isfile(file_path)

		block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

		_, filename = os.path.split(file_path)
		cached_features_file = os.path.join(
			original_data_path, "bert_cached_lm_" + str(block_size) + "_" + filename + '.hdf5'
		)

		self.cached_features_file = cached_features_file
		# self.examples = []
		self.text_interval = []
		text_str = ''
		line_num = 0
		last_flag = False
		with open(file_path, encoding="utf-8") as f:
			while True:
				textline = f.readline()
				line_num += 1
				if not textline:
					last_flag = True
				else:
					text_str = text_str + textline

				if last_flag or (line_num % line_break_num == 0):
					print(f"preprocessing text at the {line_num} line.", flush=True)
					tokenized_text = tokenizer.tokenize(text_str)
					tokenized_text = tokenizer.convert_tokens_to_ids(tokenized_text)

					for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):  # Truncate in block of block_size
						text_interval = tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
						self.text_interval.append(text_interval)

					slice_num = int(line_num / line_break_num) if not last_flag else int(line_num / line_break_num) + 1
					dataset_name = "preprocessed_data_" + str(slice_num) 
					if line_num / line_break_num == 1:
						with h5py.File(cached_features_file, "w") as h5_handler:
							dset = h5_handler.create_dataset(dataset_name, data=self.text_interval)
					else:
						with h5py.File(cached_features_file, "a") as h5_handler:
							dset = h5_handler.create_dataset(dataset_name, data=self.text_interval)

						with h5py.File(cached_features_file, 'r') as h5_handler:
							print(f"current result slices number is {len(h5_handler.keys())}.", flush=True)

					self.text_interval = []	# reset
					text_str = ''			# reset
					if last_flag:
						break

def main(args):
	'''
	tokenize data into the format that can be used.
	'''
	raw_file_path = os.path.join(args.txt_dir, args.file_name)
	print(f"the raw data path is: {raw_file_path}.")

	full_str_copy_data = f'for data in `ls {raw_file_path}`; do cp -v $data $TMPDIR & done; wait; '
	start = time.time()
	status = subprocess.call(full_str_copy_data, shell=True)
	end = time.time()
	print('Data copied in %2.2f minutes' % (float(end-start)/60.))

	tmp_data_path = os.environ['TMPDIR']
	file_path = os.path.join(tmp_data_path, args.file_name)
	# tmp_data_path

	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=None)
	text_dataset = TextDataset(tokenizer, file_path=file_path, original_data_path=tmp_data_path, line_break_num=args.line_break_num)
	print(f"tokenization finished, starting copying back to raw path: {args.txt_dir}.")

	print(text_dataset.cached_features_file)

	start = time.time()
	shutil.move(text_dataset.cached_features_file, args.txt_dir)
	end = time.time()
	print('Data moved in %2.2f minutes' % (float(end-start)/60.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-de', '--debug_mode', action='store_true', help="debug mode.")
    parser.add_argument('-file', '--file_name', type=str, default='train.txt')
    parser.add_argument('-model', '--model_name_or_path', type=str, 
    					default='../scibert_pertrained/scibert_scivocab_uncased_pytorch')
    parser.add_argument('-line', '--line_break_num', type=int, default=100000)
    parser.add_argument('-t', '--txt_dir', default='../bio_medical_data_full')
    args = parser.parse_args()
    main(args)
