'''
generate vocab.json and merges.txt based on the given txt data.
define config.json, other files related to tokenizer.
'''
import os
import torch
import transformers

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

def main(output_dir, input_dir):

	paths = [str(x) for x in Path(input_dir).glob("**/*.txt")]
	tokenizer = ByteLevelBPETokenizer()

	# Customize training
	tokenizer.train(files=paths, vocab_size=32_000, min_frequency=2, special_tokens=[
	    "<s>",
	    "<pad>",
	    "</s>",
	    "<unk>",
	    "<mask>",
	])

	# save to target directory.
	os.makedirs(output_dir, exist_ok=True)
	tokenizer.save(output_dir)

	## TODO: config files.

	# save to special_tokens_map.json.
	## TODO: fit into tokenizer.
	special_config = {
		"unk_token": "[UNK]",
		"sep_token": "[SEP]",
		"pad_token": "[PAD]",
		"cls_token": "[CLS]",
		"mask_token": "[MASK]"
	}
	with open(os.join.path(output_dir, "special_tokens_map.json"), 'w') as fp:
		json.dump(special_config, fp)

	# save to tokenzier_config.json.
	tokenizer_config = {
		"max_len": 512,
		"do_lower_case": false
	}

	with open(os.join.path(output_dir, "tokenizer_config.json"), 'w') as fp:
	    json.dump(tokenizer_config, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fine-tune scibert on COVID_19.')
    parser.add_argument('--output_dir', default="../scibert_pertrained/scibert_scivocab_uncased_pytorch", 
                        help="training data precentage.")
    parser.add_argument('--input_dir', default="../COVID_19_data/", help="the text files.")
    parser.add_argument('--vocab_size', default=52_000, help="vocab size.")
    args = parser.parse_args()
    main(args.output_dir, args.input_dir)
