'''
scripts to load SciBERT finetued on COVID_19
'''

import torch
import argparse

from transformers import *

def main(model_name_or_path):

	# model-agnostic loading: https://github.com/huggingface/transformers#quick-tour-of-model-sharing
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
	model = AutoModelWithLMHead.from_pretrained(model_name_or_path, config=config)	## fine-tuned model.

	sample_str = "Here is some text to encode"
	input_ids = torch.tensor([tokenizer.encode(sample_str, add_special_tokens=True)])
	with torch.no_grad():
		preds, hidden_states = model(input_ids)  # Models outputs are now tuples
		assert len(hidden_states), print(f"The model should be BERT-base.")
		# the first index means the input embedding.
		# https://stackoverflow.com/questions/60120849/outputting-attention-for-bert-base-uncased-with-huggingface-transformers-torch
		
		extracted_feature = hidden_states[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='load the fine-tune model.')
    parser.add_argument('--model_name_or_path', 
    					default="./model-SciBERT_lr-1e-05_maxepoch-3_bs-8/checkpoint-best")
    args = parser.parse_args()
    main(args.model_name_or_path)
