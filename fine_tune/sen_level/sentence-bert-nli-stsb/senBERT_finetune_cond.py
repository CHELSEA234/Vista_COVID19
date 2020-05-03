import math
import argparse

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
from datetime import datetime
from SentenceTransformer_custom import SentenceTransformer_tb
from datetime import datetime
from utils import *

# setup logger
logger = init_logger(__name__)

def train_config(sts_reader, model, batch_size):
	'''train dataloader and model.'''
	logger.info(f"Read STSbenchmark train dataset")
	train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
	train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
	train_loss = losses.CosineSimilarityLoss(model=model)
	train_evaluator = EmbeddingSimilarityEvaluator(train_dataloader)
	return train_data, train_loss, train_dataloader, train_evaluator

def dev_config(sts_reader, model, batch_size):
	'''dev dataloader and model'''	
	logger.info(f"Read STSbenchmark dev dataset")
	dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
	dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
	dev_loss = losses.CosineSimilarityLoss(model=model)
	dev_evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
	return dev_loss, dev_dataloader, dev_evaluator

def sentence_bert_initialization(covid_scibert_model, tokenizer, args):
	'''Use BERT for mapping tokens to embeddings'''
	if not args.should_continue:
		word_embedding_model = SciBERT(covid_scibert_model, tokenizer)
		pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
									   pooling_mode_mean_tokens=True,
									   pooling_mode_cls_token=False,
									   pooling_mode_max_tokens=False)
		model = SentenceTransformer_tb(modules=[word_embedding_model, pooling_model])
		logger.info("Initialize Sentence BERT.")
	else:
		model = SentenceTransformer_tb(saved_scibert_model_path=args.sentence_model_path)
		logger.info(f"Loading Sentence BERT from {args.sentence_model_path}.")
	return model

def main(args):
	# Read the dataset
	args.output_dir = os.path.join(args.output_dir, 
									f'training_nli_sts-sci_bert-num_epochs_{args.num_epochs}-bs_{args.batch_size}-lr_{args.learning_rate}')
	logger.info(f"the output directory is {args.output_dir}.")

	if avoid_duplicate(args):
		logger.info(f"the experiment is previously done.")
		sys.exit(0)

	if args.should_continue:
		sorted_checkpoints_list = sorted_checkpoints(args)
		if len(sorted_checkpoints_list) == 0:
			raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
		elif len(sorted_checkpoints_list) == 1:
			raise ValueError("In theory, the second last checkpoint should be read when using ephemeral.")
		else:
			args.sentence_model_path = sorted_checkpoints_list[-2]
		model = SentenceTransformer_tb(saved_scibert_model_path=args.sentence_model_path)
		# Load a pre-trained sentence transformer model
		# args.sentence_model_path has optimizer and scheduler
	else:
		model = SentenceTransformer_tb(saved_scibert_model_path=args.finetuned_model_path)

	sts_reader = STSBenchmarkDataReader(args.data_directory, normalize_scores=True)

	# Convert the dataset to a DataLoader ready for training and dev.
	train_data, train_loss, train_dataloader, train_evaluator = train_config(sts_reader, model, args.batch_size)
	dev_loss, dev_dataloader, dev_evaluator = dev_config(sts_reader, model, args.batch_size)

	# Configure the training. We skip evaluation in this example
	warmup_steps = math.ceil(len(train_data)*args.num_epochs/args.batch_size*0.1) #10% of train data for warm-up
	optimizer_params = {'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False}
	logger.info(f"Warmup-steps: {warmup_steps}")

	# Train the model
	model.fit(args=args,
			train_objectives=[(train_dataloader, train_loss)],
			eval_objectives=[(dev_dataloader, dev_loss)],
			train_evaluator=train_evaluator,
			evaluator=dev_evaluator,
			train_phase='STS',
			epochs=args.num_epochs,
			evaluation_steps=50 if not args.debug_mode else 2,
			optimizer_params=optimizer_params,
			warmup_steps=warmup_steps)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Fine tuning with STS.')
	parser.add_argument('--finetuned_model_path', default='./saved_model', type=str, help="finetuned word-level embedding.")
	parser.add_argument('--debug_mode', action='store_true', help="debug mode or not.")
	# parser.set_defaults(debug_mode=True)
	parser.add_argument('--batch_size', default=16, type=int, help="batch_size.")
	parser.add_argument('--data_directory', default='./datasets/stsbenchmark', 
						type=str, help="the original dataset.")
	parser.add_argument('--num_epochs', default=1, type=int, help='training epoch.')
	parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning rate.')
	parser.add_argument('--should_continue', action='store_true', help="continue to train.")
	parser.add_argument('--output_dir', default='output', type=str, help="model output directory.")
	parser.add_argument('--save_total_limit', default=5, type=int, 
						help="only save the last several checkpoints.")
	args = parser.parse_args()
	main(args)
