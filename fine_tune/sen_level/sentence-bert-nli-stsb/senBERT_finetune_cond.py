import logging
import math
import argparse

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
from datetime import datetime
from SentenceTransformer_custom import SentenceTransformer_tb

from utils import *

# setup logger
logger = init_logger(__name__)

def logging_starter():
    '''
    the logging setting.
    '''
    #### Just some code to print debug information to stdout
    from sentence_transformers import LoggingHandler
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

def train_config(sts_reader, model, batch_size):
	'''train dataloader and model.'''
	logging.info(f"Read STSbenchmark train dataset")
	train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
	train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
	train_loss = losses.CosineSimilarityLoss(model=model)
	train_evaluator = EmbeddingSimilarityEvaluator(train_dataloader)
	return train_data, train_loss, train_dataloader, train_evaluator

def dev_config(sts_reader, model, batch_size):
	'''dev dataloader and model'''	
	logging.info(f"Read STSbenchmark dev dataset")
	dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
	dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
	dev_loss = losses.CosineSimilarityLoss(model=model)
	dev_evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
	return dev_loss, dev_dataloader, dev_evaluator

def main(finetuned_model_path, debug_mode, batch_size, data_directory, num_epochs, learning_rate,
		logging_option):

	if logging_option:
		logging_starter()

	# Read the dataset
	model_save_path = 'output/training_nli_sts_continue_training-sci_bert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	sts_reader = STSBenchmarkDataReader(data_directory, normalize_scores=True)

	# Load a pre-trained sentence transformer model
	model = SentenceTransformer_tb(saved_scibert_model_path=finetuned_model_path)

	# Convert the dataset to a DataLoader ready for training and dev.
	train_data, train_loss, train_dataloader, train_evaluator = train_config(sts_reader, model, batch_size)
	dev_loss, dev_dataloader, dev_evaluator = dev_config(sts_reader, model, batch_size)

	# Configure the training. We skip evaluation in this example
	warmup_steps = math.ceil(len(train_data)*num_epochs/batch_size*0.1) #10% of train data for warm-up
	optimizer_params = {'lr': learning_rate, 'eps': 1e-6, 'correct_bias': False}
	logging.info(f"Warmup-steps: {warmup_steps}")

	# Train the model
	model.fit(train_objectives=[(train_dataloader, train_loss)],
			eval_objectives=[(dev_dataloader, dev_loss)],
			train_evaluator=train_evaluator,
			evaluator=dev_evaluator,
			train_phase='STS',
			epochs=num_epochs,
			evaluation_steps=10,
			optimizer_params=optimizer_params,
			warmup_steps=warmup_steps,
			output_path=model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine tuning with STS.')
    parser.add_argument('--finetuned_model_path', default='./saved_model', type=str, help="finetuned word-level embedding.")
    parser.add_argument('--debug_mode', action='store_true', help="debug mode or not.")
    # parser.set_defaults(debug_mode=True)
    parser.add_argument('--batch_size', default=16, type=int, help="batch_size.")
    parser.add_argument('--data_directory', default='./datasets/stsbenchmark', 
                        type=str, help="the original dataset.")
    parser.add_argument('--num_epochs', default=1, type=int, help='training epoch.')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate.')
    parser.add_argument('--logging_option', action='store_true', help="open the custom logging handler.")
    parser.set_defaults(logging_option=True)
    args = parser.parse_args()
    main(finetuned_model_path=args.finetuned_model_path, debug_mode=args.debug_mode, batch_size=args.batch_size,
        data_directory=args.data_directory, num_epochs=args.num_epochs, learning_rate=args.lr, 
        logging_option=args.logging_option)
