import torch
import os
import transformers
import math
import argparse
import random
random.seed(0)
import numpy as np

from torch.utils.data import DataLoader
from transformers import *
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *
from SentenceTransformer_custom import SentenceTransformer_tb
from datetime import datetime
from utils import *

# setup logger
logger = init_logger(__name__)

class SciBERT(models.BERT):
    """SciBERT model to generate token embeddings.

    Each token is mapped to an output vector from SciBERT.
    """
    def __init__(self, SciBERT_model, SciBERT_tokenizer, max_seq_length: int = 128, do_lower_case: bool = True):
        super(models.BERT, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logger.warning("BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length

        self.bert = SciBERT_model
        self.tokenizer = SciBERT_tokenizer
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

def loading_pretrained_model(args):
    '''loading pretrained SciBERT.'''
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path)
    config = AutoConfig.from_pretrained(args.finetuned_model_path, output_hidden_states=True)
    model = AutoModelWithLMHead.from_pretrained(args.finetuned_model_path, config=config)
    covid_scibert_model = model.bert
    return tokenizer, covid_scibert_model

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

    args.output_dir = os.path.join(args.output_dir, 
                                    f'training_nli_sci_bert-num_epochs_{args.num_epochs}-bs_{args.batch_size}-lr_{args.learning_rate}')
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

    tokenizer, covid_scibert_model = loading_pretrained_model(args)
    model = sentence_bert_initialization(covid_scibert_model, tokenizer, args)

    # Read the dataset
    nli_reader = NLIDataReader(os.path.join(args.data_directory, 'AllNLI'))
    sts_reader = STSBenchmarkDataReader(os.path.join(args.data_directory, 'stsbenchmark'))
    train_num_labels = nli_reader.get_num_labels()

    # Convert the dataset to a DataLoader ready for training
    # 942,069 for training, around 1000 for monitor, 1500 for evaluation 
    logger.info("Read AllNLI train dataset")
    nli_examples_ = nli_reader.get_examples('train.gz')[:1000] if args.debug_mode else nli_reader.get_examples('train.gz')
    train_data = SentencesDataset(examples=nli_examples_, model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.SoftmaxLoss(model=model, 
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
                                    num_labels=train_num_labels)

    ## to measure the accuracy over a small portion of training data.
    example_num = len(nli_examples_)
    example_subset_num = int(0.05*example_num) if args.debug_mode else int(0.001*example_num)
    nli_subset_index = random.sample(np.arange(example_num).tolist(), example_subset_num)
    nli_subset_examples_ = [nli_examples_[i] for i in nli_subset_index]
    train_subset_data = SentencesDataset(examples=nli_subset_examples_, model=model)
    train_subset_dataloader = DataLoader(train_subset_data, shuffle=True, batch_size=args.batch_size)
    train_evaluator = LabelAccuracyEvaluator(train_subset_dataloader, softmax_model=train_loss)

    # The validation dataset
    logger.info("Read STSbenchmark dev dataset")
    sts_examples_ = sts_reader.get_examples('sts-dev.csv')[:100] if args.debug_mode else sts_reader.get_examples('sts-dev.csv')
    dev_data = SentencesDataset(examples=sts_examples_, model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
    eval_loss = losses.CosineSimilarityLoss(model=model)

    # Configure the training
    optimizer_params = {'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False}
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs / args.batch_size * 0.1) #10% of train data for warm-up
    logger.info("Epoch: {}, Warmup-steps: {}, learning_rate".format(args.num_epochs, warmup_steps, args.learning_rate))

    ## train_objectives enable the composite loss on the same model.
    model.fit(args=args,
            train_objectives=[(train_dataloader, train_loss)],
            train_evaluator=train_evaluator,
            evaluation_loss=eval_loss,
            evaluator=evaluator,
            eval_dataloader=dev_dataloader,
            epochs=args.num_epochs,
            evaluation_steps=750 if not args.debug_mode else 2,
            optimizer_params=optimizer_params,
            warmup_steps=warmup_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine tuning on COVID-Sci-Sentence-BERT')
    parser.add_argument('--finetuned_model_path', default='./word_embedding_saved/',
                        type=str, help="finetuned weight file.")
    parser.add_argument('--debug_mode', action='store_true', help="debug mode or not.")
    # parser.set_defaults(debug_mode=True)
    parser.add_argument('--batch_size', default=32, type=int, help="batch_size.")
    parser.add_argument('--data_directory', default='../sentence-bert-nli-stsb/datasets', 
                        type=str, help="the original dataset.")
    parser.add_argument('--num_epochs', default=1, type=int, help='training epoch.')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='learning rate.')
    parser.add_argument('--should_continue', action='store_true', help="continue to train.")
    # parser.set_defaults(should_continue=True)
    parser.add_argument('--output_dir', default='output', type=str, help="model output directory.")
    parser.add_argument('--save_total_limit', default=5, type=int, 
                        help="only save the last several checkpoints.")
    args = parser.parse_args()
    main(args)
