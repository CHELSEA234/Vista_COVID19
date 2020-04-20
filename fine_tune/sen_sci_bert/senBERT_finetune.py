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
import logging
from datetime import datetime

class SciBERT(models.BERT):
    """SciBERT model to generate token embeddings.

    Each token is mapped to an output vector from SciBERT.
    """
    def __init__(self, SciBERT_model, SciBERT_tokenizer, max_seq_length: int = 128, do_lower_case: bool = True):
        super(models.BERT, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logging.warning("BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length

        self.bert = SciBERT_model
        self.tokenizer = SciBERT_tokenizer
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

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

def loading_pretrained_model(finetuned_model_path):
    '''loading pretrained SciBERT.'''
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    config = AutoConfig.from_pretrained(finetuned_model_path, output_hidden_states=True)
    model = AutoModelWithLMHead.from_pretrained(finetuned_model_path, config=config)
    covid_scibert_model = model.bert
    return tokenizer, covid_scibert_model

def sentence_bert_initialization(covid_scibert_model, tokenizer):
    '''Use BERT for mapping tokens to embeddings'''
    word_embedding_model = SciBERT(covid_scibert_model, tokenizer)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer_tb(modules=[word_embedding_model, pooling_model])
    return model

def main(finetuned_model_path, debug_mode, batch_size, data_directory, num_epochs, learning_rate,
        logging_option):
    
    if logging_option:
        logging_starter()

    tokenizer, covid_scibert_model = loading_pretrained_model(finetuned_model_path)
    model = sentence_bert_initialization(covid_scibert_model, tokenizer)

    # Read the dataset
    nli_reader = NLIDataReader(os.path.join(data_directory, 'AllNLI'))
    sts_reader = STSDataReader(os.path.join(data_directory, 'stsbenchmark'))
    train_num_labels = nli_reader.get_num_labels()
    model_save_path = 'output/training_nli_sci_bert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Convert the dataset to a DataLoader ready for training
    # 942069 for training, around 1000 for monitor, 1500 for evaluation 
    logging.info("Read AllNLI train dataset")
    nli_examples_ = nli_reader.get_examples('train.gz')[:500] if debug_mode else nli_reader.get_examples('train.gz')
    train_data = SentencesDataset(examples=nli_examples_, model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.SoftmaxLoss(model=model, 
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
                                    num_labels=train_num_labels)

    ## to measure the accuracy over a small portion of training data.
    example_num = len(nli_examples_)
    example_subset_num = int(0.05*example_num) if debug_mode else int(0.001*example_num)
    nli_subset_index = random.sample(np.arange(example_num).tolist(), example_subset_num)
    nli_subset_examples_ = [nli_examples_[i] for i in nli_subset_index]
    train_subset_data = SentencesDataset(examples=nli_subset_examples_, model=model)
    train_subset_dataloader = DataLoader(train_subset_data, shuffle=True, batch_size=batch_size)
    train_evaluator = LabelAccuracyEvaluator(train_subset_dataloader, softmax_model=train_loss)

    # The validation dataset
    logging.info("Read STSbenchmark dev dataset")
    sts_examples_ = sts_reader.get_examples('sts-dev.csv')[:100] if debug_mode else sts_reader.get_examples('sts-dev.csv')
    dev_data = SentencesDataset(examples=sts_examples_, model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
    eval_loss = losses.CosineSimilarityLoss(model=model)

    # Configure the training
    optimizer_params = {'lr': learning_rate, 'eps': 1e-6, 'correct_bias': False}
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
    logging.info("Epoch: {}, Warmup-steps: {}, learning_rate".format(num_epochs, warmup_steps, learning_rate))

    ## train_objectives enable the composite loss on the same model.
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            train_evaluator=train_evaluator,
            evaluation_loss=eval_loss,
            evaluator=evaluator,
            eval_dataloader=dev_dataloader,
            epochs=num_epochs,
            evaluation_steps=500 if not debug_mode else 2,
            optimizer_params=optimizer_params,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine tuning on COVID-Sci-Sentence-BERT')
    parser.add_argument('--finetuned_model_path', default='../SciBERT_learning/expts/model_version1_Apr_15th/checkpoint-best',
                        type=str, help="finetuned weight file.")
    parser.add_argument('--debug_mode', action='store_true', help="debug mode or not.")
    parser.set_defaults(debug_mode=True)
    parser.add_argument('--batch_size', default=15, type=int, help="batch_size.")
    parser.add_argument('--data_directory', default='./sentence-transformers/examples/datasets', 
                        type=str, help="the original dataset.")
    parser.add_argument('--num_epochs', default=1, type=int, help='training epoch.')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate.')
    parser.add_argument('--logging_option', action='store_true', help="open the custom logging handler.")
    parser.set_defaults(logging_option=False)
    args = parser.parse_args()
    main(finetuned_model_path=args.finetuned_model_path, debug_mode=args.debug_mode, batch_size=args.batch_size,
        data_directory=args.data_directory, num_epochs=args.num_epochs, learning_rate=args.lr, 
        logging_option=args.logging_option)
