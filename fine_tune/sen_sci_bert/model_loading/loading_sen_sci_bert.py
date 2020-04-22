'''
to load sentence-sci-bert here.
'''
import transformers
import argparse

from SentenceTransformer_custom_expt import SentenceTransformer_tb

def main(finetuned_model_path):

	model_loaded = SentenceTransformer_tb(saved_scibert_model_path=finetuned_model_path)

	# test it on the demo sentence.
	sentence_input = ["Xiao needs to work on the fine-tuning part **&&>> 123.",
	                "the fine-tuning part needs to be done **&&>> 123."]
	sentence_embeddings = model_loaded.encode(sentence_input)
	for sentence, embedding in zip(sentence_input, sentence_embeddings):
	    print("Sentence:", sentence)
	    print("Corresponding Embedding's shape: ", embedding.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine tuning on COVID-Sci-Sentence-BERT')
    parser.add_argument('-fs', '--finetuned_sentence_model_path', default='./saved_model',
    					type=str, help="finetuned sentence-sci-bert weight.")
    args = parser.parse_args()
    main(finetuned_model_path=args.finetuned_sentence_model_path)
