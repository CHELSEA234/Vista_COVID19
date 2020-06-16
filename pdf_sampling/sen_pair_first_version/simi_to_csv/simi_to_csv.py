import csv
import os
import glob
import argparse
import transformers
import numpy as np
import sent2vec

from scipy import spatial
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from SentenceTransformer_custom_expt import SentenceTransformer_tb

sw = stopwords.words('english')

def consine_similarity_compute(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norma = np.linalg.norm(vec1)
    normb = np.linalg.norm(vec2)
    cos = dot / (norma * normb)
    return cos

def compute_similarities_bio_senc_vec(sentence_input, model_loaded):
    '''based on the performance.'''
    vec1 = model_loaded.embed_sentence(sentence_input[0])[0]
    vec2 = model_loaded.embed_sentence(sentence_input[1])[0]
    return consine_similarity_compute(vec1, vec2)

def compute_similarities_embedding(sentence_input, model_loaded):
    sentence_embeddings = model_loaded.encode(sentence_input)
    vec1 = sentence_embeddings[0]
    vec2 = sentence_embeddings[1]
    return consine_similarity_compute(vec1, vec2)

def compute_similarities(row):
    # tokenization
    X_list = word_tokenize(row[0])
    Y_list = word_tokenize(row[1])

    l1 = []
    l2 = []
    # remove stop words from string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}
    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    return consine_similarity_compute(l1, l2)

def main(args):

    print(f"loading the sentence embedding mode.")
    model_loaded = SentenceTransformer_tb(saved_scibert_model_path=args.finetuned_model_path)
    sent_model_loaded = sent2vec.Sent2vecModel()
    sent_model_loaded.load_model(args.senc_vect_model_path)

    file_list = glob.glob(f'../{args.requirement}*')
    for cur_file in file_list:
        if '_simi.csv' in cur_file:
            continue    # do not contain simi files.
        print(f"working on the {cur_file}...")
        cur_name = (cur_file.split('/')[1]).split('.')[0]
        cur_simi_name = cur_name + '_simi.csv'

        line_count = 0
        line_idx = 0
        read_file_handler = open(cur_file, mode='r')
        csv_reader = csv.reader(read_file_handler, delimiter=',')

        writer_file_handler = open(cur_simi_name, mode='w')
        csv_writer = csv.writer(writer_file_handler, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['line_index', 'surface_simi', 'embedding_simi', 'sen_embedding_simi'])

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                score_cur = compute_similarities(row)
                score_cur_embed = compute_similarities_embedding(row, model_loaded)
                score_cur_sen_embed = compute_similarities_bio_senc_vec(row, sent_model_loaded)
                csv_writer.writerow([line_idx, score_cur, score_cur_embed, score_cur_sen_embed])
                line_count += 1
                line_idx += 1

            if args.debug_mode:
                if line_count == 10:
                    break

        writer_file_handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading pdfs from three datasets in COVID_19.')
    parser.add_argument('-de', '--debug_mode', action='store_true', help="debug mode.")
    # parser.set_defaults(debug_mode=True)
    parser.add_argument('-r', '--requirement', default='consecutive', 
                        choices=['diff_document', 'document', "paragraph", "consecutive"],
                        help="to choose different requirement here.")
    parser.add_argument('-model_path', '--finetuned_model_path', 
                        default='/nas/vista-ssd02/users/xiaoguo/COVID_research/sentence-bert-nli-stsb/saved_model')
    parser.add_argument('--senc_vect_model_path',
                        default="/nas/vista-ssd02/users/xiaoguo/COVID_research/intermediate_code/background_demo/sent2vec/pretrained_model/wiki_unigrams.bin")
    args = parser.parse_args()
    main(args)
